"""Offline speech transcription using an already-cached Faster Whisper model."""
from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

LOGGER = logging.getLogger("scribe.speech_to_text")


class SpeechModelUnavailable(RuntimeError):
    """The local speech model or its runtime is not ready."""


class AudioRecordingEmpty(ValueError):
    """The submitted recording has no audio bytes or playable duration."""


class InvalidAudioRecording(ValueError):
    """The uploaded bytes are not a supported, decodable audio recording."""


class NoTranscription(ValueError):
    """The recognizer completed but returned no usable words."""


@lru_cache(maxsize=1)
def _get_model():
    model_name = os.getenv("LOCAL_STT_MODEL", "base").strip() or "base"
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SpeechModelUnavailable("Install the backend speech-to-text requirements.") from exc
    try:
        return WhisperModel(
            model_name,
            device=os.getenv("LOCAL_STT_DEVICE", "cpu").strip() or "cpu",
            compute_type=os.getenv("LOCAL_STT_COMPUTE_TYPE", "int8").strip() or "int8",
            local_files_only=True,
        )
    except Exception as exc:
        LOGGER.warning("Local speech model unavailable model=%s type=%s", model_name, type(exc).__name__)
        raise SpeechModelUnavailable(f"The local speech model '{model_name}' is not cached on this device.") from exc


def _validate_audio(path: Path) -> float:
    """Decode container metadata before inference; don't use loudness as a speech gate."""
    try:
        import av

        with av.open(str(path)) as container:
            streams = [stream for stream in container.streams if stream.type == "audio"]
            if not streams:
                raise InvalidAudioRecording("Audio recording was invalid: no audio track was found.")
            stream = streams[0]
            duration = float(stream.duration * stream.time_base) if stream.duration and stream.time_base else 0.0
            if duration <= 0 and container.duration:
                duration = float(container.duration) / av.time_base
            if duration <= 0:
                decoded_samples = 0
                sample_rate = 0
                for frame in container.decode(stream):
                    decoded_samples += int(getattr(frame, "samples", 0) or 0)
                    sample_rate = int(getattr(frame, "sample_rate", 0) or sample_rate)
                if decoded_samples and sample_rate:
                    duration = decoded_samples / sample_rate
            if duration < 0.05:
                raise AudioRecordingEmpty("Audio recording was empty or too short to decode.")
            return duration
    except (AudioRecordingEmpty, InvalidAudioRecording):
        raise
    except Exception as exc:
        raise InvalidAudioRecording("Audio recording was invalid or could not be decoded.") from exc


def transcribe_local_audio(audio: bytes, *, content_type: str | None, language: str = "auto") -> dict[str, object]:
    """Transcribe uploaded audio locally without contacting an external speech API."""
    if not audio:
        raise AudioRecordingEmpty("Audio recording was empty.")

    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    suffixes = {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mp4": ".mp4",
        "audio/mpeg": ".mp3",
    }
    suffix = suffixes.get(media_type, ".webm")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="scribe-stt-", suffix=suffix, delete=False) as temp_audio:
            temp_audio.write(audio)
            temp_path = Path(temp_audio.name)

        duration = _validate_audio(temp_path)
        model = _get_model()
        selected_language = language.strip().lower()
        if selected_language == "auto" or len(selected_language) != 2 or not selected_language.isalpha():
            selected_language = None
        LOGGER.info(
            "STT request engine=faster-whisper model=%s language=%s duration_seconds=%.2f audio_bytes=%d mime=%s",
            os.getenv("LOCAL_STT_MODEL", "base").strip() or "base",
            selected_language or "auto",
            duration,
            len(audio),
            media_type or "unknown",
        )
        segments, info = model.transcribe(
            str(temp_path),
            language=selected_language,
            beam_size=3,
            # VAD can discard quiet starts/ends and pauses inside ordinary dictation.
            # Whisper's own decoder handles silence; only truly empty output is rejected.
            vad_filter=False,
            condition_on_previous_text=False,
        )
        segments = list(segments)
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        if not transcript:
            raise NoTranscription("Speech-to-text returned no transcription.")

        probabilities = [
            float(getattr(segment, "no_speech_prob"))
            for segment in segments
            if getattr(segment, "no_speech_prob", None) is not None
        ]
        LOGGER.info(
            "STT result engine=faster-whisper language=%s text_length=%d segments=%d mean_no_speech_probability=%s",
            getattr(info, "language", "unknown") or "unknown",
            len(transcript),
            len(segments),
            round(sum(probabilities) / len(probabilities), 3) if probabilities else "unavailable",
        )
        return {
            "transcript": transcript,
            "language": getattr(info, "language", "unknown") or "unknown",
            "model": os.getenv("LOCAL_STT_MODEL", "base").strip() or "base",
            "duration_seconds": round(duration, 2),
            "segments": len(segments),
        }
    except (AudioRecordingEmpty, InvalidAudioRecording, NoTranscription, SpeechModelUnavailable):
        raise
    except Exception as exc:
        LOGGER.warning("Local transcription failed type=%s", type(exc).__name__)
        raise RuntimeError("The local speech recognizer could not process this recording.") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.info("Could not remove temporary audio recording")
