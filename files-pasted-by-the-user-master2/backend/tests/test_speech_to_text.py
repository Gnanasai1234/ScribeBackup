from __future__ import annotations

import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import Mock, patch

from services.speech_to_text import (
    AudioRecordingEmpty,
    InvalidAudioRecording,
    NoTranscription,
    transcribe_local_audio,
)


def valid_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000)
    return buffer.getvalue()


class SpeechToTextTests(unittest.TestCase):
    @patch("services.speech_to_text._get_model")
    def test_keeps_nonempty_text_even_when_whisper_marks_high_no_speech_probability(self, get_model):
        segment = SimpleNamespace(text="  Clear speech was recognized.  ", no_speech_prob=0.98)
        model = Mock()
        model.transcribe.return_value = ([segment], SimpleNamespace(language="en"))
        get_model.return_value = model

        result = transcribe_local_audio(valid_wav(), content_type="audio/wav", language="en")

        self.assertEqual(result["transcript"], "Clear speech was recognized.")
        self.assertEqual(result["language"], "en")
        self.assertAlmostEqual(result["duration_seconds"], 1.0, places=1)
        model.transcribe.assert_called_once()
        kwargs = model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["language"], "en")
        self.assertFalse(kwargs["vad_filter"])

    @patch("services.speech_to_text._get_model")
    def test_passes_telugu_language_to_local_model(self, get_model):
        model = Mock()
        model.transcribe.return_value = (
            [SimpleNamespace(text="తెలుగు మాటలు", no_speech_prob=0.1)],
            SimpleNamespace(language="te"),
        )
        get_model.return_value = model

        result = transcribe_local_audio(valid_wav(), content_type="audio/wav", language="te")

        self.assertEqual(result["language"], "te")
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "te")

    @patch("services.speech_to_text._get_model")
    def test_passes_tamil_language_to_local_model(self, get_model):
        model = Mock()
        model.transcribe.return_value = (
            [SimpleNamespace(text="தமிழ் பேச்சு", no_speech_prob=0.1)],
            SimpleNamespace(language="ta"),
        )
        get_model.return_value = model

        result = transcribe_local_audio(valid_wav(), content_type="audio/wav", language="ta")

        self.assertEqual(result["language"], "ta")
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "ta")

    def test_empty_bytes_have_specific_error(self):
        with self.assertRaisesRegex(AudioRecordingEmpty, "Audio recording was empty"):
            transcribe_local_audio(b"", content_type="audio/webm", language="en")

    def test_invalid_bytes_have_specific_error(self):
        with self.assertRaises(InvalidAudioRecording):
            transcribe_local_audio(b"not an audio file", content_type="audio/wav", language="en")

    @patch("services.speech_to_text._get_model")
    def test_empty_decoder_output_has_specific_error(self, get_model):
        model = Mock()
        model.transcribe.return_value = ([], SimpleNamespace(language="en"))
        get_model.return_value = model

        with self.assertRaisesRegex(NoTranscription, "returned no transcription"):
            transcribe_local_audio(valid_wav(), content_type="audio/wav", language="en")


if __name__ == "__main__":
    unittest.main()
