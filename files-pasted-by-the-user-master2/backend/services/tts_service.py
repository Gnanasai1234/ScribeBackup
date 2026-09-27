from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

LOGGER = logging.getLogger("scribe.tts")

VOICE_OPTIONS: dict[str, list[dict[str, str]]] = {
    "en": [
        {"voice_id": "en-US-AriaNeural", "name": "Aria (US English)"},
        {"voice_id": "en-US-GuyNeural", "name": "Guy (US English)"},
        {"voice_id": "en-IN-NeerjaNeural", "name": "Neerja (Indian English)"},
        {"voice_id": "en-IN-PrabhatNeural", "name": "Prabhat (Indian English)"},
    ],
    "hi": [
        {"voice_id": "hi-IN-SwaraNeural", "name": "Swara (Hindi)"},
        {"voice_id": "hi-IN-MadhurNeural", "name": "Madhur (Hindi)"},
    ],
    "te": [
        {"voice_id": "te-IN-ShrutiNeural", "name": "Shruti (Telugu)"},
        {"voice_id": "te-IN-MohanNeural", "name": "Mohan (Telugu)"},
    ],
    "ta": [
        {"voice_id": "ta-IN-PallaviNeural", "name": "Pallavi (Tamil)"},
        {"voice_id": "ta-IN-ValluvarNeural", "name": "Valluvar (Tamil)"},
    ],
    "kn": [
        {"voice_id": "kn-IN-SapnaNeural", "name": "Sapna (Kannada)"},
        {"voice_id": "kn-IN-GaganNeural", "name": "Gagan (Kannada)"},
    ],
    "ml": [
        {"voice_id": "ml-IN-SobhanaNeural", "name": "Sobhana (Malayalam)"},
        {"voice_id": "ml-IN-MidhunNeural", "name": "Midhun (Malayalam)"},
    ],
    "bn": [
        {"voice_id": "bn-IN-TanishaaNeural", "name": "Tanishaa (Bengali)"},
        {"voice_id": "bn-IN-BashkarNeural", "name": "Bashkar (Bengali)"},
    ],
    "mr": [
        {"voice_id": "mr-IN-AarohiNeural", "name": "Aarohi (Marathi)"},
        {"voice_id": "mr-IN-ManoharNeural", "name": "Manohar (Marathi)"},
    ],
    "gu": [
        {"voice_id": "gu-IN-DhwaniNeural", "name": "Dhwani (Gujarati)"},
        {"voice_id": "gu-IN-NiranjanNeural", "name": "Niranjan (Gujarati)"},
    ],
    "ur": [
        {"voice_id": "ur-IN-GulNeural", "name": "Gul (Urdu)"},
        {"voice_id": "ur-IN-SalmanNeural", "name": "Salman (Urdu)"},
    ],
}


def voices_for(language: str) -> list[dict[str, str]]:
    return VOICE_OPTIONS.get(language, [])


def get_default_voice_for(language: str) -> str | None:
    voices = voices_for(language)
    return voices[0]["voice_id"] if voices else None


def split_into_sentences(text: str) -> list[str]:
    """
    Split multilingual text into clean sentences/clauses.
    Handles Western (., !, ?), Devanagari/Bengali (।, ॥), Arabic/Urdu (۔, ؟),
    and newline paragraph boundaries.
    """
    if not text or not text.strip():
        return []
    # Pattern recognizes sentences followed by whitespace/newlines or string end
    pattern = r'[^.!?।॥۔؟\n]+(?:[.!?।॥۔؟]+|\n+|$)'
    raw_matches = re.findall(pattern, text)
    sentences = [s.strip() for s in raw_matches if s.strip()]
    return sentences if sentences else [text.strip()]


def estimate_timings(text: str, total_duration: float | None = None) -> list[dict[str, Any]]:
    """
    Fallback synchronization timing calculation.
    Calculates proportional start/end timestamps for each sentence based on length and pauses.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Weights proportional to speech duration (character length + pause per sentence)
    weights = [max(1, len(s)) + 12 for s in sentences]
    total_weight = sum(weights) or 1

    # If total_duration is not known, estimate ~14 characters per second of speech
    est_duration = total_duration if (total_duration and total_duration > 0) else sum(max(1.2, len(s) / 14.0) for s in sentences)

    timings: list[dict[str, Any]] = []
    current_time = 0.0

    for idx, (sent, weight) in enumerate(zip(sentences, weights)):
        duration = round((weight / total_weight) * est_duration, 3)
        start = round(current_time, 3)
        end = round(current_time + duration, 3)
        timings.append({
            "index": idx,
            "start": start,
            "end": end,
            "text": sent,
        })
        current_time = end

    return timings


async def _synthesize_async(text: str, voice_id: str, rate: float) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Online neural voices are not installed. Run pip install -r backend/requirements.txt.") from exc

    rate_percent = round((rate - 1.0) * 100)
    LOGGER.info("Starting edge-tts synthesis: voice=%s, rate=%+d%%, text_length=%d", voice_id, rate_percent, len(text))
    communicate = edge_tts.Communicate(text, voice_id, rate=f"{rate_percent:+d}%")
    audio_parts: list[bytes] = []
    timings: list[dict[str, Any]] = []

    async for chunk in communicate.stream():
        chunk_type = chunk.get("type")
        if chunk_type == "audio":
            audio_parts.append(chunk["data"])
        elif chunk_type == "SentenceBoundary":
            # offset and duration are in 100ns (ticks)
            offset_sec = chunk.get("offset", 0) / 10_000_000.0
            dur_sec = chunk.get("duration", 0) / 10_000_000.0
            timings.append({
                "index": len(timings),
                "start": round(offset_sec, 3),
                "end": round(offset_sec + dur_sec, 3),
                "text": chunk.get("text", "").strip(),
            })

    audio = b"".join(audio_parts)
    if not audio:
        raise RuntimeError("The online speech provider returned no audio. Check your internet connection and try again.")

    # Fallback to estimated timings if edge_tts emitted audio but no SentenceBoundary events
    if not timings and text.strip():
        LOGGER.info("No SentenceBoundary events received from edge-tts; applying estimated sentence timings.")
        timings = estimate_timings(text)

    LOGGER.info("edge-tts synthesis completed successfully: audio_bytes=%d, sentence_timings_count=%d", len(audio), len(timings))
    return audio, timings


def synthesize_with_timings(text: str, voice_id: str, rate: float = 1.0) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        return asyncio.run(_synthesize_async(text, voice_id, rate))
    except RuntimeError:
        raise
    except Exception as exc:
        LOGGER.warning("Online speech generation failed for voice %s: %s", voice_id, exc)
        raise RuntimeError("Online speech generation failed. Check your internet connection and try again.") from exc


def synthesize(text: str, voice_id: str, rate: float = 1.0) -> bytes:
    audio, _ = synthesize_with_timings(text, voice_id, rate)
    return audio
