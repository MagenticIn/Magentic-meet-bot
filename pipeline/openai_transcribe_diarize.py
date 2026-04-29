"""
Magentic-meetbot  ·  openai_transcribe_diarize.py
─────────────────────────────────────────────────
Single-call transcription + speaker diarization using OpenAI
``gpt-4o-transcribe-diarize`` (Transcription API, ``diarized_json``).

Replaces the faster-whisper + whisperx/pyannote chain when
``TRANSCRIPTION_BACKEND=openai``.

By default the API ``language`` parameter is **omitted** so one meeting can
mix **Hindi and English** without forcing a single language. Optional
``OPENAI_TRANSCRIPTION_LANGUAGE`` biases the *whole file* when needed.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import structlog
from openai import OpenAI

from pipeline.hi_en_translate import translate_segment

log = structlog.get_logger("openai_transcribe_diarize")

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def _normalize_text(text: str) -> str:
    """NFC so Devanagari combining marks match reliably."""
    return unicodedata.normalize("NFC", text or "")


def _segment_language(text: str) -> str:
    """
    Per-segment label for translation / UI.

    Devanagari → Hindi (machine translation to English).
    Latin-only → English for pipeline purposes; Roman Hindi / Hinglish stays
    as-is in ``text`` and ``text_en``, and the summariser still understands it.
    """
    if _DEVANAGARI.search(_normalize_text(text)):
        return "hi"
    return "en"


def _openai_request_language() -> str | None:
    """
    Optional ISO hint for the *entire* audio file.

    Omit (return None) so the model can auto-handle **mixed Hindi + English**
    in one meeting. Use ``OPENAI_TRANSCRIPTION_LANGUAGE=hi`` or ``en`` only for
    single-language rooms where you want to bias recognition.

    ``WHISPER_LANGUAGE`` is **not** read here — it applies only to the local
    ``TRANSCRIPTION_BACKEND=whisper`` path.
    """
    raw = os.environ.get("OPENAI_TRANSCRIPTION_LANGUAGE", "").strip().lower()
    return raw if raw in {"hi", "en"} else None


def _seg_to_dict(seg: Any) -> dict[str, Any]:
    if isinstance(seg, dict):
        return seg
    if hasattr(seg, "model_dump"):
        return seg.model_dump()
    return {
        "id": getattr(seg, "id", None),
        "speaker": getattr(seg, "speaker", None),
        "start": getattr(seg, "start", 0.0),
        "end": getattr(seg, "end", 0.0),
        "text": getattr(seg, "text", "") or "",
        "type": getattr(seg, "type", None),
    }


def _response_to_dict(resp: Any) -> dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "model_dump_json"):
        return json.loads(resp.model_dump_json())
    text = getattr(resp, "text", "") or ""
    segments = getattr(resp, "segments", None) or []
    return {"text": text, "segments": [_seg_to_dict(s) for s in segments]}


def _speaker_label_map(segment_dicts: list[dict[str, Any]]) -> dict[str, str]:
    """Map API speaker labels (A, B, … or custom names) to SPEAKER_00, …"""
    mapping: dict[str, str] = {}
    for d in segment_dicts:
        raw = (d.get("speaker") or "UNKNOWN") or "UNKNOWN"
        if raw not in mapping:
            mapping[raw] = f"SPEAKER_{len(mapping):02d}"
    return mapping


def transcribe_diarize_openai(audio_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Transcribe and diarize in one API call.

    Returns
    -------
    utterances
        Same shape as ``diarize.diarize()`` output for ``summarize``.
    raw
        Full serialisable API payload for artifact ``01_transcript_raw``.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY is required when TRANSCRIPTION_BACKEND=openai")

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    api_language = _openai_request_language()

    client = OpenAI(api_key=key)

    create_kwargs: dict[str, Any] = {
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        # Required for diarize model when audio is longer than ~30s (OpenAI docs).
        "chunking_strategy": "auto",
    }
    if api_language:
        create_kwargs["language"] = api_language

    log.info(
        "openai_td.start",
        path=str(path),
        request_language=api_language or "auto_multilingual",
    )

    with open(path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            file=audio_file,
            **create_kwargs,
        )

    raw = _response_to_dict(resp)
    raw_segments = [_seg_to_dict(s) for s in (raw.get("segments") or [])]
    raw["segments"] = raw_segments

    speaker_map = _speaker_label_map(raw_segments)
    utterances: list[dict[str, Any]] = []

    for d in raw_segments:
        text = _normalize_text((d.get("text") or "").strip())
        if not text:
            continue
        raw_spk = (d.get("speaker") or "UNKNOWN") or "UNKNOWN"
        speaker = speaker_map.get(raw_spk, "SPEAKER_00")
        start = float(d.get("start", 0.0))
        end = float(d.get("end", start))
        lang = _segment_language(text)
        # Hindi (Devanagari): local MT to English. English / Roman Hinglish: pass through.
        text_en = translate_segment(text, lang) if lang == "hi" else text

        utterances.append({
            "speaker": speaker,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "text_en": text_en,
            "language": lang,
        })

    log.info(
        "openai_td.done",
        utterances=len(utterances),
        speakers=len(speaker_map),
    )
    return utterances, raw
