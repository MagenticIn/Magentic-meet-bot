"""
Magentic-meetbot  ·  transcribe.py
──────────────────────────────────
Transcribe meeting audio using **faster-whisper** with automatic
Hindi + English language detection.

Features
────────
- Model: ``large-v3-turbo`` — auto-selects CUDA/float16 or CPU/int8.
- Language: auto-detect (``language=None``), preserves Hindi text as-is
  (``task="transcribe"``, NOT ``"translate"``).
- Word-level timestamps via ``word_timestamps=True``.
- VAD-filtered silence removal.
- Hindi→English translation helper via Helsinki-NLP/opus-mt-hi-en.

CLI usage
─────────
    python -m pipeline.transcribe /data/recording.wav
    python -m pipeline.transcribe /data/recording.wav --translate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import structlog
import torch
from faster_whisper import WhisperModel

log = structlog.get_logger("transcribe")

# ─── Device detection ────────────────────────────────────────────────────
CUDA_AVAILABLE: bool = torch.cuda.is_available()
_ENV_DEVICE: str = os.getenv("WHISPER_DEVICE", "").strip().lower()
_AUTO_DEVICE: str = "cuda" if CUDA_AVAILABLE else "cpu"
if _ENV_DEVICE in {"cpu", "cuda"}:
    DEVICE: str = _ENV_DEVICE
else:
    DEVICE = _AUTO_DEVICE
if DEVICE == "cuda" and not CUDA_AVAILABLE:
    DEVICE = "cpu"

_DEFAULT_COMPUTE = "float16" if DEVICE == "cuda" else "int8"
COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", _DEFAULT_COMPUTE)
MODEL_NAME: str = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "").strip().lower()

# ─── Lazy-loaded singletons ─────────────────────────────────────────────
_whisper_model: Optional[WhisperModel] = None
_translation_pipeline: Optional[Any] = None


# ═════════════════════════════════════════════════════════════════════════
#  Whisper model
# ═════════════════════════════════════════════════════════════════════════

def _get_whisper_model() -> WhisperModel:
    """
    Load the faster-whisper model (singleton).

    - large-v3-turbo on CUDA with float16
    - large-v3-turbo on CPU with int8
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    log.info(
        "whisper.loading_model",
        model=MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cuda_available=CUDA_AVAILABLE,
    )
    t0 = time.monotonic()

    _whisper_model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    elapsed = time.monotonic() - t0
    log.info("whisper.model_loaded", elapsed_sec=f"{elapsed:.1f}")
    return _whisper_model


# ═════════════════════════════════════════════════════════════════════════
#  Transcription
# ═════════════════════════════════════════════════════════════════════════

def transcribe(audio_path: str) -> list[dict]:
    """
    Transcribe an audio file and return a list of segment dicts.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (WAV recommended, 16 kHz mono).

    Returns
    -------
    list[dict]
        Each dict contains::

            {
                "start": 0.0,         # seconds
                "end": 3.52,          # seconds
                "text": "नमस्ते, how are you?",   # raw transcript (Hindi preserved)
                "language": "hi",     # detected language for this segment
                "words": [
                    {"word": "नमस्ते,", "start": 0.0, "end": 0.8, "probability": 0.94},
                    {"word": " how",    "start": 0.9, "end": 1.1, "probability": 0.97},
                    ...
                ]
            }

    Notes
    -----
    - ``language=None`` lets the model auto-detect per-segment, handling
      Hindi ↔ English code-switching seamlessly.
    - ``task="transcribe"`` preserves Hindi Devanagari text.  We do NOT
      use ``task="translate"`` here — translation is a separate step via
      ``translate_segment()``.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    log.info("transcribe.start", path=str(path), size_mb=f"{file_size_mb:.1f}")

    model = _get_whisper_model()
    t0 = time.monotonic()

    # ── Run transcription ────────────────────────────────────────────
    # By default, language is auto-detected. For Hindi-first meetings, set:
    #   WHISPER_LANGUAGE=hi
    # This helps preserve Hindi lines in Devanagari instead of English-biased output.
    language_arg = WHISPER_LANGUAGE or None
    if language_arg:
        log.info("transcribe.language_override", language=language_arg)

    segments_iter, info = model.transcribe(
        str(path),
        beam_size=5,
        language=language_arg,                  # auto-detect unless overridden
        task="transcribe",                      # preserve original language text
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    log.info(
        "transcribe.detection_info",
        detected_language=info.language,
        language_probability=f"{info.language_probability:.2%}",
        audio_duration=f"{info.duration:.1f}s",
    )

    # ── Collect segments ─────────────────────────────────────────────
    segments: list[dict] = []
    total_words = 0

    for seg in segments_iter:
        # Build word-level data
        words: list[dict] = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 4),
                })
            total_words += len(words)

        # Determine the segment's language.
        # faster-whisper reports the dominant language in `info.language`,
        # but individual segments can differ.  We use the segment's own
        # language field when available; fall back to the global detection.
        segment_language = getattr(seg, "language", None) or info.language

        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "language": segment_language,
            "words": words,
        })

    elapsed = time.monotonic() - t0
    rtf = elapsed / info.duration if info.duration > 0 else 0

    log.info(
        "transcribe.done",
        segments=len(segments),
        words=total_words,
        elapsed_sec=f"{elapsed:.1f}",
        realtime_factor=f"{rtf:.2f}x",
        audio_duration=f"{info.duration:.1f}s",
    )

    return segments


# ═════════════════════════════════════════════════════════════════════════
#  Hindi → English translation helper
# ═════════════════════════════════════════════════════════════════════════

def _get_translation_pipeline():
    """
    Load the Helsinki-NLP/opus-mt-hi-en translation model (singleton).

    Uses HuggingFace ``transformers.pipeline`` under the hood.
    The model is downloaded on first use and cached by transformers.
    """
    global _translation_pipeline

    if _translation_pipeline is not None:
        return _translation_pipeline

    log.info("translation.loading_model", model="Helsinki-NLP/opus-mt-hi-en")
    t0 = time.monotonic()

    # Import here to avoid loading transformers if translation isn't used
    from transformers import pipeline as hf_pipeline

    _translation_pipeline = hf_pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-hi-en",
        device=0 if CUDA_AVAILABLE else -1,
    )

    elapsed = time.monotonic() - t0
    log.info("translation.model_loaded", elapsed_sec=f"{elapsed:.1f}")
    return _translation_pipeline


def translate_segment(text: str, source_lang: str) -> str:
    """
    Translate a single text segment from Hindi to English.

    Parameters
    ----------
    text : str
        The text to translate (expected to be Hindi / Devanagari).
    source_lang : str
        ISO 639-1 language code.  Only ``"hi"`` triggers translation;
        all other languages return the text unchanged.

    Returns
    -------
    str
        English translation if source_lang is ``"hi"``, otherwise the
        original text unchanged.

    Examples
    --------
    >>> translate_segment("नमस्ते, आप कैसे हैं?", "hi")
    "Hello, how are you?"

    >>> translate_segment("This is English", "en")
    "This is English"
    """
    # Only translate Hindi — everything else passes through
    if source_lang != "hi":
        return text

    # Skip empty / whitespace-only text
    if not text or not text.strip():
        return text

    pipe = _get_translation_pipeline()

    try:
        result = pipe(text, max_length=512)
        if result and isinstance(result, list) and "translation_text" in result[0]:
            translated = result[0]["translation_text"]
            log.debug(
                "translation.translated",
                source=text[:80],
                target=translated[:80],
            )
            return translated
        log.warning("translation.unexpected_result", result=result)
        return text
    except Exception as exc:
        log.error("translation.failed", text=text[:80], error=str(exc))
        return text


def translate_segments(segments: list[dict]) -> list[dict]:
    """
    Add an ``"english_text"`` field to each segment where the detected
    language is Hindi.  English segments get ``english_text = text``.

    This is a convenience wrapper around ``translate_segment()`` for
    batch processing an entire transcript.

    Parameters
    ----------
    segments : list[dict]
        Output of ``transcribe()``.

    Returns
    -------
    list[dict]
        Same segments, each augmented with an ``"english_text"`` key.
    """
    log.info("translation.batch_start", total_segments=len(segments))
    hi_count = 0

    for seg in segments:
        lang = seg.get("language", "en")
        if lang == "hi":
            seg["english_text"] = translate_segment(seg["text"], lang)
            hi_count += 1
        else:
            seg["english_text"] = seg["text"]

    log.info("translation.batch_done", hindi_translated=hi_count, total=len(segments))
    return segments


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Magentic-meetbot — Transcribe audio with faster-whisper",
    )
    parser.add_argument(
        "audio_path",
        help="Path to the audio file (WAV, MP3, etc.)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        default=False,
        help="Also translate Hindi segments to English (adds 'english_text' field)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="Pretty-print the JSON output",
    )
    args = parser.parse_args()

    # ── Transcribe ───────────────────────────────────────────────────
    segments = transcribe(args.audio_path)

    # ── Optionally translate Hindi ───────────────────────────────────
    if args.translate:
        segments = translate_segments(segments)

    # ── Print as JSON ────────────────────────────────────────────────
    indent = 2 if args.pretty else None
    output = json.dumps(segments, ensure_ascii=False, indent=indent)
    print(output)

    # ── Summary to stderr ────────────────────────────────────────────
    total_dur = segments[-1]["end"] - segments[0]["start"] if segments else 0
    langs = {}
    for s in segments:
        lang = s.get("language", "??")
        langs[lang] = langs.get(lang, 0) + 1

    print(
        f"\n--- Summary ---\n"
        f"Segments : {len(segments)}\n"
        f"Duration : {total_dur:.1f}s\n"
        f"Languages: {langs}\n"
        f"Device   : {DEVICE} ({COMPUTE_TYPE})\n"
        f"Model    : {MODEL_NAME}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
