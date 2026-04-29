"""
Magentic-meetbot  ·  hi_en_translate.py
───────────────────────────────────────
Hindi → English translation via Helsinki-NLP/opus-mt-hi-en.

Kept separate from ``transcribe.py`` so the OpenAI ASR path can use
translation without importing faster-whisper / torch CUDA stack at
module import time (lazy torch import remains inside the loader).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import structlog
import torch

log = structlog.get_logger("hi_en_translate")

CUDA_AVAILABLE: bool = torch.cuda.is_available()
_translation_pipeline: Optional[Any] = None


def _get_translation_pipeline():
    """
    Load the Helsinki-NLP/opus-mt-hi-en translation model (singleton).

    Uses HuggingFace ``transformers.pipeline`` under the hood.
    """
    global _translation_pipeline

    if _translation_pipeline is not None:
        return _translation_pipeline

    log.info("translation.loading_model", model="Helsinki-NLP/opus-mt-hi-en")
    t0 = time.monotonic()

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

    Only ``source_lang == "hi"`` triggers translation; otherwise the
    original text is returned unchanged.
    """
    if source_lang != "hi":
        return text

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
