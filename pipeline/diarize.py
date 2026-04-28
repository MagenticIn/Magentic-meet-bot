"""
Magentic-meetbot  ·  diarize.py
───────────────────────────────
Speaker diarization using **whisperx** — combines faster-whisper
transcription segments with pyannote-based speaker identification.

Pipeline
────────
1.  Load audio via whisperx.
2.  Load a phoneme alignment model for the dominant language.
3.  Align Whisper segments to get precise word-level timestamps.
4.  Run pyannote speaker diarization (requires HF_TOKEN).
5.  Assign speaker labels to each word.
6.  Merge consecutive words from the same speaker into clean utterances.
7.  Translate Hindi utterances to English via ``transcribe.translate_segment``.

Graceful degradation
────────────────────
If diarization fails (missing HF_TOKEN, model download error, OOM, etc.)
the function returns segments with ``speaker = "UNKNOWN"`` so downstream
processing (summarisation, PM sync) can still proceed.

CLI usage
─────────
    python -m pipeline.diarize /data/recording.wav --transcript /data/transcript.json
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
import whisperx

from pipeline.transcribe import translate_segment

log = structlog.get_logger("diarize")

# ─── Config ──────────────────────────────────────────────────────────────
CUDA_AVAILABLE: bool = torch.cuda.is_available()
DEVICE: str = "cuda" if CUDA_AVAILABLE else "cpu"


# ═════════════════════════════════════════════════════════════════════════
#  Language helpers
# ═════════════════════════════════════════════════════════════════════════

def _detect_primary_language(segments: list[dict]) -> str:
    """
    Determine the dominant language across all segments by majority vote.
    Falls back to ``"en"`` if no language info is present.
    """
    counts: dict[str, int] = {}
    for seg in segments:
        lang = seg.get("language", "en")
        counts[lang] = counts.get(lang, 0) + 1

    if not counts:
        return "en"

    primary = max(counts, key=lambda k: counts[k])
    log.info("diarize.language_detected", primary=primary, distribution=counts)
    return primary


# whisperx align model supports a specific set of languages.
# If the detected language isn't in this set, fall back to "en".
_WHISPERX_ALIGN_LANGUAGES = {
    "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt",
    "ar", "cs", "ru", "pl", "hu", "fi", "fa", "el", "tr", "da",
    "he", "vi", "ko", "ur", "te", "hi", "ta", "th", "id", "tl",
}


def _resolve_align_language(detected: str) -> str:
    """Return a language code that whisperx.load_align_model supports."""
    if detected in _WHISPERX_ALIGN_LANGUAGES:
        return detected
    log.warning(
        "diarize.unsupported_align_language",
        detected=detected,
        fallback="en",
    )
    return "en"


# ═════════════════════════════════════════════════════════════════════════
#  Utterance merging
# ═════════════════════════════════════════════════════════════════════════

def _merge_words_to_utterances(
    segments_with_speakers: list[dict],
    whisper_segments: list[dict],
) -> list[dict]:
    """
    Merge word-level speaker assignments back into clean utterances
    where the same speaker speaks consecutively.

    Each utterance dict:
        {
            "speaker": "SPEAKER_00",
            "start": float,
            "end": float,
            "text": str,
            "language": str,
        }
    """
    # Build a flat list of (word_text, start, end, speaker, language)
    word_entries: list[dict] = []

    # Build a segment-index → language mapping from the original whisper
    # segments (since whisperx alignment strips language info).
    seg_languages: list[str] = [s.get("language", "en") for s in whisper_segments]

    for seg_idx, seg in enumerate(segments_with_speakers):
        seg_lang = seg_languages[seg_idx] if seg_idx < len(seg_languages) else "en"
        seg_speaker = seg.get("speaker", "UNKNOWN")

        words = seg.get("words", [])
        if words:
            for w in words:
                word_entries.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", seg.get("start", 0.0)),
                    "end": w.get("end", seg.get("end", 0.0)),
                    "speaker": w.get("speaker", seg_speaker),
                    "language": seg_lang,
                })
        else:
            # No word-level data — treat the entire segment as one block
            word_entries.append({
                "word": seg.get("text", ""),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "speaker": seg_speaker,
                "language": seg_lang,
            })

    if not word_entries:
        return []

    # ── Merge consecutive words with the same speaker ────────────────
    utterances: list[dict] = []
    current_speaker = word_entries[0]["speaker"]
    current_lang = word_entries[0]["language"]
    current_start = word_entries[0]["start"]
    current_end = word_entries[0]["end"]
    current_words: list[str] = [word_entries[0]["word"]]

    for entry in word_entries[1:]:
        if entry["speaker"] == current_speaker:
            # Same speaker — extend the current utterance
            current_words.append(entry["word"])
            current_end = entry["end"]
            # If language changes mid-utterance, keep the first one
            # (code-switching within a single speaker turn is common)
        else:
            # Different speaker — flush the current utterance
            text = "".join(current_words).strip()
            if text:
                utterances.append({
                    "speaker": current_speaker,
                    "start": round(current_start, 3),
                    "end": round(current_end, 3),
                    "text": text,
                    "language": current_lang,
                })

            # Start a new utterance
            current_speaker = entry["speaker"]
            current_lang = entry["language"]
            current_start = entry["start"]
            current_end = entry["end"]
            current_words = [entry["word"]]

    # Flush the last utterance
    text = "".join(current_words).strip()
    if text:
        utterances.append({
            "speaker": current_speaker,
            "start": round(current_start, 3),
            "end": round(current_end, 3),
            "text": text,
            "language": current_lang,
        })

    return utterances


# ═════════════════════════════════════════════════════════════════════════
#  Fallback — no diarization
# ═════════════════════════════════════════════════════════════════════════

def _fallback_no_diarization(whisper_segments: list[dict]) -> list[dict]:
    """
    Return segments with ``speaker = "UNKNOWN"`` and translated text
    when diarization is not available.
    """
    log.warning("diarize.fallback", reason="returning segments without speaker labels")
    utterances: list[dict] = []

    for seg in whisper_segments:
        lang = seg.get("language", "en")
        text = seg.get("text", "").strip()
        text_en = translate_segment(text, lang) if lang != "en" else text

        utterances.append({
            "speaker": "UNKNOWN",
            "start": round(seg.get("start", 0.0), 3),
            "end": round(seg.get("end", 0.0), 3),
            "text": text,
            "text_en": text_en,
            "language": lang,
        })

    return utterances


# ═════════════════════════════════════════════════════════════════════════
#  Main diarization function
# ═════════════════════════════════════════════════════════════════════════

def diarize(
    audio_path: str,
    whisper_segments: list[dict],
    hf_token: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list[dict]:
    """
    Combine Whisper transcription with speaker diarization.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (WAV recommended, 16 kHz mono).
    whisper_segments : list[dict]
        Output of ``transcribe.transcribe()`` — each dict has
        ``start``, ``end``, ``text``, ``language``, ``words``.
    hf_token : str
        Hugging Face token for downloading pyannote diarization models.
        Obtain from https://huggingface.co/settings/tokens and accept
        the pyannote model licence agreements.
    min_speakers / max_speakers : int, optional
        Hints for the diarization pipeline if you know the expected
        number of participants.

    Returns
    -------
    list[dict]
        Merged utterances, each containing::

            {
                "speaker": "SPEAKER_00",
                "start": 12.45,
                "end": 17.82,
                "text": "Aaj ka agenda discuss karte hain",
                "text_en": "Let's discuss today's agenda",
                "language": "hi"
            }

    If diarization fails, returns segments with ``speaker = "UNKNOWN"``.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if not whisper_segments:
        log.warning("diarize.empty_segments")
        return []

    if not hf_token:
        log.error("diarize.no_hf_token")
        return _fallback_no_diarization(whisper_segments)

    log.info(
        "diarize.start",
        audio=str(path),
        segments=len(whisper_segments),
        device=DEVICE,
    )
    t0 = time.monotonic()

    try:
        # ── 1. Load audio ────────────────────────────────────────────
        log.info("diarize.step", step="load_audio")
        audio = whisperx.load_audio(str(path))

        # ── 2. Prepare segments for whisperx alignment ───────────────
        #    whisperx.align() expects [{"start", "end", "text"}, ...]
        align_input = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in whisper_segments
        ]

        # ── 3. Load alignment model ─────────────────────────────────
        primary_lang = _detect_primary_language(whisper_segments)
        align_lang = _resolve_align_language(primary_lang)

        log.info("diarize.step", step="load_align_model", language=align_lang)
        align_model, align_metadata = whisperx.load_align_model(
            language_code=align_lang,
            device=DEVICE,
        )

        # ── 4. Align — word-level timestamps ────────────────────────
        log.info("diarize.step", step="align")
        aligned = whisperx.align(
            align_input,
            align_model,
            align_metadata,
            audio,
            device=DEVICE,
            return_char_alignments=False,
        )

        aligned_segments: list[dict] = aligned.get("segments", [])
        log.info("diarize.aligned", segments=len(aligned_segments))

        # Free alignment model memory (can be large)
        del align_model
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()

        # ── 5. Run pyannote speaker diarization ─────────────────────
        log.info("diarize.step", step="diarization_pipeline")
        diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=DEVICE,
        )

        diarize_kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

        diarize_result = diarize_pipeline(audio, **diarize_kwargs)

        # ── 6. Assign speaker labels to each word ───────────────────
        log.info("diarize.step", step="assign_word_speakers")
        assigned = whisperx.assign_word_speakers(diarize_result, aligned)
        assigned_segments: list[dict] = assigned.get("segments", [])

        # Count unique speakers
        speakers_found: set[str] = set()
        for seg in assigned_segments:
            if seg.get("speaker"):
                speakers_found.add(seg["speaker"])
            for w in seg.get("words", []):
                if w.get("speaker"):
                    speakers_found.add(w["speaker"])

        log.info(
            "diarize.speakers_identified",
            count=len(speakers_found),
            speakers=sorted(speakers_found),
        )

        # ── 7. Merge words into clean utterances ────────────────────
        log.info("diarize.step", step="merge_utterances")
        utterances = _merge_words_to_utterances(assigned_segments, whisper_segments)

        # ── 8. Add English translations for Hindi utterances ────────
        log.info("diarize.step", step="translate")
        for utt in utterances:
            lang = utt.get("language", "en")
            if lang == "en":
                utt["text_en"] = utt["text"]
            else:
                utt["text_en"] = translate_segment(utt["text"], lang)

        elapsed = time.monotonic() - t0
        log.info(
            "diarize.done",
            utterances=len(utterances),
            speakers=len(speakers_found),
            elapsed_sec=f"{elapsed:.1f}",
        )

        return utterances

    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.exception(
            "diarize.failed",
            error=str(exc),
            elapsed_sec=f"{elapsed:.1f}",
        )
        # Graceful degradation — return without speaker labels
        return _fallback_no_diarization(whisper_segments)


# ═════════════════════════════════════════════════════════════════════════
#  Transcript formatter
# ═════════════════════════════════════════════════════════════════════════

def _format_timestamp(seconds: float) -> str:
    """Convert seconds (float) to HH:MM:SS format."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript(utterances: list[dict]) -> str:
    """
    Format diarized utterances into a human-readable transcript.

    Output format::

        [00:01:23] SPEAKER_00: Aaj ka agenda discuss karte hain...
        [00:01:23] SPEAKER_00 (EN): Let's discuss today's agenda...

        [00:01:42] SPEAKER_01: Sure, let me share my screen.

    Rules:
    - Hindi utterances get two lines: original + English translation.
    - English utterances get a single line (no duplicate).
    - Blank line between different speakers for readability.

    Parameters
    ----------
    utterances : list[dict]
        Output of ``diarize()``.

    Returns
    -------
    str
        Formatted multi-line transcript.
    """
    if not utterances:
        return "(empty transcript)"

    lines: list[str] = []
    prev_speaker: Optional[str] = None

    for utt in utterances:
        speaker = utt.get("speaker", "UNKNOWN")
        start = utt.get("start", 0.0)
        text = utt.get("text", "").strip()
        text_en = utt.get("text_en", "").strip()
        lang = utt.get("language", "en")
        timestamp = _format_timestamp(start)

        # Blank line between different speakers
        if prev_speaker is not None and speaker != prev_speaker:
            lines.append("")

        # Original line
        lines.append(f"[{timestamp}] {speaker}: {text}")

        # English translation line (only for non-English utterances)
        if lang != "en" and text_en and text_en != text:
            lines.append(f"[{timestamp}] {speaker} (EN): {text_en}")

        prev_speaker = speaker

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="diarize",
        description="Magentic-meetbot — Speaker diarization with whisperx",
    )
    parser.add_argument(
        "audio_path",
        help="Path to the audio file (WAV, 16 kHz mono recommended)",
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to the JSON transcript file (output of transcribe.py)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers (hint for diarization)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers (hint for diarization)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format: 'json' (structured) or 'text' (human-readable)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="Pretty-print JSON output",
    )
    args = parser.parse_args()

    # ── Load transcript ──────────────────────────────────────────────
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"ERROR: Transcript file not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    whisper_segments = json.loads(transcript_path.read_text(encoding="utf-8"))
    log.info("cli.loaded_transcript", segments=len(whisper_segments))

    # ── Validate HF token ────────────────────────────────────────────
    if not args.hf_token:
        print(
            "WARNING: No HF_TOKEN provided. Diarization will fall back to "
            "UNKNOWN speakers. Set HF_TOKEN env var or pass --hf-token.",
            file=sys.stderr,
        )

    # ── Run diarization ──────────────────────────────────────────────
    utterances = diarize(
        audio_path=args.audio_path,
        whisper_segments=whisper_segments,
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    # ── Output ───────────────────────────────────────────────────────
    if args.format == "text":
        print(format_transcript(utterances))
    else:
        indent = 2 if args.pretty else None
        print(json.dumps(utterances, ensure_ascii=False, indent=indent))

    # ── Summary to stderr ────────────────────────────────────────────
    speakers = sorted({u.get("speaker", "?") for u in utterances})
    total_dur = (utterances[-1]["end"] - utterances[0]["start"]) if utterances else 0
    print(
        f"\n--- Summary ---\n"
        f"Utterances: {len(utterances)}\n"
        f"Speakers  : {speakers}\n"
        f"Duration  : {total_dur:.1f}s\n"
        f"Device    : {DEVICE}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
