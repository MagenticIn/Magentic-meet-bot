"""
Magentic-meetbot  ·  summarize.py
─────────────────────────────────
Generate structured meeting notes from a diarized transcript using
OpenAI gpt-4o-mini (primary) or a local Ollama llama3.1:8b (fallback).

Input
─────
- ``utterances`` — output of ``diarize.diarize()``
- ``meeting_meta`` — dict with title, date, duration, attendees

Output
──────
A ``MeetingNotes`` dataclass serialised as dict:
    summary, key_points, action_items, decisions,
    next_meeting, sentiment, raw_transcript, translated_transcript

CLI
───
    python -m pipeline.summarize /data/meeting_id/transcript_diarized.json
    python -m pipeline.summarize /data/meeting_id/transcript_diarized.json --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger("summarize")

# ─── Config ──────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", "2048"))
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


# ═════════════════════════════════════════════════════════════════════════
#  System prompt
# ═════════════════════════════════════════════════════════════════════════

LLM_SYSTEM_PROMPT: str = """\
You are a meeting notes assistant for an Indian tech team that speaks Hindi and English (often mixed). You receive a full meeting transcript with speaker labels. Some segments may be in Hindi, some in English, some mixed (Hinglish).

Your job is to produce structured meeting notes in JSON format only. No markdown, no preamble, no explanation — output raw JSON only.

Rules:
- Write all notes in clear English
- If a speaker said something in Hindi, understand the meaning and include it in English
- Extract only concrete decisions and assigned tasks — ignore small talk and tangents
- For action items, always try to identify the owner (who was assigned) and deadline (if mentioned)
- Be concise in the summary (3-5 sentences max)
- If no deadline was mentioned for an action item, set deadline to null

Output this exact JSON schema:
{
  "summary": "string — 3 to 5 sentence overview of what was discussed and decided",
  "key_points": ["string", "string"],
  "action_items": [
    {
      "task": "string — what needs to be done",
      "owner": "string — person's name or UNKNOWN",
      "deadline": "string like 'Friday' or '2025-05-01' or null"
    }
  ],
  "decisions": ["string — each major decision made"],
  "next_meeting": "string describing next meeting if mentioned, or null",
  "sentiment": "one of: positive, neutral, needs_followup",
  "topics_discussed": ["string"]
}

The transcript will follow in the user message formatted as:
[HH:MM:SS] SPEAKER_NAME: original text
[HH:MM:SS] SPEAKER_NAME (EN): english translation (if original was Hindi)

Return only the JSON object. No other text.
"""


# ═════════════════════════════════════════════════════════════════════════
#  MeetingNotes dataclass
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class ActionItem:
    """A single action item extracted from the meeting."""
    task: str = ""
    owner: str = ""
    deadline: Optional[str] = None


@dataclass
class MeetingNotes:
    """Structured meeting notes produced by the LLM."""

    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    topics_discussed: list[str] = field(default_factory=list)
    next_meeting: Optional[str] = None
    sentiment: str = "neutral"          # positive | neutral | needs_followup
    raw_transcript: str = ""
    translated_transcript: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (action_items as list of dicts)."""
        d = asdict(self)
        return d


# ═════════════════════════════════════════════════════════════════════════
#  Transcript formatting
# ═════════════════════════════════════════════════════════════════════════

def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _build_raw_transcript(utterances: list[dict]) -> str:
    """
    Build the original-language transcript string.

    Format:
        [00:12] SPEAKER_00: नमस्ते, आज का agenda discuss karte hain
        [00:25] SPEAKER_01: Sure, let's start with the Q3 numbers
    """
    if not utterances:
        return "(empty transcript)"

    lines: list[str] = []
    prev_speaker: Optional[str] = None

    for utt in utterances:
        speaker = utt.get("speaker", "UNKNOWN")
        start = utt.get("start", 0.0)
        text = utt.get("text", "").strip()
        ts = _format_timestamp(start)

        if prev_speaker is not None and speaker != prev_speaker:
            lines.append("")  # blank line between speakers
        lines.append(f"[{ts}] {speaker}: {text}")
        prev_speaker = speaker

    return "\n".join(lines)


def _build_translated_transcript(utterances: list[dict]) -> str:
    """
    Build a transcript with English translations alongside Hindi originals.

    Format:
        [00:12] SPEAKER_00: HI: नमस्ते, आज का agenda discuss karte hain | EN: Hello, let's discuss today's agenda
        [00:25] SPEAKER_01: HI: Sure, let's start with the Q3 numbers | EN: Sure, let's start with the Q3 numbers
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
        ts = _format_timestamp(start)

        if prev_speaker is not None and speaker != prev_speaker:
            lines.append("")
        english = text_en or text
        src_lang = (utt.get("language", "en") or "en").lower()
        # Show the actual detected source language to avoid misleading "HI" labels.
        lines.append(f"[{ts}] {speaker}: SRC({src_lang}): {text} | EN: {english}")

        prev_speaker = speaker

    return "\n".join(lines)


def _build_llm_prompt(
    utterances: list[dict],
    meeting_meta: dict[str, Any],
) -> str:
    """Build the user-side prompt for the LLM."""
    translated_transcript = _build_translated_transcript(utterances)

    # Meeting metadata header
    title = meeting_meta.get("title", "Untitled Meeting")
    date = meeting_meta.get("date", "Unknown date")
    duration = meeting_meta.get("duration_minutes", 0)
    attendees = meeting_meta.get("attendees", [])
    attendee_str = ", ".join(attendees) if attendees else "Not specified"

    # Count unique speakers from transcript
    speakers = sorted({u.get("speaker", "UNKNOWN") for u in utterances})

    prompt = (
        f"Meeting: {title}\n"
        f"Date: {date}\n"
        f"Duration: ~{duration} minutes\n"
        f"Attendees: {attendee_str}\n"
        f"Speakers detected: {len(speakers)} ({', '.join(speakers)})\n"
        f"\n"
        f"--- TRANSCRIPT ---\n"
        f"{translated_transcript}\n"
        f"--- END TRANSCRIPT ---\n"
        f"\n"
        f"Produce the structured meeting notes JSON now."
    )
    return prompt


# ═════════════════════════════════════════════════════════════════════════
#  JSON extraction
# ═════════════════════════════════════════════════════════════════════════

def _extract_json(raw: str) -> dict[str, Any]:
    """
    Parse JSON from the LLM response, handling common quirks:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace or commentary
    """
    text = raw.strip()

    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Attempt direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost JSON object in the text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in LLM response", text, 0)


def _parse_meeting_notes(
    parsed: dict[str, Any],
    raw_transcript: str,
    translated_transcript: str,
) -> MeetingNotes:
    """Convert the parsed LLM JSON into a MeetingNotes dataclass."""
    def _normalize_nullable(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.lower() in {"null", "none", "n/a", ""}:
                return None
            return cleaned
        return str(value)

    action_items: list[ActionItem] = []
    for item in parsed.get("action_items", []):
        if isinstance(item, dict):
            action_items.append(ActionItem(
                task=item.get("task", ""),
                owner=item.get("owner", ""),
                deadline=_normalize_nullable(item.get("deadline")),
            ))

    notes = MeetingNotes(
        summary=parsed.get("summary", ""),
        key_points=parsed.get("key_points", []),
        action_items=action_items,
        decisions=parsed.get("decisions", []),
        topics_discussed=parsed.get("topics_discussed", []),
        next_meeting=_normalize_nullable(parsed.get("next_meeting")),
        sentiment=parsed.get("sentiment", "neutral"),
        raw_transcript=raw_transcript,
        translated_transcript=translated_transcript,
    )
    return _backfill_missing_sections(notes, raw_transcript)


def _backfill_missing_sections(notes: MeetingNotes, raw_transcript: str) -> MeetingNotes:
    """
    Fill missing note sections when model output is sparse.
    This keeps dashboard sections useful even when LLM omits optional arrays.
    """
    def _clean_list(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        out: list[str] = []
        for v in values:
            s = str(v).strip()
            if s and s.lower() not in {"none", "null", "n/a"}:
                out.append(s)
        return out

    notes.key_points = _clean_list(notes.key_points)
    notes.decisions = _clean_list(notes.decisions)
    notes.topics_discussed = _clean_list(notes.topics_discussed)

    if not notes.summary.strip():
        lines = [ln.strip() for ln in raw_transcript.splitlines() if ln.strip()]
        sample = " ".join(line.split(": ", 1)[-1] for line in lines[:3]).strip()
        notes.summary = sample[:500] if sample else "Meeting discussion captured."

    if not notes.key_points:
        lines = [ln.strip() for ln in raw_transcript.splitlines() if ln.strip()]
        extracted: list[str] = []
        for line in lines:
            text = line.split(": ", 1)[-1].strip()
            if len(text) >= 20:
                extracted.append(text[:200])
            if len(extracted) == 3:
                break
        notes.key_points = extracted

    if not notes.topics_discussed:
        notes.topics_discussed = notes.key_points[:]

    if notes.sentiment not in {"positive", "neutral", "needs_followup"}:
        notes.sentiment = "neutral"

    return notes


# ═════════════════════════════════════════════════════════════════════════
#  OpenAI backend
# ═════════════════════════════════════════════════════════════════════════

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_openai(formatted_transcript: str) -> dict[str, Any]:
    """
    Call the OpenAI Chat Completions API and return parsed JSON.
    Retries up to 3 times with exponential backoff.
    """
    log.info(
        "summarize.openai.calling",
        model="gpt-4o-mini",
        max_tokens=MAX_TOKENS,
        prompt_chars=len(formatted_transcript),
    )

    t0 = time.monotonic()
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_transcript},
        ],
    )

    raw_content = response.choices[0].message.content or "{}"
    result = _extract_json(raw_content)
    elapsed = time.monotonic() - t0

    log.info(
        "summarize.openai.response",
        elapsed_sec=f"{elapsed:.1f}",
        input_tokens=(response.usage.prompt_tokens if response.usage else 0),
        output_tokens=(response.usage.completion_tokens if response.usage else 0),
        model="gpt-4o-mini",
    )

    return result


# ═════════════════════════════════════════════════════════════════════════
#  Ollama (local) fallback backend
# ═════════════════════════════════════════════════════════════════════════

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _call_ollama(user_prompt: str) -> str:
    """
    Call a local Ollama instance as a fallback when OPENAI_API_KEY
    is not available.

    Uses llama3.1:8b by default.
    POST to http://localhost:11434/api/generate
    """
    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{LLM_SYSTEM_PROMPT}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": MAX_TOKENS,
        },
    }

    log.info(
        "summarize.ollama.calling",
        model=OLLAMA_MODEL,
        url=url,
        prompt_chars=len(full_prompt),
    )

    t0 = time.monotonic()
    with httpx.Client(timeout=120) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()

    result = resp.json()
    raw_text = result.get("response", "").strip()
    elapsed = time.monotonic() - t0

    log.info(
        "summarize.ollama.response",
        elapsed_sec=f"{elapsed:.1f}",
        model=OLLAMA_MODEL,
        response_chars=len(raw_text),
        eval_count=result.get("eval_count", 0),
    )

    return raw_text


# ═════════════════════════════════════════════════════════════════════════
#  Main summarize function
# ═════════════════════════════════════════════════════════════════════════

def summarize(utterances: list[dict], meeting_meta: dict) -> dict[str, Any]:
    """
    Generate structured meeting notes from diarized utterances.

    Parameters
    ----------
    utterances : list[dict]
        Output of ``diarize.diarize()`` — each dict contains
        ``speaker``, ``start``, ``end``, ``text``, ``text_en``, ``language``.
    meeting_meta : dict, optional
        Meeting metadata::

            {
                "title": "Sprint Planning",
                "date": "2025-04-27",
                "duration_minutes": 45,
                "attendees": ["Alice", "Bob"]
            }

        If not provided, defaults are inferred from the utterances.

    Returns
    -------
    dict
        Serialised ``MeetingNotes`` dataclass with keys:
        ``summary``, ``key_points``, ``action_items``, ``decisions``,
        ``next_meeting``, ``sentiment``, ``raw_transcript``,
        ``translated_transcript``.
    """
    if not utterances:
        log.warning("summarize.empty_utterances")
        return MeetingNotes(
            summary="No transcript available.",
            sentiment="neutral",
        ).to_dict()

    # ── Infer missing metadata from utterances ───────────────────────
    if meeting_meta is None:
        meeting_meta = {}

    if "duration_minutes" not in meeting_meta and utterances:
        first_start = utterances[0].get("start", 0.0)
        last_end = utterances[-1].get("end", 0.0)
        meeting_meta["duration_minutes"] = round((last_end - first_start) / 60, 1)

    if "attendees" not in meeting_meta:
        speakers = sorted({u.get("speaker", "UNKNOWN") for u in utterances})
        meeting_meta["attendees"] = speakers

    meeting_meta.setdefault("title", "Untitled Meeting")
    meeting_meta.setdefault("date", "Unknown date")

    # ── Build transcripts ────────────────────────────────────────────
    raw_transcript = _build_raw_transcript(utterances)
    translated_transcript = _build_translated_transcript(utterances)
    user_prompt = _build_llm_prompt(utterances, meeting_meta)

    # ── Call LLM ─────────────────────────────────────────────────────
    backend = "none"
    try:
        if OPENAI_API_KEY:
            backend = "openai"
            parsed = _call_openai(user_prompt)
        else:
            backend = "ollama"
            log.warning(
                "summarize.no_openai_key",
                fallback="ollama",
                model=OLLAMA_MODEL,
            )
            raw_response = _call_ollama(user_prompt)

    except Exception as exc:
        log.exception("summarize.llm_call_failed", backend=backend, error=str(exc))
        # Return a minimal notes object with just the transcripts
        return MeetingNotes(
            summary=f"LLM summarisation failed ({backend}): {exc}",
            sentiment="needs_followup",
            raw_transcript=raw_transcript,
            translated_transcript=translated_transcript,
        ).to_dict()

    if backend == "ollama":
        try:
            parsed = _extract_json(raw_response)
        except json.JSONDecodeError as exc:
            log.error(
                "summarize.json_parse_error",
                raw=raw_response[:500],
                error=str(exc),
                backend=backend,
            )
            return MeetingNotes(
                summary=raw_response[:2000],
                sentiment="needs_followup",
                raw_transcript=raw_transcript,
                translated_transcript=translated_transcript,
            ).to_dict()

    # ── Build MeetingNotes ───────────────────────────────────────────
    notes = _parse_meeting_notes(parsed, raw_transcript, translated_transcript)

    log.info(
        "summarize.done",
        backend=backend,
        action_items=len(notes.action_items),
        decisions=len(notes.decisions),
        sentiment=notes.sentiment,
    )

    return notes.to_dict()


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="summarize",
        description="Magentic-meetbot — Summarize diarized transcript with LLM",
    )
    parser.add_argument(
        "transcript_path",
        help="Path to the diarized transcript JSON (output of diarize.py)",
    )
    parser.add_argument(
        "--title",
        default="Untitled Meeting",
        help="Meeting title",
    )
    parser.add_argument(
        "--date",
        default="Unknown date",
        help="Meeting date",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="Pretty-print the JSON output",
    )
    args = parser.parse_args()

    # ── Load utterances ──────────────────────────────────────────────
    path = Path(args.transcript_path)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    utterances = json.loads(path.read_text(encoding="utf-8"))
    log.info("cli.loaded", utterances=len(utterances))

    meta = {"title": args.title, "date": args.date}

    # ── Summarize ────────────────────────────────────────────────────
    result = summarize(utterances, meeting_meta=meta)

    # ── Output ───────────────────────────────────────────────────────
    indent = 2 if args.pretty else None
    print(json.dumps(result, ensure_ascii=False, indent=indent))

    # ── Summary to stderr ────────────────────────────────────────────
    backend = "openai" if OPENAI_API_KEY else "ollama"
    print(
        f"\n--- Summary Stats ---\n"
        f"Backend      : {backend} ({'gpt-4o-mini' if OPENAI_API_KEY else OLLAMA_MODEL})\n"
        f"Sentiment    : {result.get('sentiment', '?')}\n"
        f"Key Points   : {len(result.get('key_points', []))}\n"
        f"Action Items : {len(result.get('action_items', []))}\n"
        f"Decisions    : {len(result.get('decisions', []))}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
