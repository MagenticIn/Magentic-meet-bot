from __future__ import annotations

import importlib
import json
import re
import sys
import types
import uuid

from api.models import HealthResponse, MeetingOut, MeetingStatus, TriggerMeetingRequest


def test_transcribe_mock_returns_shape(tmp_path, monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

    class FakeWord:
        word = "hello"
        start = 0.0
        end = 0.5
        probability = 0.95

    class FakeSegment:
        start = 0.0
        end = 1.0
        text = "hello"
        language = "en"
        words = [FakeWord()]

    class FakeInfo:
        language = "en"
        language_probability = 0.99
        duration = 1.0

    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, *_args, **_kwargs):
            return iter([FakeSegment()]), FakeInfo()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    transcribe_mod = importlib.reload(importlib.import_module("pipeline.transcribe"))

    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    data = transcribe_mod.transcribe(str(wav))
    assert isinstance(data, list)
    assert {"start", "end", "text", "language", "words"}.issubset(data[0].keys())


def test_diarize_format_transcript(sample_utterances, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "whisperx", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=object))
    diarize_mod = importlib.reload(importlib.import_module("pipeline.diarize"))
    formatted = diarize_mod.format_transcript(sample_utterances)
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] SPEAKER_00:", formatted)


def test_summarize_mock_openai(sample_utterances, sample_meeting_meta, mock_openai_response, monkeypatch):
    summarize_mod = importlib.reload(importlib.import_module("pipeline.summarize"))
    monkeypatch.setattr(summarize_mod, "OPENAI_API_KEY", "test-key")

    class _Msg:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _Msg(content)

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 20

    class _Resp:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]
            self.usage = _Usage()

    class _Completions:
        @staticmethod
        def create(**_kwargs):
            return _Resp(mock_openai_response["choices"][0]["message"]["content"])

    class _Chat:
        completions = _Completions()

    class _OpenAI:
        def __init__(self, *_args, **_kwargs):
            self.chat = _Chat()

    monkeypatch.setattr(summarize_mod, "OpenAI", _OpenAI)
    result = summarize_mod.summarize(sample_utterances, sample_meeting_meta)
    assert {"summary", "key_points", "action_items", "decisions", "sentiment"}.issubset(result.keys())


def test_models_validate_request_and_output():
    req = TriggerMeetingRequest(meeting_url="https://meet.google.com/abc-defg-hij", title="Standup")
    assert req.meeting_url.startswith("https://meet.google.com/")

    meeting = MeetingOut(
        id=uuid.uuid4(),
        meeting_url="https://meet.google.com/abc-defg-hij",
        title="Standup",
        date=None,
        status=MeetingStatus.JOINING,
        duration=20,
        summary=None,
        key_points=[],
        action_items=[],
        decisions=[],
        next_meeting=None,
        sentiment="neutral",
        transcript=[],
        raw_transcript="",
        translated_transcript="",
    )
    assert meeting.status == MeetingStatus.JOINING


def test_health_model_defaults():
    health = HealthResponse()
    assert health.status == "ok"
    assert health.version == "1.0.0"


def test_openai_fixture_content_shape(mock_openai_response):
    payload = json.loads(mock_openai_response["choices"][0]["message"]["content"])
    assert "summary" in payload
    assert "action_items" in payload
