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


def test_openai_transcribe_diarize_maps_utterances(tmp_path, monkeypatch):
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeSeg:
        def model_dump(self):
            return {"speaker": "A", "start": 0.0, "end": 1.5, "text": "Hello"}

    class FakeResp:
        def model_dump(self):
            return {"text": "Hello", "segments": [FakeSeg()]}

    class FakeTranscriptions:
        @staticmethod
        def create(**kwargs):
            assert kwargs.get("model") == "gpt-4o-transcribe-diarize"
            assert kwargs.get("response_format") == "diarized_json"
            assert kwargs.get("chunking_strategy") == "auto"
            return FakeResp()

    class FakeAudio:
        transcriptions = FakeTranscriptions()

    class FakeClient:
        def __init__(self, **_kw):
            self.audio = FakeAudio()

    import pipeline.openai_transcribe_diarize as otd

    monkeypatch.setattr(otd, "OpenAI", FakeClient)
    monkeypatch.setattr(otd, "translate_segment", lambda text, lang: text)
    utt, raw = otd.transcribe_diarize_openai(str(wav))
    assert len(utt) == 1
    assert utt[0]["speaker"] == "SPEAKER_00"
    assert utt[0]["text"] == "Hello"
    assert utt[0]["language"] == "en"
    assert utt[0]["text_en"] == "Hello"
    assert raw.get("segments")


def test_openai_segment_language_devanagari_vs_english():
    import pipeline.openai_transcribe_diarize as otd

    assert otd._segment_language("नमस्ते, कैसे हैं आप?") == "hi"
    assert otd._segment_language("Hello, let's start the standup.") == "en"
    assert otd._segment_language("Chalo agenda discuss karte hain") == "en"


def test_openai_request_language_respects_env(monkeypatch):
    import pipeline.openai_transcribe_diarize as otd

    monkeypatch.delenv("OPENAI_TRANSCRIPTION_LANGUAGE", raising=False)
    assert otd._openai_request_language() is None
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_LANGUAGE", "hi")
    assert otd._openai_request_language() == "hi"
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_LANGUAGE", "EN")
    assert otd._openai_request_language() == "en"


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
