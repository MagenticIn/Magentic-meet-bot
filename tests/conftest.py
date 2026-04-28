from __future__ import annotations

import json

import pytest


@pytest.fixture
def sample_utterances() -> list[dict]:
    return [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 4.5,
            "text": "Namaste team, sprint status discuss karte hain.",
            "text_en": "Hello team, let's discuss sprint status.",
            "language": "hi",
        },
        {
            "speaker": "SPEAKER_01",
            "start": 5.0,
            "end": 9.0,
            "text": "Deployment can happen by Friday.",
            "text_en": "Deployment can happen by Friday.",
            "language": "en",
        },
    ]


@pytest.fixture
def sample_meeting_meta() -> dict:
    return {
        "title": "Sprint Sync",
        "date": "2026-04-27",
        "duration_minutes": 30,
        "attendees": ["Rahul", "Priya"],
    }


@pytest.fixture
def mock_openai_response() -> dict:
    return {
        "id": "chatcmpl-test123",
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "summary": "Test meeting summary",
                            "tldr": "Test tldr",
                            "key_points": ["Point 1", "Point 2"],
                            "action_items": [{"task": "Fix bug", "owner": "Rahul", "deadline": "Friday"}],
                            "decisions": ["Decided to use React"],
                            "risks": [],
                            "next_steps": ["Deploy by EOD"],
                            "topics_discussed": ["bug fix", "deployment"],
                            "mom": "The meeting was held and attended by the team.",
                            "next_meeting": None,
                            "sentiment": "positive",
                            "follow_up_required": False,
                        }
                    )
                }
            }
        ],
    }
