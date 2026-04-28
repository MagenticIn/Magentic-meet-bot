"""
Magentic-meetbot  ·  pm_client.py
─────────────────────────────────
HTTP client that POSTs meeting notes (transcript + summary) to an external
Project-Management (PM) API.

The target API is expected to accept a JSON payload at:

    POST  {PM_API_BASE_URL}/meetings/notes

Headers:
    Authorization: Bearer {PM_API_KEY}
    Content-Type: application/json

Payload schema:

    {
        "meeting_id": "uuid",
        "meeting_url": "https://meet.google.com/...",
        "summary": { ... },       # structured summary from LLM
        "transcript": [ ... ],    # diarized transcript segments
        "source": "magentic-meetbot"
    }
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

PM_API_BASE_URL = os.getenv("PM_API_BASE_URL", "")
PM_API_KEY = os.getenv("PM_API_KEY", "")


class PMClientError(Exception):
    """Raised when the PM API returns a non-2xx response."""


class PMClient:
    """Synchronous HTTP client for the external PM API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or PM_API_BASE_URL).rstrip("/")
        self.api_key = api_key or PM_API_KEY
        self.timeout = timeout

        if not self.base_url:
            log.warning("pm_client.no_base_url — PM sync will be skipped")
        if not self.api_key:
            log.warning("pm_client.no_api_key — PM sync may fail")

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "Magentic-meetbot/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        reraise=True,
    )
    def post_meeting_notes(
        self,
        meeting_id: str,
        meeting_url: str,
        summary: dict[str, Any],
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        POST the meeting notes to the PM API.

        Returns the parsed JSON response body on success.
        Raises PMClientError on non-2xx after retries.
        """
        if not self.base_url:
            log.info("pm_client.skipped — no PM_API_BASE_URL configured")
            return {"status": "skipped", "reason": "no_base_url"}

        payload = {
            "meeting_id": meeting_id,
            "meeting_url": meeting_url,
            "summary": summary,
            "transcript": transcript,
            "source": "magentic-meetbot",
        }

        url = f"{self.base_url}/meetings/notes"
        log.info("pm_client.posting", url=url, meeting_id=meeting_id)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._headers())

            if resp.status_code >= 400:
                log.error(
                    "pm_client.error",
                    status=resp.status_code,
                    body=resp.text[:500],
                )
                resp.raise_for_status()

        result = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
        log.info("pm_client.success", status=resp.status_code, meeting_id=meeting_id)
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        reraise=True,
    )
    def update_meeting_status(
        self,
        meeting_id: str,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        """
        PATCH the meeting status on the PM API (optional endpoint).
        Useful for notifying the PM tool about failures or cancellations.
        """
        if not self.base_url:
            return {"status": "skipped"}

        url = f"{self.base_url}/meetings/{meeting_id}/status"
        payload: dict[str, Any] = {"status": status}
        if error:
            payload["error"] = error

        log.info("pm_client.update_status", url=url, status=status)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.patch(url, json=payload, headers=self._headers())
            resp.raise_for_status()

        return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
