"""
Magentic-meetbot  ·  meet_bot.py
────────────────────────────────
Production-ready async Playwright bot for Google Meet.

Usage
─────
    python -m bot.meet_bot <MEET_URL> <RECORDING_OUTPUT_PATH>

    # or via env:
    MEET_URL=https://meet.google.com/abc-defg-hij \
    RECORDING_OUTPUT_PATH=/data/recording.wav \
    python -m bot.meet_bot

Lifecycle
─────────
1.  Parse CLI args / env for the Meet URL and output path.
2.  Launch headless Chromium with stealth + media flags.
3.  Authenticate with Google (email → password).
4.  Navigate to the Meet URL, handle pre-join screen.
5.  Click "Join now" / "Ask to join"; wait up to 60 s for host approval.
6.  Mute mic + turn off camera AFTER joining.
7.  Signal audio_capture.py via /tmp/meetbot_state.json ("recording_started").
8.  Poll every 10 s for meeting end conditions.
9.  Write "recording_stopped" to the state file and exit.

Exit codes
──────────
    0   — clean exit (meeting ended normally)
    1   — fatal / unrecoverable error
    2   — removed by host
    3   — meeting was full
    4   — network failure during meeting
    5   — host did not approve within timeout
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
import structlog
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeout,
)

log = structlog.get_logger("meet_bot")

# ─── Configuration ───────────────────────────────────────────────────────
GOOGLE_EMAIL: str = os.environ.get("GOOGLE_EMAIL", "")
GOOGLE_PASSWORD: str = os.environ.get("GOOGLE_PASSWORD", "")
BOT_DISPLAY_NAME: str = os.getenv("BOT_DISPLAY_NAME", "Magentic Notetaker")
STATE_FILE: Path = Path("/tmp/meetbot_state.json")
HOST_APPROVAL_TIMEOUT_SEC: int = int(os.getenv("BOT_JOIN_TIMEOUT_SEC", "60"))
POLL_INTERVAL_SEC: int = 10
MAX_MEETING_DURATION_SEC: int = int(os.getenv("BOT_MAX_MEETING_DURATION_SEC", "7200"))
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://api:8000")
BOT_HEADLESS: bool = os.getenv("BOT_HEADLESS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
END_DETECTION_STABLE_POLLS: int = int(os.getenv("BOT_END_DETECTION_STABLE_POLLS", "2"))

CHROMIUM_ARGS: list[str] = [
    "--use-fake-ui-for-media-stream",
    "--disable-blink-features=AutomationControlled",
    "--autoplay-policy=no-user-gesture-required",
    "--disable-features=WebRtcHideLocalIpsWithMdns",
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-accelerated-2d-canvas",
    "--disable-gpu",
]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# ─── Exit codes ──────────────────────────────────────────────────────────
class ExitCode(int, Enum):
    OK = 0
    FATAL = 1
    REMOVED_BY_HOST = 2
    MEETING_FULL = 3
    NETWORK_DROP = 4
    HOST_APPROVAL_TIMEOUT = 5


# ─── Custom exceptions ──────────────────────────────────────────────────
class MeetingFullError(Exception):
    """Raised when Google Meet reports the meeting is full."""


class RemovedByHostError(Exception):
    """Raised when the bot is removed from the meeting by a host."""


class HostApprovalTimeoutError(Exception):
    """Raised when the host does not admit the bot within the approval window."""


class NetworkDropError(Exception):
    """Raised on unrecoverable network-level disconnection."""


class LoginFailedError(Exception):
    """Raised when Google sign-in does not succeed."""


# ─── API Notification ────────────────────────────────────────────────────
async def _notify_api_recording_complete(meeting_id: str, audio_path: str, end_reason: str) -> None:
    """Notify the API that the bot has finished recording."""
    if not meeting_id:
        return

    url = f"{API_BASE_URL}/api/v1/webhook/recording-complete"
    payload = {
        "meeting_id": meeting_id,
        "audio_path": audio_path,
        "end_reason": end_reason,
    }

    log.info("bot.notify_api", url=url, meeting_id=meeting_id)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        log.info("bot.notify_api_success")
    except Exception as exc:
        log.warning("bot.notify_api_failed", error=str(exc))


# ─── Bot state ───────────────────────────────────────────────────────────
@dataclass
class BotState:
    """Tracks the bot's lifecycle; serialised to STATE_FILE for IPC."""

    status: str = "initializing"       # initializing → joining → in_meeting → recording_started → recording_stopped → error
    meeting_url: str = ""
    recording_output_path: str = ""
    join_timestamp: Optional[float] = None
    recording_start_timestamp: Optional[float] = None
    recording_stop_timestamp: Optional[float] = None
    end_reason: str = ""               # meeting_ended | participants_left | removed | timeout | error
    error_message: str = ""
    participant_count: int = 0
    elapsed_seconds: float = 0.0

    def save(self, path: Path = STATE_FILE) -> None:
        """Atomically write state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        log.debug("state.saved", status=self.status, path=str(path))

    def set_error(self, message: str) -> None:
        self.status = "error"
        self.error_message = message
        self.save()


# ─── Google sign-in ──────────────────────────────────────────────────────
async def google_sign_in(page: Page) -> None:
    """
    Full Google authentication flow using email + password.
    Raises LoginFailedError on failure.
    """
    if not GOOGLE_EMAIL or not GOOGLE_PASSWORD:
        raise LoginFailedError("GOOGLE_EMAIL and GOOGLE_PASSWORD must be set in environment")

    log.info("auth.start", email=GOOGLE_EMAIL[:4] + "****")
    await page.goto("https://accounts.google.com/signin", wait_until="networkidle")

    # ── Email step ───────────────────────────────────────────────────
    try:
        email_input = page.locator('input[type="email"]')
        await email_input.wait_for(state="visible", timeout=15_000)
        await email_input.fill(GOOGLE_EMAIL)

        next_btn = page.locator("#identifierNext")
        await next_btn.click()
        log.info("auth.email_submitted")

        # Wait for the email step to transition
        await page.wait_for_timeout(3_000)
    except PlaywrightTimeout:
        raise LoginFailedError("Timed out waiting for email input field")

    # ── Check for identity step errors ────────────────────────────────
    try:
        error_el = page.locator('div[aria-live="assertive"]:has-text("Couldn")')
        if await error_el.count() > 0:
            text = await error_el.first.inner_text()
            lowered = text.lower()
            if "browser" in lowered and ("secure" in lowered or "supported" in lowered):
                raise LoginFailedError(
                    f"Google blocked automated sign-in for this browser/session: {text}. "
                    "Set BOT_HEADLESS=false and retry."
                )
            raise LoginFailedError(f"Google rejected the email: {text}")
    except PlaywrightTimeout:
        pass

    # ── Password step ────────────────────────────────────────────────
    try:
        password_input = page.locator('input[type="password"]')
        await password_input.wait_for(state="visible", timeout=15_000)
        await password_input.fill(GOOGLE_PASSWORD)

        pass_next = page.locator("#passwordNext")
        await pass_next.click()
        log.info("auth.password_submitted")

        await page.wait_for_timeout(5_000)
    except PlaywrightTimeout:
        raise LoginFailedError("Timed out waiting for password input field")

    # ── Check for "Wrong password" error ─────────────────────────────
    try:
        wrong_pw = page.locator('span:has-text("Wrong password")')
        if await wrong_pw.count() > 0:
            raise LoginFailedError("Google rejected the password (wrong password)")
    except PlaywrightTimeout:
        pass

    # ── 2FA / challenge detection ────────────────────────────────────
    challenge_indicators = [
        "2-Step Verification",
        "Verify it's you",
        "Confirm your recovery",
        "security challenge",
    ]
    for indicator in challenge_indicators:
        try:
            el = page.locator(f'text="{indicator}"')
            if await el.count() > 0:
                raise LoginFailedError(
                    f"Google requires additional verification: '{indicator}'. "
                    "Use an App Password or disable 2FA for the bot account."
                )
        except PlaywrightTimeout:
            pass

    # ── Verify sign-in succeeded ─────────────────────────────────────
    current = page.url
    if any(x in current for x in ["myaccount.google.com", "accounts.google.com/b", "mail.google.com"]):
        log.info("auth.success", url=current)
    elif "signin" in current.lower():
        raise LoginFailedError(f"Still on sign-in page after credentials — URL: {current}")
    else:
        # Some redirects are fine; warn but don't fail
        log.warning("auth.uncertain_redirect", url=current)


# ─── Pre-join screen handling ────────────────────────────────────────────
async def _dismiss_overlays(page: Page) -> None:
    """Dismiss common pre-join pop-ups and prompts."""

    # "Got it" buttons (cookie consent, feature announcements)
    for label in ["Got it", "Dismiss", "OK", "Close"]:
        try:
            btn = page.locator(f'button:has-text("{label}")')
            if await btn.count() > 0:
                await btn.first.click()
                log.info("prejoin.dismissed", label=label)
                await page.wait_for_timeout(800)
        except Exception:
            pass

    # "Use without an account" / "Not now" prompts
    for text in ["Use without an account", "Not now", "Continue without signing in"]:
        try:
            el = page.locator(f'text="{text}"')
            if await el.count() > 0:
                log.info("prejoin.found_prompt", text=text)
                # We do NOT click this — we are signed in.  Dismiss if it's blocking.
                close_btn = page.locator(
                    'button[aria-label="Close"], '
                    'button:has-text("Not now"), '
                    'button:has-text("Dismiss")'
                )
                if await close_btn.count() > 0:
                    await close_btn.first.click()
                    await page.wait_for_timeout(800)
        except Exception:
            pass


async def _set_display_name(page: Page) -> None:
    """Fill in the bot's display name if the input is visible."""
    try:
        name_input = page.locator('input[aria-label="Your name"], input[placeholder="Your name"]')
        if await name_input.count() > 0:
            await name_input.first.clear()
            await name_input.first.fill(BOT_DISPLAY_NAME)
            log.info("prejoin.name_set", name=BOT_DISPLAY_NAME)
            await page.wait_for_timeout(500)
    except Exception:
        pass


async def _detect_meeting_full(page: Page) -> None:
    """Raise MeetingFullError if the meeting is full."""
    full_indicators = [
        "This meeting is full",
        "can't join this meeting",
        "maximum number of participants",
    ]
    for text in full_indicators:
        try:
            el = page.locator(f'text="{text}"')
            if await el.count() > 0:
                raise MeetingFullError(f"Meeting is full: '{text}'")
        except MeetingFullError:
            raise
        except Exception:
            pass


async def _click_join_button(page: Page) -> str:
    """
    Click "Join now" or "Ask to join" and return which button was clicked.
    Raises PlaywrightTimeout if neither button appears within 30 s.
    """
    log.info("prejoin.waiting_for_join_button")

    # Wait for either button to appear
    join_selector = (
        'button:has-text("Join now"), '
        'button:has-text("Ask to join"), '
        'button:has-text("Join")'
    )
    join_btn = page.locator(join_selector)

    try:
        await join_btn.first.wait_for(state="visible", timeout=30_000)
    except PlaywrightTimeout:
        # Last resort: look for any button that looks join-related
        fallback = page.locator('[data-mdc-dialog-action="join"], [jsname="Qx7uuf"]')
        if await fallback.count() > 0:
            await fallback.first.click()
            return "fallback_join"
        raise

    btn_text = (await join_btn.first.inner_text()).strip()
    await join_btn.first.click()
    log.info("prejoin.clicked_join", button=btn_text)
    return btn_text


async def _wait_for_host_approval(page: Page) -> None:
    """
    If we clicked "Ask to join", wait up to HOST_APPROVAL_TIMEOUT_SEC
    for the host to admit us (the pre-join screen disappears).
    """
    log.info("prejoin.waiting_for_host_approval", timeout=HOST_APPROVAL_TIMEOUT_SEC)

    deadline = time.monotonic() + HOST_APPROVAL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        # Detect "Asking to be let in..." text
        asking = page.locator('text="Asking to be let in"')
        waiting = page.locator('text="Waiting for someone to let you in"')

        asking_visible = await asking.count() > 0
        waiting_visible = await waiting.count() > 0

        if not asking_visible and not waiting_visible:
            # The waiting screen is gone — we are in (or were rejected)
            log.info("prejoin.host_approved")
            return

        # Check if host denied
        denied_texts = [
            "Your request to join was denied",
            "You can't join this meeting",
            "denied your request",
        ]
        for text in denied_texts:
            el = page.locator(f'text="{text}"')
            if await el.count() > 0:
                raise RemovedByHostError(f"Join request denied: '{text}'")

        await page.wait_for_timeout(2_000)

    raise HostApprovalTimeoutError(
        f"Host did not approve within {HOST_APPROVAL_TIMEOUT_SEC}s"
    )


# ─── In-meeting controls ────────────────────────────────────────────────
async def mute_mic(page: Page) -> None:
    """Click the mic button to mute. Tolerates already-muted state."""
    # Approach 1: aria-label with mute state
    selectors = [
        '[aria-label*="Turn off microphone"]',
        '[aria-label*="turn off microphone"]',
        '[aria-label*="Mute microphone"]',
        '[data-is-muted="false"][aria-label*="microphone" i]',
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel)
            if await btn.count() > 0:
                await btn.first.click()
                log.info("meeting.mic_muted", selector=sel)
                return
        except Exception:
            continue

    # Approach 2: keyboard shortcut – Ctrl+D
    try:
        await page.keyboard.press("Control+d")
        log.info("meeting.mic_muted", method="keyboard_shortcut")
    except Exception:
        log.warning("meeting.mic_mute_failed")


async def turn_off_camera(page: Page) -> None:
    """Click the camera button to turn off. Tolerates already-off state."""
    selectors = [
        '[aria-label*="Turn off camera"]',
        '[aria-label*="turn off camera"]',
        '[data-is-muted="false"][aria-label*="camera" i]',
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel)
            if await btn.count() > 0:
                await btn.first.click()
                log.info("meeting.camera_off", selector=sel)
                return
        except Exception:
            continue

    # Keyboard shortcut – Ctrl+E
    try:
        await page.keyboard.press("Control+e")
        log.info("meeting.camera_off", method="keyboard_shortcut")
    except Exception:
        log.warning("meeting.camera_off_failed")


# ─── Meeting-end detection ───────────────────────────────────────────────
async def _get_participant_count(page: Page) -> int:
    """
    Attempt to read the participant count from the Meet UI.
    Returns -1 if the count cannot be determined.
    """
    # Method 1: People panel counter badge
    selectors = [
        '[aria-label*="participant" i] span',
        '[data-participant-id]',
        '[aria-label*="people" i] .OA0qNb',
    ]
    for sel in selectors:
        try:
            els = page.locator(sel)
            count = await els.count()
            if count > 0:
                # Try to parse the text as a number
                text = await els.first.inner_text()
                digits = "".join(c for c in text if c.isdigit())
                if digits:
                    return int(digits)
        except Exception:
            continue

    # Method 2: Count participant tiles directly
    try:
        tiles = page.locator('[data-self-name], [data-participant-id], [data-requested-participant-id]')
        count = await tiles.count()
        if count > 0:
            return count
    except Exception:
        pass

    return -1


async def detect_meeting_end(page: Page) -> Optional[str]:
    """
    Check if the meeting has ended.  Returns a reason string or None.

    Reasons:
        "meeting_ended"      — Google's "This meeting has ended" banner
        "removed_by_host"    — "You've been removed from the meeting"
        "participants_left"  — participant count dropped to 0
        "navigated_away"     — page URL left meet.google.com
        None                 — meeting still active
    """
    # ── Banner / overlay detection ───────────────────────────────────
    end_indicators: dict[str, str] = {
        "The meeting has ended": "meeting_ended",
        "Meeting ended": "meeting_ended",
        "You've left the meeting": "meeting_ended",
        "Return to home screen": "meeting_ended",
        "You've been removed from the meeting": "removed_by_host",
        "You've been removed": "removed_by_host",
        "A moderator has removed you": "removed_by_host",
        "Rejoin": "meeting_ended",
    }
    for text, reason in end_indicators.items():
        try:
            el = page.locator(f'text="{text}"')
            if await el.count() > 0:
                return reason
        except Exception:
            continue

    # ── URL navigated away ───────────────────────────────────────────
    try:
        url = page.url
        if "meet.google.com" not in url:
            return "navigated_away"
    except Exception:
        return "navigated_away"

    # ── Participant count ────────────────────────────────────────────
    try:
        count = await _get_participant_count(page)
        # In many Meet UIs, when everyone else leaves, only the bot remains (count == 1).
        if count >= 0 and count <= 1:
            return "participants_left"
    except Exception:
        pass

    return None


# ─── Main bot flow ───────────────────────────────────────────────────────
async def run_bot(
    meeting_url: str,
    recording_output_path: str,
    meeting_id: Optional[str] = None,
    notify_api: bool = True,
) -> ExitCode:
    """
    Complete meeting bot lifecycle.  Returns an ExitCode.
    """
    state = BotState(
        meeting_url=meeting_url,
        recording_output_path=recording_output_path,
    )
    state.save()

    browser: Optional[Browser] = None
    context: Optional[BrowserContext] = None

    try:
        # ── 1. Launch browser ────────────────────────────────────────
        log.info("bot.launching_browser")
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(
            headless=BOT_HEADLESS,
            args=CHROMIUM_ARGS,
        )
        context = await browser.new_context(
            permissions=["microphone", "camera", "notifications"],
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        )
        # Stealth: remove navigator.webdriver flag
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            // Override permissions query to always return 'granted'
            const origQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : origQuery(parameters)
            );
        """)

        page = await context.new_page()

        # ── 2. Google sign-in ────────────────────────────────────────
        state.status = "signing_in"
        state.save()
        await google_sign_in(page)

        # ── 3. Navigate to Meet ──────────────────────────────────────
        state.status = "joining"
        state.save()
        log.info("bot.navigating_to_meet", url=meeting_url)
        await page.goto(meeting_url, wait_until="networkidle", timeout=30_000)
        await page.wait_for_timeout(3_000)

        # ── 4. Handle pre-join screen ────────────────────────────────
        await _dismiss_overlays(page)
        await _detect_meeting_full(page)
        await _set_display_name(page)

        btn_text = await _click_join_button(page)

        # If "Ask to join", wait for host approval
        if "ask" in btn_text.lower():
            await _wait_for_host_approval(page)

        # Give the call a moment to initialize
        await page.wait_for_timeout(5_000)

        # ── 5. Verify we are in the meeting ──────────────────────────
        # After joining, the URL typically stays at meet.google.com/xxx
        # and the meeting UI loads.  Check for common in-meeting elements.
        in_meeting = False
        verify_selectors = [
            '[aria-label*="microphone" i]',
            '[aria-label*="camera" i]',
            '[aria-label*="Leave call" i]',
            '[data-call-id]',
        ]
        for sel in verify_selectors:
            try:
                el = page.locator(sel)
                if await el.count() > 0:
                    in_meeting = True
                    break
            except Exception:
                continue

        if not in_meeting:
            # One more check — the end-detection might fire if we never got in
            end_reason = await detect_meeting_end(page)
            if end_reason == "removed_by_host":
                raise RemovedByHostError("Removed before fully joining")
            elif end_reason:
                log.warning("bot.join_unclear", end_reason=end_reason, url=page.url)

        log.info("bot.in_meeting")

        # ── 6. Mute mic + turn off camera ────────────────────────────
        await mute_mic(page)
        await page.wait_for_timeout(500)
        await turn_off_camera(page)

        state.status = "in_meeting"
        state.join_timestamp = time.time()
        state.save()

        # ── 7. Signal recording start ────────────────────────────────
        state.status = "recording_started"
        state.recording_start_timestamp = time.time()
        state.save()
        log.info("bot.recording_started", output=recording_output_path)

        # ── 8. Poll for meeting end ──────────────────────────────────
        meeting_start = time.monotonic()
        consecutive_end_detections = 0
        log.info("bot.monitoring", poll_interval=POLL_INTERVAL_SEC)

        while True:
            elapsed = time.monotonic() - meeting_start
            state.elapsed_seconds = round(elapsed, 1)

            if elapsed > MAX_MEETING_DURATION_SEC:
                log.info("bot.max_duration_reached", elapsed=elapsed)
                state.end_reason = "timeout"
                break

            end_reason = await detect_meeting_end(page)
            if end_reason:
                # Require stable detection across consecutive polls to avoid transient UI glitches.
                consecutive_end_detections += 1
                log.info(
                    "bot.meeting_end_detected",
                    reason=end_reason,
                    streak=consecutive_end_detections,
                    required=END_DETECTION_STABLE_POLLS,
                    elapsed=f"{elapsed:.0f}s",
                )
                if consecutive_end_detections >= END_DETECTION_STABLE_POLLS:
                    log.info("bot.meeting_ended", reason=end_reason, elapsed=f"{elapsed:.0f}s")
                    state.end_reason = end_reason
                    if end_reason == "removed_by_host":
                        raise RemovedByHostError("Removed by host during meeting")
                    break
            else:
                consecutive_end_detections = 0

            # Update participant count in state
            count = await _get_participant_count(page)
            if count >= 0:
                state.participant_count = count
            state.save()

            await page.wait_for_timeout(POLL_INTERVAL_SEC * 1_000)

        # ── 9. Signal recording stop + clean exit ────────────────────
        state.status = "recording_stopped"
        state.recording_stop_timestamp = time.time()
        state.save()
        log.info("bot.recording_stopped", reason=state.end_reason)

        return ExitCode.OK

    # ── Error handling ───────────────────────────────────────────────
    except MeetingFullError as exc:
        log.error("bot.meeting_full", error=str(exc))
        state.set_error(str(exc))
        state.end_reason = "meeting_full"
        state.status = "recording_stopped"
        state.recording_stop_timestamp = time.time()
        state.save()
        return ExitCode.MEETING_FULL

    except RemovedByHostError as exc:
        log.error("bot.removed_by_host", error=str(exc))
        state.set_error(str(exc))
        state.end_reason = "removed_by_host"
        state.status = "recording_stopped"
        state.recording_stop_timestamp = time.time()
        state.save()
        return ExitCode.REMOVED_BY_HOST

    except HostApprovalTimeoutError as exc:
        log.error("bot.host_approval_timeout", error=str(exc))
        state.set_error(str(exc))
        state.end_reason = "host_approval_timeout"
        state.status = "recording_stopped"
        state.recording_stop_timestamp = time.time()
        state.save()
        return ExitCode.HOST_APPROVAL_TIMEOUT

    except LoginFailedError as exc:
        log.error("bot.login_failed", error=str(exc))
        state.set_error(str(exc))
        return ExitCode.FATAL

    except (ConnectionError, OSError) as exc:
        log.error("bot.network_drop", error=str(exc))
        state.set_error(f"Network error: {exc}")
        state.end_reason = "network_drop"
        state.status = "recording_stopped"
        state.recording_stop_timestamp = time.time()
        state.save()
        return ExitCode.NETWORK_DROP

    except PlaywrightError as exc:
        error_msg = str(exc).lower()
        if "net::" in error_msg or "navigation" in error_msg:
            log.error("bot.network_drop_playwright", error=str(exc))
            state.set_error(f"Network error: {exc}")
            state.end_reason = "network_drop"
            state.status = "recording_stopped"
            state.recording_stop_timestamp = time.time()
            state.save()
            return ExitCode.NETWORK_DROP
        log.exception("bot.playwright_error")
        state.set_error(str(exc))
        return ExitCode.FATAL

    except KeyboardInterrupt:
        log.info("bot.interrupted")
        state.end_reason = "interrupted"
        state.status = "recording_stopped"
        state.recording_stop_timestamp = time.time()
        state.save()
        return ExitCode.OK

    except Exception as exc:
        log.exception("bot.unexpected_error")
        state.set_error(str(exc))
        return ExitCode.FATAL

    finally:
        # ── Cleanup ──────────────────────────────────────────────────
        if state.status != "recording_stopped" and state.recording_start_timestamp:
            state.status = "recording_stopped"
            state.recording_stop_timestamp = time.time()
            state.save()

        if context:
            try:
                await context.close()
            except Exception:
                pass
        if browser:
            try:
                await browser.close()
            except Exception:
                pass
        try:
            await pw.stop()  # type: ignore[possibly-undefined]
        except Exception:
            pass

        log.info("bot.shutdown_complete")

        # ── 7. Notify API ────────────────────────────────────────────
        if meeting_id and notify_api:
            await _notify_api_recording_complete(
                meeting_id=meeting_id,
                audio_path=recording_output_path,
                end_reason=state.end_reason or "unknown"
            )


# ─── CLI entrypoint ──────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="meet_bot",
        description="Magentic-meetbot — Google Meet Playwright bot",
    )
    parser.add_argument(
        "meeting_url",
        nargs="?",
        default=os.getenv("MEET_URL", ""),
        help="Google Meet URL (or set MEET_URL env var)",
    )
    parser.add_argument(
        "recording_output_path",
        nargs="?",
        default=os.getenv("RECORDING_OUTPUT_PATH", "/data/recording.wav"),
        help="Path to write the recording (or set RECORDING_OUTPUT_PATH env var)",
    )
    parser.add_argument(
        "--meeting-id",
        default=os.getenv("MEET_ID", ""),
        help="Internal meeting UUID (or set MEET_ID env var)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if not args.meeting_url:
        log.error("bot.no_meeting_url", hint="Pass as arg or set MEET_URL")
        sys.exit(ExitCode.FATAL)

    if not GOOGLE_EMAIL or not GOOGLE_PASSWORD:
        log.error("bot.missing_credentials", hint="Set GOOGLE_EMAIL and GOOGLE_PASSWORD env vars")
        sys.exit(ExitCode.FATAL)

    log.info(
        "bot.starting",
        meeting_url=args.meeting_url,
        output=args.recording_output_path,
        meeting_id=args.meeting_id,
    )

    exit_code = await run_bot(
        args.meeting_url,
        args.recording_output_path,
        meeting_id=args.meeting_id
    )
    log.info("bot.exiting", code=exit_code.value, name=exit_code.name)
    sys.exit(exit_code.value)


if __name__ == "__main__":
    asyncio.run(main())
