"""Safety guardrails for Monarch inference."""

import re
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime


@dataclass
class GuardResult:
    """Result of a guard check."""
    allowed: bool
    reason: Optional[str] = None


class MonarchGuardrails:
    """Input/output safety checks and rate limiting for Monarch."""

    def __init__(self, config: dict):
        """Initialize guardrails from config dict.

        Args:
            config: Dict with keys like max_input_length, rate_limit, blocked_keywords, log_file
        """
        self.enabled = config.get("enabled", True)
        self.max_input_length = config.get("max_input_length", 2000)

        rate_limit_config = config.get("rate_limit", {})
        self.rate_limit_requests = rate_limit_config.get("requests", 20)
        self.rate_limit_window = rate_limit_config.get("window_seconds", 60)

        self.blocked_keywords = config.get("blocked_keywords", [])
        self.log_file = Path(config.get("log_file", "logs/guardrail_events.jsonl"))

        # In-memory rate limit tracking: session_id -> list of timestamps
        self._rate_limit_tracker: Dict[str, list] = {}

        # Compile jailbreak patterns
        self._jailbreak_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
            r"pretend\s+(you are|to be|that)",
            r"you\s+(are|must)\s+now\s+(act as|be|respond as)",
            r"(DAN|do anything now)",
            r"\[?system\]?:",
            r"<\|system\|>",
            r"forget\s+(your|all)\s+(training|guidelines|rules|restrictions)",
            r"jailbreak",
            r"bypass\s+(your\s+)?(safety|filter|guardrails)",
        ]
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self._jailbreak_patterns]

    def check_input(self, prompt: str) -> GuardResult:
        """Check input prompt for safety issues.

        Returns GuardResult with allowed=False and reason if blocked.
        """
        if not self.enabled:
            return GuardResult(allowed=True)

        # 1. Length check
        if len(prompt) > self.max_input_length:
            return GuardResult(
                allowed=False,
                reason=f"Input exceeds max length of {self.max_input_length} chars"
            )

        # 2. Jailbreak pattern check
        for pattern in self._compiled_patterns:
            if pattern.search(prompt):
                return GuardResult(
                    allowed=False,
                    reason="Detected jailbreak attempt"
                )

        # 3. Blocked keywords check
        prompt_lower = prompt.lower()
        for keyword in self.blocked_keywords:
            if keyword.lower() in prompt_lower:
                return GuardResult(
                    allowed=False,
                    reason=f"Input contains blocked content"
                )

        return GuardResult(allowed=True)

    def check_output(self, response: str) -> GuardResult:
        """Check output response for safety issues.

        Returns GuardResult with allowed=False and reason if blocked.
        """
        if not self.enabled:
            return GuardResult(allowed=True)

        # Check for blocked keywords in output
        response_lower = response.lower()
        for keyword in self.blocked_keywords:
            if keyword.lower() in response_lower:
                return GuardResult(
                    allowed=False,
                    reason="Output contains blocked content"
                )

        return GuardResult(allowed=True)

    def check_rate_limit(self, session_id: str) -> GuardResult:
        """Check rate limit for session.

        Uses sliding window: tracks timestamps and removes old ones.
        """
        if not self.enabled:
            return GuardResult(allowed=True)

        now = time.time()

        # Initialize if needed
        if session_id not in self._rate_limit_tracker:
            self._rate_limit_tracker[session_id] = []

        # Remove timestamps outside the window
        cutoff = now - self.rate_limit_window
        self._rate_limit_tracker[session_id] = [
            ts for ts in self._rate_limit_tracker[session_id]
            if ts > cutoff
        ]

        # Check if over limit
        if len(self._rate_limit_tracker[session_id]) >= self.rate_limit_requests:
            return GuardResult(
                allowed=False,
                reason=f"Rate limit exceeded ({self.rate_limit_requests} per {self.rate_limit_window}s)"
            )

        # Add current request
        self._rate_limit_tracker[session_id].append(now)

        return GuardResult(allowed=True)

    def log_event(self, event_type: str, prompt: str, reason: Optional[str] = None) -> None:
        """Log a guardrail event to JSON lines file.

        Args:
            event_type: BLOCKED_INPUT, BLOCKED_OUTPUT, RATE_LIMITED, or ALLOWED
            prompt: The input prompt
            reason: Optional reason for blocking
        """
        # Create logs directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        prompt_preview = prompt[:200]  # First 200 chars

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "prompt_preview": prompt_preview,
            "reason": reason,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
