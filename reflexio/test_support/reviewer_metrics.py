"""Safe per-call request/repair observations for serial reviewer evaluations."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any


class ReviewCallMetrics(logging.Handler):
    """Observe one serial replay call without retaining log text or source data."""

    def __init__(self, model: str, max_tokens: int | None, effort: str | None):
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.effective_max_tokens = max_tokens
        self.effort = effort
        self.owner_thread = threading.get_ident()
        self.log = logging.getLogger("reflexio.server.llm.litellm_client")
        self.previous_level = self.log.level
        self.data: dict[str, Any] = {
            "provider_requests": [],
            "request_ends": 0,
            "repair_attempts": 0,
            "repair_successes": 0,
            "repair_kinds": [],
            "attempt_seconds": [],
            "usage": [],
        }

    def __enter__(self) -> ReviewCallMetrics:
        self.started = time.monotonic()
        self.log.setLevel(logging.INFO)
        self.log.addHandler(self)
        return self

    def __exit__(self, *_: object) -> None:
        self.log.removeHandler(self)
        self.log.setLevel(self.previous_level)
        self.data["wall_seconds"] = time.monotonic() - self.started
        requests = len(self.data["provider_requests"])
        lifecycle_complete = bool(requests and requests == self.data["request_ends"])
        self.data["request_lifecycle_complete"] = lifecycle_complete
        self.data["timing_complete"] = bool(
            lifecycle_complete and len(self.data["attempt_seconds"]) == requests
        )
        self.data["usage_complete"] = bool(
            lifecycle_complete and len(self.data["usage"]) == requests
        )
        self.data["estimated_cost_complete"] = bool(
            self.data["usage_complete"]
            and all("estimated_cost_usd" in usage for usage in self.data["usage"])
        )
        self.data["observation_complete"] = bool(
            self.data["timing_complete"] and self.data["usage_complete"]
        )

    def guard(
        self,
        params: dict[str, Any],
        _timeout: float,
        ladder: tuple[str, ...],
        parse_structured: bool,
    ) -> None:
        """Fail before dispatch if any initial/repair request drifts from settings."""
        expected_body = (
            {
                "reasoning_effort": self.effort,
                "thinking": {
                    "type": "disabled" if self.effort == "none" else "enabled"
                },
            }
            if self.effort
            else {}
        )
        if self.max_tokens is None and not self.data["provider_requests"]:
            self.effective_max_tokens = params.get("max_tokens")
        if (
            params.get("model") != self.model
            or tuple(ladder) != (self.model,)
            or params.get("temperature") != 0.7
            or params.get("num_retries") != 0
            or not parse_structured
            or params.get("max_tokens") != self.effective_max_tokens
            or (params.get("extra_body") or {}) != expected_body
        ):
            raise ValueError("Reviewer inference request drifted from frozen settings")
        # Explicit allowlist: never serialize params, messages, headers or credentials.
        self.data["provider_requests"].append(
            {
                "model": params["model"],
                "temperature": params["temperature"],
                "max_tokens": params.get("max_tokens"),
                "reasoning_effort": self.effort,
            }
        )

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.owner_thread:
            return
        text = record.getMessage()
        if "event=llm_request_end " in text:
            self.data["request_ends"] += 1
            if match := re.search(r"elapsed_seconds=([0-9.]+)", text):
                self.data["attempt_seconds"].append(float(match[1]))
        if "event=llm_structured_repair_attempted " in text:
            self.data["repair_attempts"] += 1
            if match := re.search(
                r"failure_kind=(parse|semantic|blank|refusal)\b", text
            ):
                self.data["repair_kinds"].append(match[1])
        if "event=llm_structured_repair_succeeded " in text:
            self.data["repair_successes"] += 1
        if text.startswith("Token usage - "):
            match = re.search(r"input: (\d+), output: (\d+), total: (\d+)", text)
            if match:
                usage: dict[str, Any] = {
                    "input_tokens": int(match[1]),
                    "output_tokens": int(match[2]),
                    "total_tokens": int(match[3]),
                }
                if cost := re.search(r"cost: \$([0-9.]+)", text):
                    usage["estimated_cost_usd"] = float(cost[1])
                self.data["usage"].append(usage)
