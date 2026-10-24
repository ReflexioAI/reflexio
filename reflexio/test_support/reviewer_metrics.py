"""Safe per-call request/repair observations for serial reviewer evaluations."""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from reflexio.server.llm._litellm_text_generation import (
    TextGenerationMixin,
    _response_hidden_params,
)


class ReviewCallMetrics(logging.Handler):
    """Observe one serial replay call without retaining log text or source data."""

    def __init__(
        self,
        model: str,
        max_tokens: int | None,
        effort: str | None,
        *,
        expected_response_model: str | None = None,
        expected_temperature: float = 0.7,
    ):
        super().__init__()
        self.model = model
        if (
            isinstance(expected_temperature, bool)
            or not math.isfinite(expected_temperature)
            or not 0 <= expected_temperature <= 2
        ):
            raise ValueError("expected_temperature must be finite and within [0, 2]")
        self.expected_temperature = expected_temperature
        self.request_drifted = False
        self.expected_response_model = (
            model if expected_response_model is None else expected_response_model
        )
        if not self.expected_response_model.strip():
            raise ValueError("expected_response_model must be nonblank")
        self.max_tokens = max_tokens
        self.effective_max_tokens = max_tokens
        self.effort = effort
        self.response_observation_failed = False
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
            "provider_responses": [],
            "response_observation_errors": [],
            "expected_response_model": self.expected_response_model,
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
        responses = self.data["provider_responses"]
        self.data["model_identity_complete"] = bool(
            lifecycle_complete
            and len(responses) == requests
            and not self.response_observation_failed
            and not self.data["response_observation_errors"]
            and all(
                response["model"]
                in (
                    self.expected_response_model,
                    self.expected_response_model.split("/", 1)[-1],
                )
                for response in responses
            )
        )
        self.data["observation_complete"] = bool(
            not self.request_drifted
            and self.data["timing_complete"]
            and self.data["usage_complete"]
            and self.data["model_identity_complete"]
        )

    @contextmanager
    def observe_responses(self, client: Any) -> Iterator[None]:
        """Project safe metadata at the raw-response seam; never alter responses.

        Only an omitted provider prefix is normalized, never a model-version alias.
        This instance-local observer covers initial and repair completions alike.
        """
        original = client._completion_with_hard_timeout
        attributes = vars(client)
        had_override = "_completion_with_hard_timeout" in attributes
        previous_override = attributes.get("_completion_with_hard_timeout")

        def observed(*args: Any, **kwargs: Any) -> Any:
            response = original(*args, **kwargs)
            try:
                provenance = TextGenerationMixin._build_model_provenance(
                    client, response
                )
                model = provenance.model_name
                if self.model.lower().startswith("claude-code/"):
                    hidden = _response_hidden_params(response)
                    # Public response.model is the requested bridge route, not
                    # served identity. Missing bridge metadata cannot qualify it.
                    if not all(
                        isinstance(hidden.get(key), str)
                        and re.fullmatch(r"[\w./:@+-]{1,200}", hidden[key])
                        for key in ("reflexio_served_model", "reflexio_served_provider")
                    ):
                        model = None
                if not isinstance(model, str) or not re.fullmatch(
                    r"[\w./:@+-]{1,200}", model
                ):
                    model = None
                raw_choices = response.choices
                if len(raw_choices) != 1:
                    raise ValueError("Expected exactly one completion choice")
                choices = []
                for choice in raw_choices:
                    message = choice.message
                    content = getattr(message, "content", None)
                    reasoning = getattr(message, "reasoning_content", None)
                    finish = getattr(choice, "finish_reason", None)
                    choices.append(
                        {
                            "finish_reason": finish
                            if finish
                            in (
                                "stop",
                                "length",
                                "tool_calls",
                                "function_call",
                                "content_filter",
                            )
                            else None,
                            "content_chars": len(content)
                            if isinstance(content, str)
                            else None,
                            "reasoning_chars": len(reasoning)
                            if isinstance(reasoning, str)
                            else None,
                        }
                    )
                self.data["provider_responses"].append(
                    {"model": model, "choices": choices}
                )
            except Exception as exc:
                self.response_observation_failed = True
                # Observation failures must never replace a received response.
                with suppress(Exception):
                    self.data["response_observation_errors"].append(type(exc).__name__)
                    self.data["provider_responses"].append({"model": None})
            return response

        client._completion_with_hard_timeout = observed
        try:
            yield
        finally:
            if had_override:
                client._completion_with_hard_timeout = previous_override
            else:
                del client._completion_with_hard_timeout

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
            or params.get("temperature") != self.expected_temperature
            or params.get("num_retries") != 0
            or not parse_structured
            or params.get("max_tokens") != self.effective_max_tokens
            or (params.get("extra_body") or {}) != expected_body
        ):
            self.request_drifted = True
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
                r"failure_kind=(parse|semantic|blank|refusal|empty|reasoning_exhausted)\b",
                text,
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
