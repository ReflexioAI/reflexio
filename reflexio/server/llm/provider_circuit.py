"""Per-process circuit breaker for LLM providers.

Durable work must survive a provider outage of any length without spending
its retry allowance. That needs one fact the work itself cannot see: is the
provider failing for everyone, or only for this item? The breaker answers it
from the calls this process has made.

- **Closed** (normal): calls go through. ``OPEN_AFTER_FAILURES`` provider
  failures in a row open it.
- **Open**: calls are refused without reaching the provider
  (:class:`ProviderCircuitOpenError`). Once the open period ends, exactly one
  call is admitted as a probe.
- A probe that succeeds closes the breaker; one that fails re-opens it for
  twice as long, up to ``MAX_OPEN_SECONDS``. Any successful call closes it.

Only provider-availability failures (timeouts, connection errors, rate limits,
overload; the caller decides) move the breaker. A malformed or invalid answer
is still an answer: the provider is up.

**Charging invariant.** A durable item's failure spends one of its retry or
attempt allowances only when the provider was serving other calls. Every such
allowance asks :func:`reflexio.server.llm.failure.classify_llm_failure` about
the failure: any verdict but ``OutageWait`` means the failing route's breaker
is closed (or the failure is not the provider's), so the item owns its failure
and is charged; ``OutageWait(delay_seconds)`` means the provider is down for
everyone, or the call was never sent (a refusal, or our own provider cap was
full), so the item waits that long and is not charged. The allowances that
apply it:

- extraction window attempts (``DurableLearningWorker``);
- resume attempts (``ExtractionResumeWorker``);
- finalization attempts (``publication.settle_finalization_failure``);
- the overflow attempt journal, with its package stage caps and
  reconciliation allowance (``OverflowCoordinator._settle_failure``);
- agent-success evaluation retries (``generation_service``), deferred tagging
  passes (``tagging_scheduler``) and offline-tuner rule-evidence attempts
  (``rule_evidence_runtime``).

The failures that open the breaker happen while it is closed and are charged:
until then nothing tells an outage from the item's own failure.

State is per process and keyed by route: the model, its endpoint and a one-way
digest of the credential, so one account's or endpoint's outage never blocks
another's. A failure carries its route (``circuit_route``), and only that
route's breaker decides whether durable work is charged. Each process learns an
outage from its own calls and sends its own probes; with a handful of server
processes that is a few calls per open period.
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from reflexio.server.error_reporting import capture_anomaly
from reflexio.server.llm._litellm_types import LiteLLMClientError

logger = logging.getLogger(__name__)

OPEN_AFTER_FAILURES = 5
FIRST_OPEN_SECONDS = 60.0
MAX_OPEN_SECONDS = 1800.0
# One report per outage that lasts this long; recovery is then reported too.
ANNOUNCE_AFTER_SECONDS = 1800.0
# A probe whose caller never reported back (killed thread, refused before
# dispatch) stops blocking the next probe after this long. Longer than any
# single call can last (the ladder's own 600 s wall clock).
_PROBE_STALE_SECONDS = 900.0

UNAVAILABLE_SIGNAL = "llm.provider_unavailable"
RECOVERED_SIGNAL = "llm.provider_recovered"


class ProviderCircuitOpenError(LiteLLMClientError):
    """The provider's breaker is open; no request was sent."""

    quota_exhausted: bool = False


def circuit_route(params: Mapping[str, Any]) -> str:
    """Route identity of a completion: model, endpoint, credential digest.

    The credential appears only as a truncated SHA-256 digest, so the route is
    safe to log and to compare without carrying the secret.
    """
    credential = params.get("api_key")
    digest = (
        hashlib.sha256(str(credential).encode()).hexdigest()[:12] if credential else "-"
    )
    endpoint = params.get("api_base") or params.get("base_url") or ""
    return f"{params.get('model')}|{endpoint}|{digest}"


def _route_ref(route: str) -> str:
    """Short value-free handle that tells two routes of one model apart."""
    return hashlib.sha256(route.encode()).hexdigest()[:8]


@dataclass
class _Circuit:
    model: str = ""
    failures: int = 0
    open_until: float | None = None
    open_seconds: float = FIRST_OPEN_SECONDS
    outage_started: float | None = None
    probe_started: float | None = None
    announced: bool = False
    quota_exhausted: bool = False


class CallTicket:
    """One admitted call. Report its outcome once; ``close`` ends it either way."""

    def __init__(self, breaker: ProviderCircuitBreaker, route: str, probe: bool):
        self._breaker = breaker
        self.route = route
        self._probe = probe
        self._settled = False

    def succeeded(self) -> None:
        if not self._settled:
            self._settled = True
            self._breaker._record(self.route, ok=True, probe=self._probe)

    def failed(self, *, quota_exhausted: bool = False) -> None:
        """The provider was unavailable for this call."""
        if not self._settled:
            self._settled = True
            self._breaker._record(
                self.route, ok=False, probe=self._probe, quota_exhausted=quota_exhausted
            )

    def close(self) -> None:
        """No verdict (the call ended for an unrelated reason); free a probe."""
        if not self._settled:
            self._settled = True
            if self._probe:
                self._breaker._release_probe(self.route)


class ProviderCircuitBreaker:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._clock = clock
        self._jitter = jitter
        self._lock = threading.Lock()
        self._circuits: dict[str, _Circuit] = {}

    def admit(self, route: str, *, model: str) -> CallTicket:
        """Admit one call on ``route``, or raise :class:`ProviderCircuitOpenError`."""
        announce: tuple[str, float] | None = None
        with self._lock:
            circuit = self._circuits.setdefault(route, _Circuit(model=model))
            if circuit.open_until is None:
                return CallTicket(self, route, probe=False)
            now = self._clock()
            announce = self._due_announcement(circuit, now)
            probe_free = circuit.probe_started is None or (
                now - circuit.probe_started >= _PROBE_STALE_SECONDS
            )
            if now >= circuit.open_until and probe_free:
                circuit.probe_started = now
                ticket = CallTicket(self, route, probe=True)
            else:
                ticket = None
            quota_exhausted = circuit.quota_exhausted
        if announce is not None:
            self._announce_unavailable(*announce)
        if ticket is None:
            refused = ProviderCircuitOpenError(
                f"Provider circuit open for {model}; request not sent"
            )
            refused.circuit_route = route
            refused.quota_exhausted = quota_exhausted
            raise refused
        return ticket

    def outage_retry_delay(self, route: str) -> float | None:
        """Seconds a durable item that failed on ``route`` should wait.

        ``None`` while that route's breaker is closed. Otherwise the wait runs
        to its next probe plus a random share of its open period, so waiting
        items do not all return at the instant the provider recovers.
        """
        with self._lock:
            circuit = self._circuits.get(route)
            if circuit is None or circuit.open_until is None:
                return None
            return (
                max(0.0, circuit.open_until - self._clock())
                + self._jitter() * circuit.open_seconds / 2
            )

    def refusing_model(self, model: str) -> bool:
        """Whether some route of ``model`` is open and not yet due for its probe.

        For best-effort callers that would rather skip a call than spend their
        budget on a provider that is down. Unlike the charging decision, this
        reads every route of the model, not one: such a caller cannot name its
        route before the client resolves the credential, and skipping costs
        only the optional result. A route's own probe is never suppressed: once
        its open period ends this returns ``False`` again.
        """
        with self._lock:
            now = self._clock()
            return any(
                circuit.model == model
                and circuit.open_until is not None
                and now < circuit.open_until
                for circuit in self._circuits.values()
            )

    def reset(self) -> None:
        """Forget every breaker. Test-only."""
        with self._lock:
            self._circuits.clear()

    def _record(
        self, route: str, *, ok: bool, probe: bool, quota_exhausted: bool = False
    ) -> None:
        recovered: tuple[str, float] | None = None
        announce: tuple[str, float] | None = None
        with self._lock:
            circuit = self._circuits.get(route)
            if circuit is None:  # forgotten by ``reset`` while the call ran
                return
            model, ref = circuit.model, _route_ref(route)
            now = self._clock()
            if probe:
                circuit.probe_started = None
            circuit.quota_exhausted |= quota_exhausted
            if ok:
                if circuit.open_until is not None:
                    outage = now - (circuit.outage_started or now)
                    logger.warning(
                        "event=provider_circuit_closed model=%s route=%s "
                        "outage_seconds=%.0f",
                        model,
                        ref,
                        outage,
                    )
                    if circuit.announced:
                        recovered = (model, outage)
                self._circuits[route] = _Circuit(model=model)
            elif circuit.open_until is None:
                circuit.failures += 1
                if quota_exhausted or circuit.failures >= OPEN_AFTER_FAILURES:
                    circuit.outage_started = now
                    circuit.open_until = now + circuit.open_seconds
                    logger.warning(
                        "event=provider_circuit_opened model=%s route=%s "
                        "consecutive_failures=%d retry_in_seconds=%.0f",
                        model,
                        ref,
                        circuit.failures,
                        circuit.open_seconds,
                    )
            elif probe:
                circuit.open_seconds = min(circuit.open_seconds * 2, MAX_OPEN_SECONDS)
                circuit.open_until = now + circuit.open_seconds
                logger.warning(
                    "event=provider_circuit_reopened model=%s route=%s "
                    "outage_seconds=%.0f retry_in_seconds=%.0f",
                    model,
                    ref,
                    now - (circuit.outage_started or now),
                    circuit.open_seconds,
                )
                announce = self._due_announcement(circuit, now)
            # A non-probe failure while open was dispatched before the breaker
            # opened; it adds nothing.
        if announce is not None:
            self._announce_unavailable(*announce)
        if recovered is not None:
            capture_anomaly(
                RECOVERED_SIGNAL,
                level="warning",
                signal=RECOVERED_SIGNAL,
                release_blocking="false",
                model=recovered[0],
                outage_minutes=round(recovered[1] / 60),
            )

    def _release_probe(self, route: str) -> None:
        with self._lock:
            circuit = self._circuits.get(route)
            if circuit is not None:
                circuit.probe_started = None

    @staticmethod
    def _due_announcement(circuit: _Circuit, now: float) -> tuple[str, float] | None:
        """Claim the one sustained-outage report, under the lock."""
        if circuit.announced or circuit.outage_started is None:
            return None
        outage = now - circuit.outage_started
        if outage < ANNOUNCE_AFTER_SECONDS:
            return None
        circuit.announced = True
        return circuit.model, outage

    @staticmethod
    def _announce_unavailable(model: str, outage: float) -> None:
        # A provider outage is not a defect of the running release, so it must
        # not hold a deploy; it is routed by ``signal``, not by level.
        capture_anomaly(
            UNAVAILABLE_SIGNAL,
            level="warning",
            signal=UNAVAILABLE_SIGNAL,
            release_blocking="false",
            model=model,
            outage_minutes=round(outage / 60),
        )


provider_circuit = ProviderCircuitBreaker()
