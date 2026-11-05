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

Callers holding durable work ask :meth:`ProviderCircuitBreaker.outage_retry_delay`
when an item fails on a provider error. ``None`` means every breaker is closed,
so the provider is serving other calls and the item owns its failure. A delay
means the provider is down: the item waits that long and is not charged.

State is per process and keyed by model. Each process learns an outage from its
own calls and sends its own probes; with a handful of server processes that is
a few calls per open period.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass
class _Circuit:
    failures: int = 0
    open_until: float | None = None
    open_seconds: float = FIRST_OPEN_SECONDS
    outage_started: float | None = None
    probe_started: float | None = None
    announced: bool = False


class CallTicket:
    """One admitted call. Report its outcome once; ``close`` ends it either way."""

    def __init__(self, breaker: ProviderCircuitBreaker, model: str, probe: bool):
        self._breaker = breaker
        self._model = model
        self._probe = probe
        self._settled = False

    def succeeded(self) -> None:
        if not self._settled:
            self._settled = True
            self._breaker._record(self._model, ok=True, probe=self._probe)

    def failed(self) -> None:
        """The provider was unavailable for this call."""
        if not self._settled:
            self._settled = True
            self._breaker._record(self._model, ok=False, probe=self._probe)

    def close(self) -> None:
        """No verdict (the call ended for an unrelated reason); free a probe."""
        if not self._settled:
            self._settled = True
            if self._probe:
                self._breaker._release_probe(self._model)


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

    def admit(self, model: str) -> CallTicket:
        """Admit one call, or raise :class:`ProviderCircuitOpenError`."""
        announce: tuple[str, float] | None = None
        with self._lock:
            circuit = self._circuits.setdefault(model, _Circuit())
            if circuit.open_until is None:
                return CallTicket(self, model, probe=False)
            now = self._clock()
            announce = self._due_announcement(model, circuit, now)
            probe_free = circuit.probe_started is None or (
                now - circuit.probe_started >= _PROBE_STALE_SECONDS
            )
            if now >= circuit.open_until and probe_free:
                circuit.probe_started = now
                ticket = CallTicket(self, model, probe=True)
            else:
                ticket = None
        if announce is not None:
            self._announce_unavailable(*announce)
        if ticket is None:
            raise ProviderCircuitOpenError(
                f"Provider circuit open for {model}; request not sent"
            )
        return ticket

    def outage_retry_delay(self) -> float | None:
        """Seconds a failed durable item should wait, or ``None`` if no outage.

        The wait runs to the next probe of the longest-open breaker plus a
        random share of its open period, so waiting items do not all return at
        the instant the provider recovers.
        """
        with self._lock:
            now = self._clock()
            delays = [
                max(0.0, circuit.open_until - now)
                + self._jitter() * circuit.open_seconds / 2
                for circuit in self._circuits.values()
                if circuit.open_until is not None
            ]
        return max(delays) if delays else None

    def reset(self) -> None:
        """Forget every breaker. Test-only."""
        with self._lock:
            self._circuits.clear()

    def _record(self, model: str, *, ok: bool, probe: bool) -> None:
        recovered: tuple[str, float] | None = None
        announce: tuple[str, float] | None = None
        with self._lock:
            circuit = self._circuits.setdefault(model, _Circuit())
            now = self._clock()
            if probe:
                circuit.probe_started = None
            if ok:
                if circuit.open_until is not None:
                    outage = now - (circuit.outage_started or now)
                    logger.warning(
                        "event=provider_circuit_closed model=%s outage_seconds=%.0f",
                        model,
                        outage,
                    )
                    if circuit.announced:
                        recovered = (model, outage)
                self._circuits[model] = _Circuit()
            elif circuit.open_until is None:
                circuit.failures += 1
                if circuit.failures >= OPEN_AFTER_FAILURES:
                    circuit.outage_started = now
                    circuit.open_until = now + circuit.open_seconds
                    logger.warning(
                        "event=provider_circuit_opened model=%s "
                        "consecutive_failures=%d retry_in_seconds=%.0f",
                        model,
                        circuit.failures,
                        circuit.open_seconds,
                    )
            elif probe:
                circuit.open_seconds = min(circuit.open_seconds * 2, MAX_OPEN_SECONDS)
                circuit.open_until = now + circuit.open_seconds
                logger.warning(
                    "event=provider_circuit_reopened model=%s "
                    "outage_seconds=%.0f retry_in_seconds=%.0f",
                    model,
                    now - (circuit.outage_started or now),
                    circuit.open_seconds,
                )
                announce = self._due_announcement(model, circuit, now)
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

    def _release_probe(self, model: str) -> None:
        with self._lock:
            circuit = self._circuits.get(model)
            if circuit is not None:
                circuit.probe_started = None

    @staticmethod
    def _due_announcement(
        model: str, circuit: _Circuit, now: float
    ) -> tuple[str, float] | None:
        """Claim the one sustained-outage report, under the lock."""
        if circuit.announced or circuit.outage_started is None:
            return None
        outage = now - circuit.outage_started
        if outage < ANNOUNCE_AFTER_SECONDS:
            return None
        circuit.announced = True
        return model, outage

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


def refused_by_open_circuit(exc: BaseException) -> bool:
    """True when ``exc`` (or its cause chain) is a breaker refusal."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, ProviderCircuitOpenError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def provider_wait_seconds(exc: BaseException) -> float | None:
    """How long durable work that failed with ``exc`` should wait, uncharged.

    ``None`` means charge the failure as usual: either ``exc`` is not a
    provider-availability failure, or the provider is serving other calls
    (every breaker closed), so this item owns it. A breaker refusal is never
    charged -- no request was sent -- even if the breaker closed since.
    """
    from reflexio.server.llm._litellm_text_generation import (
        is_provider_unavailable_error,
    )

    if not is_provider_unavailable_error(exc):
        return None
    delay = provider_circuit.outage_retry_delay()
    if delay is None and refused_by_open_circuit(exc):
        return 0.0
    return delay
