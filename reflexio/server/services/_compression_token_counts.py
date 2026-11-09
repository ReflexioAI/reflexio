"""Exact, bounded count reuse owned by one compression plan, never global."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator

_CL100K_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)
_BOUNDARIES = re.compile(r'(?<=[A-Za-z0-9])(?=[\\"])')
_MAX_ENTRIES = 32_768


class ExactTokenCounts:
    """Reuse counts only across proven cl100k pretokenization boundaries.

    Letters/digits end before backslash or quote; punctuation cannot absorb the
    preceding alphanumeric. BPE operates within regex pieces, so these cuts keep
    complete token sequences unchanged. Arbitrary token counts are NOT additive.
    Unknown patterns use the original full-text counter. Cache keys retain only
    digests, not source text. Locking permits repeated/concurrent plan execution;
    the original counter runs outside the lock, including on simultaneous misses.
    """

    def __init__(self, original: Callable[[str], int], *, pattern: str) -> None:
        self._original = original
        self._enabled = pattern == _CL100K_PATTERN
        self._cache: OrderedDict[bytes, int] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def chunks(text: str) -> Iterator[str]:
        start = 0
        for match in _BOUNDARIES.finditer(text):
            end = match.start()
            yield text[start:end]
            start = end
        yield text[start:]

    def __call__(self, text: str) -> int:
        if not self._enabled:
            return self._original(text)
        return sum(self._chunk_count(chunk) for chunk in self.chunks(text))

    def _chunk_count(self, text: str) -> int:
        key = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).digest()
        with self._lock:
            value = self._cache.get(key)
            if value is not None:
                self._cache.move_to_end(key)
                return value
        value = self._original(text)
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > _MAX_ENTRIES:
                self._cache.popitem(last=False)
        return value
