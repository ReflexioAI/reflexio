"""Exact-count reuse must not change tokenizer or request admission semantics."""

import hashlib
import random
from concurrent.futures import ThreadPoolExecutor

import pytest
import tiktoken

from reflexio.server.services import _compression_token_counts as counts
from reflexio.server.services.input_compression import compression_tokens

pytestmark = pytest.mark.unit


def test_chunks_preserve_complete_token_ids_and_counts():
    encoding = tiktoken.get_encoding("cl100k_base")
    counter = counts.ExactTokenCounts(compression_tokens, pattern=encoding._pat_str)
    randomizer = random.Random(903)  # noqa: S311 - reproducible tokenizer fixtures
    alphabet = 'abcXYZ0123\\"\n\r\t <>|_😀中文é\u0301\ud800'
    texts = ["", "<|endoftext|>", '123456"x"', "a\ud800\\n", 'a"\\n中文😀'] + [
        "".join(randomizer.choices(alphabet, k=randomizer.randrange(1, 2500)))
        for _ in range(200)
    ]
    for text in texts:
        expected = encoding.encode(text, disallowed_special=())
        actual = [
            token
            for chunk in counter.chunks(text)
            for token in encoding.encode(chunk, disallowed_special=())
        ]
        assert actual == expected
        assert counter(text) == counter(text) == len(expected)


def test_unknown_pattern_uses_original_full_text_without_caching():
    seen = []

    def original(text):
        seen.append(text)
        return 17

    counter = counts.ExactTokenCounts(original, pattern="unrecognized")
    assert counter('a"b\\c') == counter('a"b\\c') == 17
    assert seen == ['a"b\\c', 'a"b\\c']
    assert not counter._cache


def test_cache_bounds_eviction_and_never_retains_raw_source(monkeypatch):
    monkeypatch.setattr(counts, "_MAX_ENTRIES", 3)
    seen = []

    def original(text):
        seen.append(text)
        return len(text)

    counter = counts.ExactTokenCounts(original, pattern=counts._CL100K_PATTERN)
    for text in ("private-a", "private-b", "private-c", "private-a", "private-d"):
        assert counter(text) == len(text)
    assert len(counter._cache) == 3
    assert hashlib.sha256(b"private-b").digest() not in counter._cache
    assert seen.count("private-a") == 1
    assert all(isinstance(key, bytes) and len(key) == 32 for key in counter._cache)
    assert all(isinstance(value, int) for value in counter._cache.values())


def test_counter_failure_is_preserved_not_cached():
    error = ValueError("counter failed")

    def fail(_text):
        raise error

    counter = counts.ExactTokenCounts(fail, pattern=counts._CL100K_PATTERN)
    with pytest.raises(ValueError) as caught:
        counter('abc"def')
    assert caught.value is error
    assert not counter._cache


def test_concurrent_execution_and_plan_caches_are_independent(monkeypatch):
    monkeypatch.setattr(counts, "_MAX_ENTRIES", 3)
    counter = counts.ExactTokenCounts(
        compression_tokens, pattern=counts._CL100K_PATTERN
    )
    other = counts.ExactTokenCounts(compression_tokens, pattern=counts._CL100K_PATTERN)
    texts = [f'entry{index}\\nnext"' for index in range(60)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(counter, texts * 3)) == [
            compression_tokens(text) for text in texts * 3
        ]
    assert len(counter._cache) <= 3
    assert not other._cache
