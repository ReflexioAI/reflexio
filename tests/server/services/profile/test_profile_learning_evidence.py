"""Provider-free source addressing and unchanged native compiler controls."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

import pytest

from reflexio.server.services.profile.learning.source_evidence import (
    EvidenceBook,
    current_evidence_book,
)


def window(role="user", text="Delivery entrance is north.") -> list[dict[str, Any]]:
    return [
        {
            "session_id": "session",
            "request_id": "source-request",
            "source": "original",
            "interactions": [
                {
                    "interaction_id": 7,
                    "role": role,
                    "content": text,
                    "user_action_description": "Clicked 'save' — not 'delete'.",
                    "created_at": "original-clock",
                    "tools_used": ["source-tool"],
                }
            ],
        }
    ]


def book(source=None, request="request", user="owner"):
    return EvidenceBook(
        source or window(), user_id=user, request_id=request, org_id="org"
    )


def excerpts(evidence_book, field="content"):
    return evidence_book.rendered_window[0]["interactions"][0][field]["excerpts"]


def test_lossless_long_punctuation_negation_and_action_fields():
    text = (
        "I am still coordinator; don’t remove this.\n“North”—not south! " * 100
    ) + "x" * 370
    original = window(text=text)
    before = deepcopy(original)
    addressed = book(original)
    assert original == before
    assert "".join(p["text"] for p in excerpts(addressed)) == text
    assert all(0 < len(p["text"]) <= 360 for p in excerpts(addressed))
    assert all(
        text[p["offset"] : p["offset"] + len(p["text"])] == p["text"]
        for p in excerpts(addressed)
    )
    rendered = addressed.rendered_window[0]["interactions"][0]
    assert rendered["created_at"] == "original-clock"
    assert rendered["tools_used"] == ["source-tool"]
    action = excerpts(addressed, "user_action_description")[0]
    assert action["excerpt_id"] != excerpts(addressed)[0]["excerpt_id"]
    assert (
        addressed.resolve(action["excerpt_id"])["quote"]
        == before[0]["interactions"][0]["user_action_description"]
    )


@pytest.mark.parametrize("other", ["request", "owner", "source"])
def test_ids_are_request_owner_and_source_bound(other):
    a = book()
    changed = window(text="Delivery entrance is south.") if other == "source" else None
    b = book(
        changed,
        request="other" if other == "request" else "request",
        user="other" if other == "owner" else "owner",
    )
    with pytest.raises(ValueError, match="wrong-request"):
        b.resolve(excerpts(a)[0]["excerpt_id"])
    with pytest.raises(ValueError, match="Unknown"):
        a.resolve("invented")
    assert excerpts(a) == excerpts(book())


def test_duplicate_source_identity_is_rejected_and_roles_are_not_rewritten():
    source = window("assistant")
    assert (
        book(source).resolve(excerpts(book(source))[0]["excerpt_id"])["role"]
        == "assistant"
    )
    source[0]["interactions"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        book(source)


def test_context_is_nested_concurrent_and_exception_safe():
    a, b = book(), book(request="other")
    with pytest.raises(ValueError, match="No request-bound"):
        current_evidence_book()
    with a.bound():
        with pytest.raises(RuntimeError), b.bound():
            assert current_evidence_book() is b
            raise RuntimeError("parse failure")
        assert current_evidence_book() is a
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(
                lambda: pytest.raises(ValueError, current_evidence_book)
            ).result()
    with pytest.raises(ValueError, match="No request-bound"):
        current_evidence_book()
