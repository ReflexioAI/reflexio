"""Fusion V2 selection contracts: query policy, identity, caller guards and expiry."""

from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.models.config_schema import SearchMode
from reflexio.server import search_runtime
from reflexio.server.services.retrieval.profile_selection import (
    lexical_query,
    select_profiles,
)
from reflexio.server.services.storage.error import ReadinessUnavailableError

pytestmark = pytest.mark.unit


def profile(key):
    return UserProfile(
        profile_id=key,
        user_id="u",
        content=key,
        last_modified_timestamp=1,
        generated_from_request_id="r",
    )


def choose(items, secondary, **changes):
    fetch = Mock(return_value=secondary)
    options = {
        "items": items,
        "request": UnifiedSearchRequest(query="Python payments", user_id="u"),
        "query": "Python payments",
        "effective_mode": SearchMode.HYBRID,
        "query_embedding": [0.1],
        "degraded": False,
        "floor_on": False,
        "temporal_reorder": False,
        "retrieve": fetch,
    }
    options.update(changes)
    return (select_profiles(**options), fetch)


def test_broad_fusion_protects_three_and_preserves_native_object_identity():
    primary = [profile(k) for k in "ABCDE"]
    secondary = [profile(k) for k in "AFDGBCE"]
    selected, fetch = choose(primary, secondary)
    assert [p.profile_id for p in selected] == list("FDABC")
    assert selected[1] is primary[3] and selected[2:] == primary[:3]
    assert selected[0] is secondary[1]
    fetch.assert_called_once_with(
        query="python OR payments", top_k=20, search_mode=SearchMode.FTS
    )


def test_sparse_fill_keeps_prefix_appends_unique_and_uses_hybrid():
    primary = [profile(k) for k in "AB"]
    selected, fetch = choose(primary, [profile(k) for k in "BACDE"])
    assert selected[:2] == primary
    assert [p.profile_id for p in selected] == list("ABCDE")
    fetch.assert_called_once_with(
        query="Python payments", top_k=5, search_mode=SearchMode.HYBRID
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"items": []},
        {"degraded": True},
        {"query_embedding": None},
        {"effective_mode": SearchMode.FTS},
        {"effective_mode": SearchMode.VECTOR},
        {"floor_on": True},
        {"temporal_reorder": True},
        {"request": UnifiedSearchRequest(query="q", user_id=None)},
        {"request": UnifiedSearchRequest(query="q", user_id="u", threshold=0.0)},
        {"request": UnifiedSearchRequest(query="q", user_id="u", top_k=3)},
        {
            "request": UnifiedSearchRequest(
                query="q", user_id="u", entity_types=["user_playbooks"]
            )
        },
    ],
)
def test_caller_and_failure_bypasses_never_retrieve(changes):
    primary = changes.get("items", [profile(k) for k in "ABCDE"])
    selected, fetch = choose(
        primary, [], **{k: v for k, v in changes.items() if k != "items"}
    )
    assert selected is primary
    fetch.assert_not_called()


def test_empty_terms_and_small_pool_preserve_native_five(monkeypatch):
    increment = Mock()
    monkeypatch.setattr(search_runtime, "increment", increment)
    primary = [profile(k) for k in "ABCDE"]
    selected, fetch = choose(primary, [], query="a an of it xy")
    assert selected is primary
    fetch.assert_not_called()
    increment.assert_not_called()
    selected, fetch = choose(primary, [profile(k) for k in "FGHAB"])
    assert selected is primary
    fetch.assert_called_once()
    increment.assert_called_once_with("retrieval.profile_secondary")


@pytest.mark.parametrize("where", ["primary", "secondary"])
def test_duplicate_identity_is_a_failure(where):
    primary = [profile(k) for k in ("AABCD" if where == "primary" else "ABCDE")]
    secondary = [profile(k) for k in "FFGHIJ"]
    with pytest.raises(ReadinessUnavailableError):
        choose(primary, secondary)


def test_query_uses_complete_source_order_unicode_casefold_and_exact_char_limit():
    assert (
        lexical_query("THE Straße, Python Python payments in CI")
        == "strasse OR python OR payments"
    )
    words = [f"term{i}" for i in range(1000)]
    actual = lexical_query(" ".join(words))
    assert len(actual) <= 2000
    expected = []
    for word in words:
        if len(" OR ".join([*expected, word])) > 2000:
            break
        expected.append(word)
    assert actual == " OR ".join(expected)
    assert lexical_query(" ".join(["the"] * 5000 + ["payments"])) == "payments"


def test_empty_entity_types_follow_native_default():
    primary = [profile(k) for k in "ABCDE"]
    selected, fetch = choose(
        primary,
        [profile(k) for k in "FGHIJK"],
        request=UnifiedSearchRequest(query="q", user_id="u", entity_types=[]),
    )
    assert [p.profile_id for p in selected] == list("FGABC")
    fetch.assert_called_once()


def test_expiry_during_query_scan_or_after_retrieval_is_not_a_success(monkeypatch):
    calls = []

    def expire():
        calls.append(True)
        if len(calls) == 2:
            raise search_runtime.SearchDeadlineError()

    monkeypatch.setattr(search_runtime, "checkpoint", expire)
    with pytest.raises(search_runtime.SearchDeadlineError):
        lexical_query(" ".join(["the"] * 1000))
    calls.clear()
    with pytest.raises(search_runtime.SearchDeadlineError):
        choose([profile("A")], [profile("B")])
