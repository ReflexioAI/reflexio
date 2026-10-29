"""Edge cases of the pure SQLite query-building helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.service_schemas import Status
from reflexio.server.services.storage.sqlite_storage import _base


def test_unicode_lexical_helpers_treat_empty_input_as_no_match():
    assert _base._unicode_lexical_index_text("") == ""
    assert _base._unicode_lexical_index_text(None) == ""
    assert _base._unicode_lexical_score(None, "query") == 0.0
    assert _base._unicode_lexical_score("document", "") == 0.0
    # Punctuation only: no lexical terms, so nothing can match.
    assert _base._unicode_lexical_score("document", "!!! ???") == 0.0
    assert _base._unicode_lexical_score("東京の天気", "東京") == 1.0


def test_unicode_ranking_drops_rows_with_no_term_coverage():
    rows = [{"content": "東京の天気"}, {"content": "unrelated"}]
    ranked = _base._rank_unicode_lexical_rows(rows, "東京", ["content"], 10)
    assert ranked == [{"content": "東京の天気"}]


@pytest.mark.parametrize("text", ["", "   ", "AND OR NOT", "!!!"])
def test_fts_query_without_searchable_tokens_matches_nothing(text):
    assert _base._sanitize_fts_query(text) == '""'


def test_status_sql_for_empty_filter_is_unrestricted():
    assert _base._build_status_sql([]) == ("1=1", [])
    assert _base._status_value(None) is None
    assert _base._status_value("archived") is None  # type: ignore[arg-type]
    assert _base._build_status_sql([None, Status.ARCHIVED]) == (
        "(status IS NULL OR status IN (?))",
        ["archived"],
    )


def test_iso_to_epoch_parses_naive_and_zulu_and_falls_back_to_now():
    assert _base._iso_to_epoch("2026-01-01T00:00:00Z") == int(
        datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    )
    assert _base._iso_to_epoch("2026-01-01T00:00:00") == int(
        datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    )
    before = int(datetime.now(UTC).timestamp())
    for value in (None, "", "not a date"):
        assert _base._iso_to_epoch(value) >= before
