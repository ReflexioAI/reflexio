# ruff: noqa: S608 -- Fixed SQLite expression/owned fixture SQL, no caller text.
"""Built-in SQLite candidate index parity and old-writer compatibility."""

import json
import math
import sqlite3
from threading import RLock
from typing import Any

import pytest

from reflexio.server.services.profile.learning.explicit_retention import KEYS, due_epoch
from reflexio.server.services.storage.sqlite_storage.profiles._explicit_retention import (
    DEADLINE_INDEX,
    ExplicitRetentionMixin,
    deadline_projection_sql,
    install_deadline_index,
)


def authority(
    deadline: Any = 100, boundary="stop_at", key=KEYS[0], basis="explicit_retention"
):
    return {key: {"basis": basis, "logical_deadline": deadline, "boundary": boundary}}


CASES = [
    None,
    {},
    {"unrelated": 1},
    [],
    "text",
    False,
    *[authority(key=key) for key in KEYS],
    authority(boundary="through"),
    authority(-10),
    authority(0),
    authority(2**63 - 1, boundary="through"),
    authority(10**100),
    authority(-(10**100)),
    *[authority(value) for value in (True, False, 100.0, "100", None)],
    authority(boundary="invalid"),
    authority(basis="system_estimate"),
    {KEYS[0]: None},
    {KEYS[0]: []},
    {KEYS[0]: {}},
    {**authority(100), **authority(100, key=KEYS[1])},
    {**authority(100), **authority(101, key=KEYS[1])},
    {**authority(100), **authority(100.0, key=KEYS[1])},
]


class Store(ExplicitRetentionMixin):
    def __init__(self, conn):
        self.conn, self._lock = conn, RLock()


@pytest.fixture
def indexed(tmp_path):
    path = tmp_path / "indexed.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE profiles(profile_id TEXT PRIMARY KEY, user_id TEXT, content TEXT, last_modified_timestamp INTEGER, expiration_timestamp INTEGER, source_span TEXT, custom_features TEXT)"
    )
    install_deadline_index(conn)
    yield path, Store(conn)
    conn.close()


@pytest.mark.parametrize("features", CASES)
def test_sqlite_projection_is_exact_or_conservative_never_drops_authority(
    indexed, features
):
    _, store = indexed
    raw = json.dumps(features)
    store.conn.execute(
        "INSERT INTO profiles(profile_id,user_id,custom_features) VALUES ('row','owner',?)",
        (raw,),
    )
    projected = store.conn.execute(
        "SELECT " + deadline_projection_sql() + " FROM profiles"
    ).fetchone()[0]
    try:
        expected = due_epoch(features)
    except ValueError:
        assert projected == -math.inf
        with pytest.raises(ValueError):
            store.next_profile_deadline(user_id="owner")
    else:
        if expected is None:
            assert projected is None or projected == -math.inf
        else:
            assert projected is not None
            assert projected in (-math.inf, expected)
        assert store.next_profile_deadline(user_id="owner") == expected


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"rsi_lifetime_v8":',
        '{"rsi_lifetime_v8":{"basis":"system_estimate"},"rsi_lifetime_v8":{"basis":"explicit_retention","logical_deadline":5,"boundary":"through"}}',
        '{"rsi_lifetime_v8":{"basis":"system_estimate","basis":"explicit_retention","logical_deadline":5,"boundary":"stop_at"}}',
        '{"rsi_lifetime_v8":{"basis":"explicit_retention","logical_deadline":100,"logical_deadline":5,"boundary":"stop_at"}}',
        '{"\\u0072si_lifetime_v8":{"basis":"explicit_retention","logical_deadline":5,"boundary":"stop_at"}}',
    ],
)
def test_raw_old_writer_ambiguous_or_invalid_json_cannot_break_index_dml(indexed, raw):
    path, store = indexed
    # Independent connection registers no Python functions, as with an old image.
    with sqlite3.connect(path) as old:
        old.execute(
            "INSERT INTO profiles(profile_id,user_id,custom_features) VALUES ('old','owner',?)",
            (raw,),
        )
    projected = store.conn.execute(
        "SELECT " + deadline_projection_sql() + " FROM profiles"
    ).fetchone()[0]
    assert projected == -math.inf
    try:
        expected = due_epoch(json.loads(raw))
    except ValueError:
        with pytest.raises(ValueError):
            store.next_profile_deadline(user_id="owner")
    else:
        assert store.next_profile_deadline(user_id="owner") == expected


def test_uncertain_accepted_deadline_competes_with_regular_minimum_and_old_updates(
    indexed,
):
    path, store = indexed
    with sqlite3.connect(path) as old:
        old.executemany(
            "INSERT INTO profiles(profile_id,user_id,custom_features) VALUES (?,'owner',?)",
            [
                (
                    "ambiguous",
                    json.dumps({**authority(200), **authority(200, key=KEYS[1])}),
                ),
                ("regular", json.dumps(authority(100))),
            ],
        )
    assert store.next_profile_deadline(user_id="owner") == 100
    with sqlite3.connect(path) as old:
        old.execute(
            "UPDATE profiles SET custom_features=? WHERE profile_id='ambiguous'",
            (json.dumps({**authority(50), **authority(50, key=KEYS[1])}),),
        )
    assert store.next_profile_deadline(user_id="owner") == 50
    assert [job.profile_id for job in store.list_profile_deadline_jobs(now=75)] == [
        "ambiguous"
    ]


def test_selective_sqlite_plan_uses_persisted_index_and_bounds_metadata(indexed):
    _, store = indexed
    store.conn.executemany(
        "INSERT INTO profiles(profile_id,user_id,custom_features) VALUES (?,'owner',?)",
        [
            (f"noise-{i:05d}", json.dumps(authority(basis="system_estimate")))
            for i in range(3000)
        ],
    )
    store.conn.execute(
        "INSERT INTO profiles(profile_id,user_id,expiration_timestamp,custom_features) VALUES ('deadline','owner',999999,?)",
        (json.dumps(authority(100)),),
    )
    store.conn.execute("ANALYZE")
    expression = deadline_projection_sql()
    plan = store.conn.execute(
        f"EXPLAIN QUERY PLAN SELECT profile_id FROM profiles WHERE {expression} IS NOT NULL AND {expression} > (-9e999) ORDER BY {expression},profile_id LIMIT 1"
    ).fetchall()
    assert any(DEADLINE_INDEX in row[3] for row in plan)
    reads = []
    original = store._profile_deadline_rows

    def counted(**kwargs):
        result = original(**kwargs)
        reads.extend(result)
        return result

    store._profile_deadline_rows = counted
    assert store.next_profile_deadline(user_id="owner") == 100
    assert len(reads) == 1
    assert [job.profile_id for job in store.list_profile_deadline_jobs(now=100)] == [
        "deadline"
    ]


def test_wide_python_clocks_do_not_overflow_sqlite_candidate_binding(indexed):
    _, store = indexed
    store.conn.executemany(
        "INSERT INTO profiles(profile_id,user_id,custom_features) VALUES (?,'owner',?)",
        [
            ("regular", json.dumps(authority(100))),
            ("huge", json.dumps(authority(10**100))),
        ],
    )
    assert {
        job.profile_id for job in store.list_profile_deadline_jobs(now=10**101)
    } == {"regular", "huge"}
    assert store.list_profile_deadline_jobs(now=-(10**101)) == []
