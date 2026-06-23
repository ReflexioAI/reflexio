"""Compute net sessions per rule by joining PlaybookApplicationStat with success outcomes."""

from reflexio.server.services.evaluation_overview.rule_attribution import (
    RuleAttribution,
    compute_net_sessions,
)


def test_basic_join_one_rule_two_successes_one_failure() -> None:
    """Net = successes_with_rule_fired - failures_with_rule_fired."""
    citations_by_session = {
        ("u1", "sess_a"): [("playbook", "rule_42")],
        ("u1", "sess_b"): [("playbook", "rule_42")],
        ("u1", "sess_c"): [("playbook", "rule_42")],
    }
    is_success_by_session = {
        ("u1", "sess_a"): True,
        ("u1", "sess_b"): True,
        ("u1", "sess_c"): False,
    }
    rule_titles = {("playbook", "rule_42"): "Confirm address before checkout"}

    attribs = compute_net_sessions(
        citations_by_session=citations_by_session,
        is_success_by_session=is_success_by_session,
        rule_titles=rule_titles,
        top_n=5,
    )

    assert len(attribs) == 1
    a = attribs[0]
    assert a.rule_id == "rule_42"
    assert a.kind == "playbook"
    assert a.title == "Confirm address before checkout"
    assert a.successes_with == 2
    assert a.failures_with == 1
    assert a.net_sessions == 1


def test_ranks_by_net_sessions_descending_and_caps_at_top_n() -> None:
    """Top-N ordering by net_sessions desc; ties broken by total fires desc."""
    citations_by_session = {
        ("u1", "s1"): [("playbook", "good")],
        ("u1", "s2"): [("playbook", "good")],
        ("u1", "s3"): [("playbook", "good")],
        ("u1", "s4"): [("playbook", "ugly")],
        ("u1", "s5"): [("playbook", "ugly")],
        ("u1", "s6"): [("playbook", "meh")],
    }
    is_success_by_session = {
        ("u1", "s1"): True,
        ("u1", "s2"): True,
        ("u1", "s3"): True,
        ("u1", "s4"): False,
        ("u1", "s5"): False,
        ("u1", "s6"): True,
    }
    rule_titles = {
        ("playbook", "good"): "good",
        ("playbook", "ugly"): "ugly",
        ("playbook", "meh"): "meh",
    }

    top = compute_net_sessions(
        citations_by_session=citations_by_session,
        is_success_by_session=is_success_by_session,
        rule_titles=rule_titles,
        top_n=2,
    )
    assert len(top) == 2
    assert top[0].rule_id == "good"
    assert top[0].net_sessions == 3
    assert top[1].rule_id == "meh"
    assert top[1].net_sessions == 1


def test_session_missing_from_success_map_is_skipped() -> None:
    """If a citation references a session we have no AgentSuccessEvaluationResult
    for, treat it as unknown and don't count it on either side."""
    _ = RuleAttribution  # imported for export sanity; no-op
    citations_by_session = {
        ("u1", "sess_known"): [("playbook", "r1")],
        ("u1", "sess_orphan"): [("playbook", "r1")],
    }
    is_success_by_session = {("u1", "sess_known"): True}
    rule_titles = {("playbook", "r1"): "r1"}

    attribs = compute_net_sessions(
        citations_by_session=citations_by_session,
        is_success_by_session=is_success_by_session,
        rule_titles=rule_titles,
        top_n=5,
    )
    assert len(attribs) == 1
    a = attribs[0]
    assert a.successes_with == 1
    assert a.failures_with == 0
    assert a.net_sessions == 1


def test_cited_session_ids_populated_for_each_rule() -> None:
    """Each RuleAttribution row carries the session ids that cited the rule."""
    citations_by_session = {
        ("u1", "sess_alpha"): [("playbook", "rule_a"), ("playbook", "rule_b")],
        ("u1", "sess_beta"): [("playbook", "rule_a")],
        ("u1", "sess_gamma"): [("playbook", "rule_b")],
    }
    is_success_by_session = {
        ("u1", "sess_alpha"): True,
        ("u1", "sess_beta"): False,
        ("u1", "sess_gamma"): True,
    }
    rule_titles = {("playbook", "rule_a"): "A", ("playbook", "rule_b"): "B"}

    rows = compute_net_sessions(
        citations_by_session=citations_by_session,
        is_success_by_session=is_success_by_session,
        rule_titles=rule_titles,
        top_n=5,
    )
    rows_by_id = {r.rule_id: r for r in rows}

    # rule_a was cited in alpha (success) and beta (failure)
    assert set(rows_by_id["rule_a"].cited_session_ids) == {"sess_alpha", "sess_beta"}
    # rule_b was cited in alpha (success) and gamma (success)
    assert set(rows_by_id["rule_b"].cited_session_ids) == {"sess_alpha", "sess_gamma"}


def test_cited_session_ids_excludes_sessions_with_no_outcome() -> None:
    """Sessions absent from is_success_by_session are skipped on both sides."""
    citations_by_session = {
        ("u1", "graded"): [("playbook", "rule_x")],
        ("u1", "ungraded"): [("playbook", "rule_x")],
    }
    is_success_by_session = {("u1", "graded"): True}  # 'ungraded' not present

    rows = compute_net_sessions(
        citations_by_session=citations_by_session,
        is_success_by_session=is_success_by_session,
        rule_titles={},
        top_n=5,
    )
    assert len(rows) == 1
    assert rows[0].cited_session_ids == ("graded",)
    assert "ungraded" not in rows[0].cited_session_ids


def test_cited_session_ids_dedupes_within_same_session() -> None:
    """A rule cited multiple times in the same session yields a single session entry."""
    citations_by_session = {
        ("u1", "sess_a"): [
            ("playbook", "rule_x"),
            ("playbook", "rule_x"),
            ("playbook", "rule_x"),
        ],
    }
    is_success_by_session = {("u1", "sess_a"): True}

    rows = compute_net_sessions(
        citations_by_session=citations_by_session,
        is_success_by_session=is_success_by_session,
        rule_titles={},
        top_n=5,
    )
    assert rows[0].cited_session_ids == ("sess_a",)
    assert rows[0].successes_with == 1


def test_default_cited_session_ids_is_empty_tuple() -> None:
    """Backward-compat: the field defaults to () and the dataclass is hashable."""
    attrib = RuleAttribution(
        rule_id="r",
        kind="playbook",
        title="t",
        successes_with=0,
        failures_with=0,
    )
    assert attrib.cited_session_ids == ()
    # Frozen + tuple-typed default keeps it hashable
    assert hash(attrib) == hash(attrib)
