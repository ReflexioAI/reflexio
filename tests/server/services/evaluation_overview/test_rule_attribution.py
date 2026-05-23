"""Compute net sessions per rule by joining PlaybookApplicationStat with success outcomes."""

from reflexio.server.services.evaluation_overview.rule_attribution import (
    RuleAttribution,
    compute_net_sessions,
)


def test_basic_join_one_rule_two_successes_one_failure() -> None:
    """Net = successes_with_rule_fired - failures_with_rule_fired."""
    citations_by_session = {
        "sess_a": [("playbook", "rule_42")],
        "sess_b": [("playbook", "rule_42")],
        "sess_c": [("playbook", "rule_42")],
    }
    is_success_by_session = {"sess_a": True, "sess_b": True, "sess_c": False}
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
        "s1": [("playbook", "good")],
        "s2": [("playbook", "good")],
        "s3": [("playbook", "good")],
        "s4": [("playbook", "ugly")],
        "s5": [("playbook", "ugly")],
        "s6": [("playbook", "meh")],
    }
    is_success_by_session = {
        "s1": True,
        "s2": True,
        "s3": True,
        "s4": False,
        "s5": False,
        "s6": True,
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
        "sess_known": [("playbook", "r1")],
        "sess_orphan": [("playbook", "r1")],
    }
    is_success_by_session = {"sess_known": True}
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
