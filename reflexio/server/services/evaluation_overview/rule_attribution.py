"""Join PlaybookApplicationStat citations with session success outcomes.

Replaces the spec's proposed `rule_application` table with a join against
the existing `Interaction.citations` data, computing "net sessions" per
rule = successes_with_rule_fired - failures_with_rule_fired.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

CitationKey = tuple[str, str]  # (kind, real_id) — matches PlaybookApplicationStat


@dataclass(frozen=True)
class RuleAttribution:
    """One row of the "rules that moved the needle" panel."""

    rule_id: str
    kind: str
    title: str
    successes_with: int
    failures_with: int
    cited_session_ids: tuple[str, ...] = ()
    """Session IDs (within the trend window) that cited this rule. Tuple so
    the dataclass stays hashable / frozen. The frontend uses this to filter
    the /evaluations detail band to show only sessions where the rule
    actually fired — answering 'which sessions did this rule help/hurt?'
    without a second roundtrip."""

    @property
    def net_sessions(self) -> int:
        return self.successes_with - self.failures_with


def compute_net_sessions(
    *,
    citations_by_session: Mapping[str, list[CitationKey]],
    is_success_by_session: Mapping[str, bool],
    rule_titles: Mapping[CitationKey, str],
    top_n: int,
) -> list[RuleAttribution]:
    """Aggregate net sessions per rule and return the top N by net_sessions.

    Args:
        citations_by_session (Mapping[str, list[CitationKey]]): For each
            session in the window, the list of `(kind, real_id)` citations
            harvested from its interactions.
        is_success_by_session (Mapping[str, bool]): The session's evaluated
            outcome. Sessions not in this map are skipped on both sides.
        rule_titles (Mapping[CitationKey, str]): Pre-loaded display titles
            for each rule (empty string when the underlying row is gone).
        top_n (int): How many rows to return at most. Ordering: net_sessions
            desc, then total fires desc.

    Returns:
        list[RuleAttribution]: At most `top_n` rows in the order described.
        Each row carries the session ids that cited the rule so the
        frontend can drill from a rule into the sessions it fired in.
    """
    successes: dict[CitationKey, int] = {}
    failures: dict[CitationKey, int] = {}
    # Map (kind, real_id) -> ordered list of session_ids that cited this rule.
    # Ordering preserves caller-provided iteration order; the frontend sorts
    # the detail band by created_at when displaying.
    sessions_by_rule: dict[CitationKey, list[str]] = {}
    for session_id, citations in citations_by_session.items():
        outcome = is_success_by_session.get(session_id)
        if outcome is None:
            continue
        # A rule cited multiple times in the same session counts once.
        unique_cites = set(citations)
        for key in unique_cites:
            if outcome:
                successes[key] = successes.get(key, 0) + 1
            else:
                failures[key] = failures.get(key, 0) + 1
            sessions_by_rule.setdefault(key, []).append(session_id)

    all_keys = set(successes) | set(failures)
    rows = [
        RuleAttribution(
            rule_id=real_id,
            kind=kind,
            title=rule_titles.get((kind, real_id), ""),
            successes_with=successes.get((kind, real_id), 0),
            failures_with=failures.get((kind, real_id), 0),
            cited_session_ids=tuple(sessions_by_rule.get((kind, real_id), [])),
        )
        for (kind, real_id) in all_keys
    ]

    rows.sort(
        key=lambda r: (-r.net_sessions, -(r.successes_with + r.failures_with)),
    )
    return rows[:top_n]
