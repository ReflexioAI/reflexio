"""Cheap, high-precision guard for explicit personalization opt-outs."""

from __future__ import annotations

import re

_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "`": "'",
    }
)

_NEGATOR = r"(?:do\s+not|don't|dont|never)"
_PROFILE_TARGET = r"""
    (?:any\s+(?:of\s+)?)?
    (?:
        (?:my|the\s+user'?s|the\s+user)\s+
        (?:(?:saved|stored|previous|prior|past|historical)\s+)*
        (?:profiles?|preferences?|memor(?:y|ies)|(?:conversation\s+)?history|context|data|information)
      |
        (?:(?:saved|stored|previous|prior|past|historical|personal|user)\s+)+
        (?:profiles?|preferences?|memor(?:y|ies)|(?:conversation\s+)?history|context|data|information)
      |
        (?:profiles?|preferences?|memor(?:y|ies))
    )
"""
_PROFILE_ACTION = r"""
    (?:
        use|apply|consider|reference|retrieve|include|consult|access|rely\s+on|
        draw\s+(?:on|from)|take\s+into\s+account
    )
"""

_DIRECT_OPTOUT_PATTERNS = (
    re.compile(
        rf"""
        \b{_NEGATOR}\s+
        (?:
            personali[sz]e
          |
            (?:use|apply|include)\s+(?:any\s+)?personali[sz]ation
          |
            (?:use|apply|include)\s+(?:any\s+)?personali[sz]ed\s+
            (?:context|information|data|content)
        )\b
        """,
        re.VERBOSE,
    ),
    re.compile(
        rf"\b{_NEGATOR}\s+{_PROFILE_ACTION}\s+{_PROFILE_TARGET}\b",
        re.VERBOSE,
    ),
    re.compile(
        rf"\b(?:i\s+)?(?:do\s+not|don't|dont)\s+want\s+you\s+to\s+"
        rf"{_PROFILE_ACTION}\s+{_PROFILE_TARGET}\b",
        re.VERBOSE,
    ),
)

_WITHOUT_OPTOUT_PATTERNS = (
    re.compile(
        r"""
        \b(?:without|no)\s+(?:any\s+)?
        (?:
            personali[sz]ation
          |
            personali[sz]ed\s+(?:context|information|data)
        )\b
        """,
        re.VERBOSE,
    ),
    re.compile(
        rf"""
        \bwithout\s+
        (?:using|applying|considering|referencing|retrieving|including|consulting|accessing)
        \s+{_PROFILE_TARGET}\b
        """,
        re.VERBOSE,
    ),
)

_IMPERATIVE_OPTOUT_PATTERN = re.compile(
    rf"\b(?:ignore|disregard|skip|exclude|omit|avoid\s+using)\s+{_PROFILE_TARGET}\b",
    re.VERBOSE,
)
_NEGATED_PREFIX = re.compile(
    r"\b(?:"
    r"(?:do\s+not|don't|dont|never|not)\b"
    r"(?:[\s,]+[\w'-]+){0,4}[\s,]*|"
    r"with\s+(?:and|or)\s+|"
    r"(?:of|about|regarding)\s+"
    r")$"
)
_META_CLAUSE_PREFIX = re.compile(
    r"(?:^|[.!?;]\s*)(?:please\s+)?"
    r"(?:explain|describe|discuss|analy[sz]e|compare|quote|repeat)\b"
    r"[^.!?;]*$"
)
_REPORTED_NEGATION_PREFIX = re.compile(
    r"(?:^|[.!?;]\s*)[^.!?;]*\b"
    r"(?:never|did\s+not|didn't|didnt|do\s+not|don't|dont)\s+"
    r"(?:say|said|ask|asked|write|wrote)\s+['\"]?$"
)
_POSITIVE_USER_CONTEXT_PATTERN = re.compile(
    rf"\b{_PROFILE_ACTION}\s+{_PROFILE_TARGET}\b", re.VERBOSE
)


def _normalize(query: str) -> str:
    return " ".join(query.translate(_APOSTROPHE_TRANSLATION).casefold().split())


def _has_negated_prefix(query: str, start: int) -> bool:
    return bool(_NEGATED_PREFIX.search(query[max(0, start - 96) : start]))


def _has_meta_clause_prefix(query: str, start: int) -> bool:
    prefix = query[:start]
    return bool(
        _META_CLAUSE_PREFIX.search(prefix) or _REPORTED_NEGATION_PREFIX.search(prefix)
    )


def _has_positive_user_context_request(query: str) -> bool:
    for match in _POSITIVE_USER_CONTEXT_PATTERN.finditer(query):
        if not _has_negated_prefix(query, match.start()):
            return True
    return False


def _is_explicit_directive(query: str, start: int) -> bool:
    return not (
        _has_negated_prefix(query, start) or _has_meta_clause_prefix(query, start)
    )


def should_suppress_user_context(query: str | None) -> bool:
    """Return whether ``query`` explicitly opts out of personalized context.

    This intentionally recognizes only high-precision English directives. Ambiguous
    language fails open so ordinary profile and user-playbook retrieval is unchanged.
    """
    if not query or not query.strip():
        return False

    normalized = _normalize(query)
    if _has_positive_user_context_request(normalized):
        return False

    for pattern in _DIRECT_OPTOUT_PATTERNS:
        for match in pattern.finditer(normalized):
            if not _has_meta_clause_prefix(normalized, match.start()):
                return True

    for pattern in _WITHOUT_OPTOUT_PATTERNS:
        for match in pattern.finditer(normalized):
            if _is_explicit_directive(normalized, match.start()):
                return True

    for match in _IMPERATIVE_OPTOUT_PATTERN.finditer(normalized):
        if _is_explicit_directive(normalized, match.start()):
            return True
    return False
