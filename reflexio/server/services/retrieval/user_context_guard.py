"""Cheap, high-precision guard for explicit personalization opt-outs."""

from __future__ import annotations

import re
import unicodedata

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
_ZH_OPTOUT_PATTERN = re.compile(
    r"(?:^|[。！？!?；;，,])\s*"
    r"(?:不要|別|别|請勿|请勿|無需|无需|不用|不需要)"
    r"(?:再)?(?:使用|採用|采用|參考|参考|考慮|考虑|讀取|读取|檢索|检索|訪問|访问|依賴|依赖)?"
    r"(?:任何)?(?:我的|使用者的|用户的|個人|个人)?"
    r"(?:個人化|个性化|個人資料|个人资料|偏好|記憶|记忆|歷史記錄|历史记录|上下文|個人資訊|个人信息|資料|数据)"
)


def _normalize(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query)
    return " ".join(normalized.translate(_APOSTROPHE_TRANSLATION).casefold().split())


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


def should_suppress_user_context(
    query: str | None,
    *,
    include_user_context: bool | None = None,
) -> bool:
    """Return whether ``query`` explicitly opts out of personalized context.

    An explicit API flag wins over text detection. Text detection intentionally
    recognizes only high-precision English and Chinese directives; ambiguous
    language fails open so ordinary retrieval is unchanged.
    """
    if include_user_context is not None:
        return not include_user_context
    if not query or not query.strip():
        return False

    normalized = _normalize(query)
    if _has_positive_user_context_request(normalized):
        return False
    if _ZH_OPTOUT_PATTERN.search(normalized):
        return True

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
