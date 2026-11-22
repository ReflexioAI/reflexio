"""Served-model identity admission shared by HTTP and native CLI transports."""

from __future__ import annotations

import datetime
import re

from reflexio.server.env_utils import env_str

# ASCII digits only: `\d` would also match Arabic-Indic or fullwidth digits.
_PROVIDER_DATE_STAMP = re.compile(
    r"-(?:([0-9]{4})-([0-9]{2})-([0-9]{2})|([0-9]{4})([0-9]{2})([0-9]{2}))"
)


def _is_provider_date_stamp(suffix: str) -> bool:
    """Return whether ``suffix`` is one ``-YYYY-MM-DD`` / ``-YYYYMMDD`` real date.

    Args:
        suffix (str): Text following the configured model id.

    Returns:
        bool: True only for a well-formed stamp naming a valid calendar date.
    """
    match = _PROVIDER_DATE_STAMP.fullmatch(suffix)
    if match is None:
        return False
    year, month, day = (int(part) for part in match.groups() if part is not None)
    try:
        datetime.date(year, month, day)
    except ValueError:
        return False
    return True


def _ends_in_date_stamp(model: str) -> bool:
    """Return whether ``model`` is an id followed by a real date stamp."""
    return any(
        len(model) > width and _is_provider_date_stamp(model[-width:])
        for width in (len("-YYYY-MM-DD"), len("-YYYYMMDD"))
    )


def served_model_admitted(*, configured_bare: str, observed_bare: str) -> bool:
    """Decide whether a served model is the configured revision.

    Args:
        configured_bare (str): Configured model id without its route prefix.
        observed_bare (str): Served model id without its route prefix.

    Returns:
        bool: True for an exact match, or -- only when the configured id is not
            itself a dated snapshot -- for the configured id followed by
            exactly one ``-YYYY-MM-DD`` / ``-YYYYMMDD`` stamp of a real date.
            A pinned snapshot needs an exact echo.
    """
    if observed_bare == configured_bare:
        return True
    if _ends_in_date_stamp(configured_bare):
        return False
    # An unchanged string means the configured id is not a prefix (or is empty).
    suffix = observed_bare.removeprefix(configured_bare)
    return suffix != observed_bare and _is_provider_date_stamp(suffix)


# Claude Code's documented model aliases (code.claude.com/docs/en/model-config,
# "Model aliases"), each mapped to the model FAMILIES it may resolve to. An
# alias resolves to a concrete id the CLI then reports (`--model sonnet` ->
# `claude-sonnet-4-5-20250929`), so an alias pin can only be checked by
# family. `best` "uses the same model as `opus`"; `opusplan` "uses `opus`
# during plan mode, then switches to `sonnet`". No list exists in this
# codebase to derive it from, so it is spelled here.
_CLAUDE_CODE_ALIAS_FAMILIES: dict[str, tuple[str, ...]] = {
    "sonnet": ("sonnet",),
    "opus": ("opus",),
    "best": ("opus",),
    "haiku": ("haiku",),
    "fable": ("fable",),
    "opusplan": ("opus", "sonnet"),
}
# "append the `[1m]` suffix to a model alias or a full model name" to select
# the 1M-context variant; the reported id names the underlying model.
_CLAUDE_CODE_EXTENDED_CONTEXT_SUFFIX = "[1m]"
# "ANTHROPIC_DEFAULT_SONNET_MODEL" etc. remap what an alias of that family
# resolves to; the launch identity binds them (the `ANTHROPIC_` namespace).
_CLAUDE_CODE_FAMILY_MODEL_ENV = "ANTHROPIC_DEFAULT_{family}_MODEL"
# Claude ids as a cloud provider reports them: Bedrock prefixes
# `anthropic.` and optionally a cross-region inference profile (`us.`,
# `eu.`, `apac.`, `global.`); Vertex appends `@<version>` to the bare id.
_BEDROCK_CLAUDE_PREFIX = re.compile(r"^(?:[a-z]+\.)?anthropic\.", re.IGNORECASE)


def _claude_family_served(*, family: str, served: str) -> bool:
    """Whether ``served`` is a Claude id of ``family``, bare or cloud-qualified."""
    bare = _BEDROCK_CLAUDE_PREFIX.sub("", served.casefold(), count=1)
    return bare.startswith(f"claude-{family}-")


def _claude_code_alias_admitted(*, family: str, served: str) -> bool:
    """One alias family: the operator's remapped id if set, else the family.

    A configured ``ANTHROPIC_DEFAULT_<FAMILY>_MODEL`` is the exact id that
    alias launches, so only that id (with the same snapshot rule as any pin)
    is admitted; without it the alias resolves to the account's model of the
    family, possibly cloud-qualified.
    """
    configured = env_str(_CLAUDE_CODE_FAMILY_MODEL_ENV.format(family=family.upper()))
    if configured:
        suffix = _CLAUDE_CODE_EXTENDED_CONTEXT_SUFFIX
        if configured.casefold().endswith(suffix):
            configured = configured[: -len(suffix)]
        return served_model_admitted(configured_bare=configured, observed_bare=served)
    return _claude_family_served(family=family, served=served)


def native_served_model_admitted(*, host: str | None, pin: str, served: str) -> bool:
    """Decide whether a CLI-reported served model honours an explicit pin.

    For Claude Code only: a ``[1m]`` extended-context suffix is not part of the
    compared id; a documented alias (any case) names its families and admits, per
    family, the id ``ANTHROPIC_DEFAULT_<FAMILY>_MODEL`` configures, or else
    ``claude-<family>-...`` bare or Bedrock/Vertex-qualified. Every other pin,
    and every other host, uses ``_served_model_admitted``.

    Args:
        host (str | None): The native host of the configured reference.
        pin (str): The pinned model id.
        served (str): The model the CLI reported.

    Returns:
        bool: Whether ``served`` is the pinned model.
    """
    if host == "claude-code":
        suffix = _CLAUDE_CODE_EXTENDED_CONTEXT_SUFFIX
        if pin.casefold().endswith(suffix):
            pin = pin[: -len(suffix)]
        alias = pin.casefold()
        # No `default` exemption: an explicit `/default` is the literal model
        # ID `default` (docs/native-cli-model-selection.md), so it is held to
        # the explicit-pin rule. Only a host-only reference delegates, and it
        # never reaches here (it pins nothing).
        families = _CLAUDE_CODE_ALIAS_FAMILIES.get(alias)
        if families is not None:
            return any(
                _claude_code_alias_admitted(family=family, served=served)
                for family in families
            )
    return served_model_admitted(configured_bare=pin, observed_bare=served)
