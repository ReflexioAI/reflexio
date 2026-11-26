"""Explicit projection of the six-field historical extraction-window policy.

These functions consume canonical *historical* names after the ordinary raw
name converter. Storage owners must dispatch on their recorded generation and
retain the original documents. Nothing here writes, stamps a generation, or
authorizes a current reader to accept historical names.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from reflexio.config_env import ConfigurationMigrationError
from reflexio.models.extraction_window import ExtractionWindow

_KINDS: tuple[Literal["profile"], Literal["playbook"]] = ("profile", "playbook")
_EXTRACTORS = ("profile_extractor_config", "user_playbook_extractor_config")
_DIMENSIONS = ("window_size", "stride_size")
EXTRACTION_WINDOW_FIELDS = frozenset(
    f"{kind}_{dimension}" for kind in _KINDS for dimension in _DIMENSIONS
)
LEGACY_EXTRACTOR_WINDOW_FIELDS = ("window_size_override", "stride_size_override")


class _Overrides(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    window_size_override: int | None = Field(default=None, gt=0)
    stride_size_override: int | None = Field(default=None, gt=0)


class _HistoricalWindows(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    # Frozen pre-transition schema data, not current execution defaults. Future
    # default changes must not reinterpret an old missing/null field.
    window_size: int = Field(default=10, gt=0)
    stride_size: int = Field(default=8, gt=0)
    profile_extractor_config: _Overrides | None = None
    user_playbook_extractor_config: _Overrides | None = None

    @model_validator(mode="before")
    @classmethod
    def null_global_values_use_historical_defaults(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: item
                for key, item in value.items()
                if key not in _DIMENSIONS or item is not None
            }
        return value

    @model_validator(mode="after")
    def require_valid_historical_global_pair(self) -> Self:
        ExtractionWindow(self.window_size, self.stride_size).require_automatic()
        return self

    def resolve(self, kind: Literal["profile", "playbook"]) -> ExtractionWindow:
        overrides = (
            self.profile_extractor_config
            if kind == "profile"
            else self.user_playbook_extractor_config
        )
        width = overrides.window_size_override if overrides else None
        stride = overrides.stride_size_override if overrides else None
        return ExtractionWindow(
            width if width is not None else self.window_size,
            stride if stride is not None else self.stride_size,
        )


def project_legacy_extractor_window_metadata(frozen: dict[str, Any]) -> dict[str, Any]:
    """Remove only validated inert geometry from an explicit frozen snapshot.

    Its storage/window owner must authenticate the untouched witness and retain
    resolved geometry/selected inputs. This projection does not approve a parent
    binding, choose current policy, or rewrite the historical record.
    """
    try:
        _Overrides.model_validate(frozen)
    except ValidationError:
        raise ConfigurationMigrationError(
            "Invalid frozen extractor-window metadata."
        ) from None
    projected = deepcopy(frozen)
    for name in LEGACY_EXTRACTOR_WINDOW_FIELDS:
        projected.pop(name, None)
    return projected


def _require_canonical_historical_root_names(payload: dict[str, Any]) -> None:
    from reflexio.config_migration import (
        _CONFIG_FIELD_MIGRATION,
        _LEGACY_SINGLE_CONFIG_FIELDS,
    )

    # Refuse a caller that skipped raw name conversion; otherwise an old alias
    # or preset could silently become a missing field/default at this seam.
    if (
        EXTRACTION_WINDOW_FIELDS.intersection(payload)
        or set(_CONFIG_FIELD_MIGRATION).intersection(payload)
        or any(old in payload for old, _new in _LEGACY_SINGLE_CONFIG_FIELDS)
        or "extraction_preset" in payload
    ):
        raise ConfigurationMigrationError(
            "Extraction-window conversion requires canonical historical policy; "
            "current fields and historical aliases cannot select its generation."
        )


def _read_canonical_historical_windows(payload: dict[str, Any]) -> _HistoricalWindows:
    from reflexio.config_migration import _EXTRACTOR_OVERRIDE_MIGRATION

    _require_canonical_historical_root_names(payload)
    for name in _EXTRACTORS:
        extractor = payload.get(name)
        if isinstance(extractor, dict) and set(
            _EXTRACTOR_OVERRIDE_MIGRATION
        ).intersection(extractor):
            raise ConfigurationMigrationError(
                "Translate historical extractor-window names before conversion."
            )
    try:
        return _HistoricalWindows.model_validate(payload)
    except ValidationError:
        raise ConfigurationMigrationError(
            "Invalid historical extraction-window policy."
        ) from None


def historical_global_extraction_window(payload: dict[str, Any]) -> ExtractionWindow:
    """Read the captured global pair for an explicitly historical operation.

    For example, unversioned report transport used the global width, not an
    extractor's override. The caller authenticates/selects the original document;
    this port never selects current policy, writes, or grants upgrade authority.
    """
    historical = _read_canonical_historical_windows(payload)
    return ExtractionWindow(historical.window_size, historical.stride_size)


def convert_canonical_legacy_extraction_windows(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Replace six historical selectors with four explicit effective values.

    Disabled extractors retain their policy for a later deliberate enablement.
    Other fields and exact opaque metadata remain unchanged. The caller owns
    full-document validation and publication after this pure projection.
    """
    historical = _read_canonical_historical_windows(payload)
    converted = deepcopy(payload)
    for name in _DIMENSIONS:
        converted.pop(name, None)
    for kind, name in zip(_KINDS, _EXTRACTORS, strict=True):
        width, stride = historical.resolve(kind)
        converted[f"{kind}_window_size"] = width
        converted[f"{kind}_stride_size"] = stride
        extractor = converted.get(name)
        if isinstance(extractor, dict):
            for dimension in _DIMENSIONS:
                extractor.pop(f"{dimension}_override", None)
    return converted


def convert_canonical_legacy_project_windows(
    original_workspace: dict[str, Any], overlay: dict[str, Any]
) -> dict[str, Any]:
    """Preserve the effective policy of a sparse historical project overlay.

    When a workspace extractor overrode a global value, that project dimension
    was inherited for that extractor. Leave it inherited after conversion.
    The publication owner retains the exact historical overlay separately.

    Project raw-name/preset conversion must already have run. A project preset
    never selected geometry, unlike a workspace preset. This function cannot
    establish that the supplied workspace is an authenticated retained original.
    The native owner must bind that original to its current revision before CAS.
    """
    workspace = _read_canonical_historical_windows(original_workspace)
    _require_canonical_historical_root_names(overlay)
    dimensions = {key: overlay[key] for key in _DIMENSIONS if key in overlay}
    effective = _read_canonical_historical_windows(
        {**workspace.model_dump(mode="python"), **dimensions}
    )
    converted = deepcopy(overlay)
    for kind, name in zip(_KINDS, _EXTRACTORS, strict=True):
        overrides = getattr(workspace, name)
        width, stride = effective.resolve(kind)
        for dimension, value in zip(_DIMENSIONS, (width, stride), strict=True):
            if dimension not in dimensions:
                continue
            if (
                overrides is not None
                and getattr(overrides, f"{dimension}_override") is not None
            ):
                continue
            converted[f"{kind}_{dimension}"] = value
    for dimension in dimensions:
        converted.pop(dimension)
    return converted
