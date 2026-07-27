"""The bundled openclaw plugin's wire allowlist must track InteractionData.

The plugin builds its publish payload from a literal frozenset of field names
rather than importing the model, so the module keeps no runtime dependency on
``reflexio``. That literal can drift.

This guard lives in ``tests/`` deliberately. The plugin has its own suite under
``reflexio/integrations/openclaw/plugin/tests/``, but ``testpaths = ["tests"]``
in pyproject.toml means the root pytest run never collects it and no CI workflow
invokes it — so an equivalent assertion there would only ever run when someone
remembered to call it by hand with ``PYTHONPATH`` set. A drift guard that does
not execute is not a guard.

The plugin's own suite keeps its round-trip test (it needs the real slicer);
what must be pinned *here* is the field list, because that is what breaks
silently when a field is added to or renamed on the model.
"""

import sys
from pathlib import Path

from reflexio.models.api_schema.domain.entities import InteractionData

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]  # tests/models/api_schema/ -> repo root
_PLUGIN_SRC = _REPO_ROOT / "reflexio" / "integrations" / "openclaw" / "plugin" / "src"


def _plugin_allowlist() -> set[str]:
    """Import the bundled plugin's wire allowlist.

    The path is inside this repo, so this resolves deterministically — no
    ``importorskip``, because a skip here would silently disable the guard.
    """
    if str(_PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(_PLUGIN_SRC))
    from openclaw_smart import state

    return set(state._INTERACTION_DATA_FIELDS)


def test_plugin_src_is_where_this_test_expects() -> None:
    """Fail loudly if the plugin moves, rather than skipping the real check."""
    assert (_PLUGIN_SRC / "openclaw_smart" / "state.py").is_file(), (
        f"plugin source not found at {_PLUGIN_SRC}; update this guard"
    )


def test_plugin_allowlist_matches_interaction_data_fields() -> None:
    """A field added to or renamed on the model must be mirrored in the plugin.

    Without this, a new field is silently stripped from every plugin publish —
    the same class of silent data loss the surrounding validation exists to
    surface.
    """
    model_fields = set(InteractionData.model_fields)
    allowlist = _plugin_allowlist()
    assert allowlist == model_fields, (
        f"plugin allowlist drifted from InteractionData:"
        f" missing={sorted(model_fields - allowlist)}"
        f" extra={sorted(allowlist - model_fields)}"
    )
