"""SQLite construction preserves model choices when retiring an ignored sitevar."""

from pathlib import Path
from unittest.mock import patch

import pytest

from reflexio.models.config_schema import LLMConfig
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.site_var.site_var_manager import SiteVarManager


@pytest.mark.parametrize("legacy_model", [None, "openai/text-embedding-3-large"])
@pytest.mark.parametrize("override", [None, "openai/text-embedding-3-small"])
def test_embedding_selection_ignores_legacy_sitevar_and_honors_config(
    tmp_path: Path, legacy_model: str | None, override: str | None
) -> None:
    settings = {"default_generation_model_name": "openai/gpt-4o-mini"}
    if legacy_model is not None:
        settings["embedding_model_name"] = legacy_model
    with patch.object(SiteVarManager, "get_site_var", return_value=settings):
        storage = SQLiteStorage(
            org_id="model-selection",
            db_path=str(tmp_path / "storage.db"),
            llm_config=LLMConfig(embedding_model_name=override),
        )
    try:
        assert storage.embedding_model_name == (override or "local/minilm-l6-v2")
        assert storage._expansion_model_name == "openai/gpt-4o-mini"
    finally:
        storage.conn.close()
