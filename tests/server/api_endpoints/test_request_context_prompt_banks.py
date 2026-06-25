from pathlib import Path
from typing import Any, cast

from reflexio.models.config_schema import StorageConfigSQLite
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.configurator.base_configurator import BaseConfigurator


def _write_prompt(
    prompt_bank_path: Path,
    prompt_id: str,
    content: str,
    variables: list[str],
) -> None:
    prompt_dir = prompt_bank_path / prompt_id
    prompt_dir.mkdir(parents=True)
    lines = [
        "---",
        "active: true",
        "variables:",
        *(f"  - {variable}" for variable in variables),
        "---",
        "",
    ]
    (prompt_dir / "v1.0.0.prompt.md").write_text(
        "\n".join(lines) + content,
        encoding="utf-8",
    )


class FakeConfigurator:
    def __init__(self, prompt_bank_path: Path) -> None:
        self.prompt_bank_path = prompt_bank_path

    def get_prompt_bank_paths(self) -> list[Path]:
        return [self.prompt_bank_path]

    def get_current_storage_configuration(self) -> StorageConfigSQLite:
        return StorageConfigSQLite(db_path=":memory:")

    def create_storage(self, storage_config: StorageConfigSQLite) -> Any:
        return object()


def test_request_context_loads_configurator_prompt_banks(tmp_path):
    prompt_bank_path = tmp_path / "enterprise_prompt_bank"
    _write_prompt(
        prompt_bank_path,
        "enterprise_prompt",
        "Enterprise {name}",
        ["name"],
    )

    context = RequestContext(
        org_id="org-1",
        configurator=cast(BaseConfigurator, FakeConfigurator(prompt_bank_path)),
    )

    assert (
        context.prompt_manager.render_prompt("enterprise_prompt", {"name": "Ada"})
        == "Enterprise Ada"
    )
