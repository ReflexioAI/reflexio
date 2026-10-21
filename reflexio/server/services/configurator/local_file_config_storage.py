import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from reflexio.cli.paths import reflexio_home
from reflexio.config_document import decode_config_document
from reflexio.config_migration import StoredConfigurationError
from reflexio.models.config_schema import (
    Config,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
    validate_stored_config,
)
from reflexio.server.services.configurator.config_storage import ConfigStorage

logger = logging.getLogger(__name__)


class LocalFileConfigStorage(ConfigStorage):
    """
    Local JSON file-based configuration storage implementation.
    Saves/loads configuration to/from local JSON files.
    """

    def __init__(self, org_id: str, base_dir: str | None = None):
        super().__init__(org_id=org_id)
        if base_dir:
            # Ensure base_dir is absolute
            base_path = Path(base_dir)
            abs_base_dir = (
                str(base_path.resolve()) if not base_path.is_absolute() else base_dir
            )
            self.base_dir = str(Path(abs_base_dir) / "configs")
            self.config_file = str(Path(self.base_dir) / f"config_{org_id}.json")
            print(
                f"LocalFileConfigStorage will save config for {org_id} to a local file at {self.config_file}"
            )
        else:
            self.base_dir = str(reflexio_home() / "configs")
            self.config_file = str(Path(self.base_dir) / f"config_{org_id}.json")

    def _default_storage_config(self) -> StorageConfigSQLite:
        """Return the default storage config (always SQLite for local file storage)."""
        return StorageConfigSQLite()

    def get_default_config(self) -> Config:
        """
        Returns a default configuration with SQLite storage.

        Returns:
            Config: Default configuration with SQLite storage
        """
        return Config(
            storage_config=self._default_storage_config(),
            profile_extractor_config=ProfileExtractorConfig(
                extraction_definition_prompt="Extract key user information including name, role, preferences, and any other relevant profile details from the conversation.",
            ),
            user_playbook_extractor_config=PlaybookConfig(
                extraction_definition_prompt="Extract playbook rules about agent performance, including areas where the agent was helpful, areas for improvement, and any issues encountered during the interaction.",
            ),
        )

    def load_config(self) -> Config:
        """Read existing policy without rewriting it; initialize only absent files.

        Invalid saved policy and read failures never select another database or
        defaults. Historical conversion belongs to explicit upgrade operations.
        """
        path = Path(self.config_file)
        try:
            try:
                content = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                config = self.get_default_config()
                if self._publish_config_document(
                    config.model_dump_json(), create_only=True
                ):
                    return config
                # Another creator won. Read its complete document once; if it
                # disappeared, propagate that failure instead of recreating it.
                content = path.read_text(encoding="utf-8")
            data = decode_config_document(content)
        except ValueError:
            raise StoredConfigurationError(
                "Saved configuration is not a valid finite JSON object. "
                "Correct or restore the original file before continuing."
            ) from None
        return validate_stored_config(data)

    def save_config(self, config: Config) -> None:
        """
        Saves the configuration to the local JSON file.

        Args:
            config (Config): Configuration object to save
        """
        if self.base_dir and self.config_file:
            self._save_config_to_local_dir(config=config)
        else:
            print(
                f"Cannot save config for org {self.org_id}: no local directory configured"
            )

    def get_version(self) -> tuple[str, Any] | None:
        """Return the on-disk mtime of the org's config file, if it exists.

        The Reflexio cache uses this to detect out-of-band edits to
        ``~/.reflexio/configs/config_<org-id>.json`` (e.g. an operator
        editing the file directly while the server is running). If the
        file is missing — for example, before the first ``set_config``
        write — return None so the cache leaves the entry alone rather
        than thrashing every request.

        Returns:
            tuple[str, float] | None: ``("file", mtime_seconds)`` when
            the config file exists, ``None`` otherwise (missing file or
            stat failure).
        """
        try:
            return ("file", Path(self.config_file).stat().st_mtime)
        except OSError:
            return None

    def _save_config_to_local_dir(self, config: Config) -> None:
        """Publish an explicitly supplied full configuration by atomic replace."""
        self._publish_config_document(config.model_dump_json())

    def _publish_config_document(
        self, document: str, *, create_only: bool = False
    ) -> bool:
        """Atomically publish a complete file; initialization never replaces one.

        The raw-document seam also serves subclasses that own schema overlays.
        Unsupported exclusive publication and other I/O failures propagate.
        Returns False only when another initializer already created the path.
        """
        if not (self.base_dir and self.config_file):
            raise ValueError("base_dir and config_file must be set")
        final_path = Path(self.config_file)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(
            f"{final_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            tmp_path.write_text(document, encoding="utf-8")
            if create_only:
                try:
                    os.link(tmp_path, final_path)
                except FileExistsError:
                    return False
            else:
                tmp_path.replace(final_path)
            return True
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
