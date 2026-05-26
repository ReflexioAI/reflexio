import contextlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from reflexio.cli.paths import reflexio_home
from reflexio.models.config_schema import (
    Config,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigDisk,
    StorageConfigSQLite,
)
from reflexio.server.services.configurator.config_storage import ConfigStorage


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

    def _default_storage_config(self) -> StorageConfigSQLite | StorageConfigDisk:
        """Select default storage config based on REFLEXIO_STORAGE env var."""
        backend = os.environ.get("REFLEXIO_STORAGE", "sqlite").lower()
        if backend == "disk":
            return StorageConfigDisk(dir_path=self.base_dir)
        return StorageConfigSQLite()

    def get_default_config(self) -> Config:
        """
        Returns a default configuration with storage based on REFLEXIO_STORAGE env var.

        Returns:
            Config: Default configuration with appropriate storage type
        """
        return Config(
            storage_config=self._default_storage_config(),
            profile_extractor_config=ProfileExtractorConfig(
                extractor_name="default_profile_extractor",
                extraction_definition_prompt="Extract key user information including name, role, preferences, and any other relevant profile details from the conversation.",
            ),
            user_playbook_extractor_config=PlaybookConfig(
                extractor_name="default_playbook_extractor",
                extraction_definition_prompt="Extract playbook rules about agent performance, including areas where the agent was helpful, areas for improvement, and any issues encountered during the interaction.",
            ),
        )

    def load_config(self) -> Config:
        """
        Loads the current configuration from local JSON file. If the file doesn't exist,
        creates a default configuration and saves it.

        Returns:
            Config: Loaded configuration object
        """
        if not Path(self.config_file).exists():
            config = self.get_default_config()
            self._save_config_to_local_dir(config=config)
            return config

        try:
            with Path(self.config_file).open(encoding="utf-8") as f:
                config_content = f.read()
                config: Config = Config(**json.loads(str(config_content)))
                return config
        except Exception as e:
            print(f"{str(e)}")
            tbs = traceback.format_exc().split("\n")
            for tb in tbs:
                print(f"  {tb}")
            # Create a default config if anything goes wrong.
            return self.get_default_config()

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
        """
        Saves configuration to the local JSON file atomically.

        Writes to a per-write unique ``.tmp`` file first, then renames it
        over the final path. ``Path.replace`` is atomic on POSIX (same
        filesystem), so concurrent ``set_config`` calls across multiple
        workers cannot observe a partially-written file, and a crash
        mid-write leaves the previous good file intact. The tmp filename
        embeds the pid and a nanosecond timestamp so concurrent writers
        do not race on the same temp path.

        Args:
            config (Config): Configuration object to save

        Raises:
            OSError: If the write or atomic rename fails. The tmp file is
                cleaned up on failure (best-effort), but the original
                exception propagates so callers see the failure rather
                than a false success.
        """
        if not (self.base_dir and self.config_file):
            raise ValueError("base_dir and config_file must be set")

        final_path = Path(self.config_file)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(
            f"{final_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            tmp_path.write_text(config.model_dump_json(), encoding="utf-8")
            tmp_path.replace(final_path)
        except OSError:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise
