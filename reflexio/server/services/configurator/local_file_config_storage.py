import json
import os
import traceback
from pathlib import Path

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
            self.base_dir = str(Path.home() / ".reflexio" / "configs")
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
            profile_extractor_configs=[
                ProfileExtractorConfig(
                    extractor_name="default_profile_extractor",
                    extraction_definition_prompt="Extract key user information including name, role, preferences, and any other relevant profile details from the conversation.",
                ),
            ],
            user_playbook_extractor_configs=[
                PlaybookConfig(
                    extractor_name="default_playbook_extractor",
                    extraction_definition_prompt="Extract playbook rules about agent performance, including areas where the agent was helpful, areas for improvement, and any issues encountered during the interaction.",
                ),
            ],
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

    def _save_config_to_local_dir(self, config: Config) -> None:
        """
        Saves configuration to the local JSON file.

        Args:
            config (Config): Configuration object to save
        """
        if not (self.base_dir and self.config_file):
            raise ValueError("base_dir and config_file must be set")

        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        try:
            with Path(self.config_file).open("w", encoding="utf-8") as f:
                f.write(config.model_dump_json())
        except Exception as e:
            print(f"{str(e)}")
            tbs = traceback.format_exc().split("\n")
            for tb in tbs:
                print(f"  {tb}")
