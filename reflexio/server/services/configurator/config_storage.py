from abc import ABC, abstractmethod

from reflexio.models.config_schema import Config


class ConfigStorage(ABC):
    """
    Abstract base class for configuration storage operations.
    Defines the interface for saving/loading configuration to/from persistent storage.
    """

    def __init__(self, org_id: str):
        self.org_id: str = org_id

    @abstractmethod
    def get_default_config(self) -> Config:
        """
        Returns a default configuration that is uninitialized.

        Returns:
            Config: Default configuration object
        """

    @abstractmethod
    def load_config(self) -> Config:
        """
        Loads the current configuration of the organization. If the organization does
        not exist, or if no config exists, or if the current saved config is no longer valid,
        this routine creates a default one but will not update the persistent storage.

        Returns:
            Config: Loaded configuration object
        """

    @abstractmethod
    def save_config(self, config: Config) -> None:
        """
        Saves the configuration to the persistent storage.

        Args:
            config (Config): Configuration object to save
        """
