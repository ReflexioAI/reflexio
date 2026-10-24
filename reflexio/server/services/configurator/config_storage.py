from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Final

from reflexio.models.config_schema import Config


class _ConfigVersionProbeFailure(Enum):
    FAILED = "failed"


CONFIG_VERSION_PROBE_FAILED: Final = _ConfigVersionProbeFailure.FAILED
ConfigVersionProbe = tuple[str, Any] | None | _ConfigVersionProbeFailure


class ConfigWriteConflictError(RuntimeError):
    """Raised when a conditional config write loses a freshness race."""


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
        Load the organization's saved policy. An adapter may initialize defaults
        when no configuration exists. Invalid or unreadable saved configuration
        must raise rather than silently select defaults. Existing-file reads do
        not publish replacements; explicit versioned migrations belong to their
        owning adapter's documented lifecycle.

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

    def get_version(self) -> tuple[str, Any] | None:
        """Cheap probe of the persisted config version, used by the per-org cache.

        Backends that can cheaply detect out-of-band mutations should
        override this. The Reflexio cache stamps the returned tuple at
        load time and re-probes on every cache hit; if the value
        changes, the cached instance is evicted and rebuilt with fresh
        configuration.

        Returns:
            tuple[str, Any] | None: A ``(kind, value)`` tuple where
            ``kind`` is a short string identifying the probe type
            (e.g. ``"file"`` for mtime, ``"db"`` for a row version).
            Returns ``None`` only when probing is unsupported. Probe failures
            raise so callers can retry rather than permanently disable probes.
        """
        return None
