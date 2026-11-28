"""Request guard errors available without loading provider configuration."""


class ProviderRequestGuardError(RuntimeError):
    """Raised when a caller rejects built params at the provider boundary."""
