"""Environment parsing shared by import-light launchers and the server."""


def env_truthy(value: str) -> bool:
    """Parse the historical permissive boolean format.

    New boolean settings should use ``server.env_utils.env_bool`` for strict
    validation. This parser retains compatibility for existing callers and
    migration checks without importing the server's startup side effects.
    """
    return value.strip().lower() in {"1", "true", "yes", "on"}
