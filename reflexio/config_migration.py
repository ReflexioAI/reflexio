"""Errors that stored-configuration loaders must never turn into defaults."""


class ConfigurationMigrationError(ValueError):
    """An explicit migration is required before this saved policy can load."""
