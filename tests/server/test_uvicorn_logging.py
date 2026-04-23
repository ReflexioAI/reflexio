"""Tests for the uvicorn log-config dict handed to ``uvicorn.run``."""

from __future__ import annotations

import logging
import logging.config

from reflexio.server.uvicorn_logging import (
    ACCESS_FORMAT,
    LEVEL_FORMAT,
    UVICORN_LOG_CONFIG,
)


class TestUvicornLogConfig:
    def test_level_format_has_no_padding(self) -> None:
        """The format renders as ``INFO: msg`` / ``ERROR: msg`` — a single
        space after the colon, no right-padded level prefix."""
        assert LEVEL_FORMAT == "%(levelname)s: %(message)s"

    def test_access_format_includes_request_line(self) -> None:
        assert "%(request_line)s" in ACCESS_FORMAT
        assert ACCESS_FORMAT.startswith("%(levelname)s: ")

    def test_dict_is_valid_dictconfig(self) -> None:
        """``logging.config.dictConfig`` must accept the dict — catches
        typos / schema drift in the bundled config."""
        logging.config.dictConfig(UVICORN_LOG_CONFIG)

    def test_loggers_wire_uvicorn_names(self) -> None:
        names = set(UVICORN_LOG_CONFIG["loggers"])
        assert {"uvicorn", "uvicorn.error", "uvicorn.access"}.issubset(names)
