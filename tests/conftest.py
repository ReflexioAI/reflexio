"""Test configuration — delegates to shared reflexio.test_support module."""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # tests/
PROJECT_ROOT = _THIS_DIR.parent.parent  # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reflexio.test_support.llm_mock import cleanup_llm_mock, configure_llm_mock


def pytest_configure(config):
    configure_llm_mock(config)


def pytest_unconfigure(config):
    cleanup_llm_mock(config)
