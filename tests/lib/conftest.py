"""Pytest configuration for reflexio_lib tests.

This conftest provides fixtures for unit tests with mocked LLM responses.
The LLM mock is applied globally in the parent conftest.py.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from reflexio.lib.reflexio_lib import Reflexio


@pytest.fixture(autouse=True, scope="function")
def ensure_mock_env():
    """Function-scoped fixture to ensure mock mode is enabled for each test.

    This provides extra safety by re-verifying the env var is set.
    """
    os.environ["MOCK_LLM_RESPONSE"] = "true"
    yield


@pytest.fixture
def reflexio_mock():
    """Reflexio instance with mocked storage, llm_client, and request_context.

    Bypasses __init__ entirely so no real storage/LLM setup is required.
    """
    with patch.object(Reflexio, "__init__", lambda _self: None):
        instance = Reflexio()
    instance.org_id = "test_org"
    instance.request_context = MagicMock()
    instance.request_context.org_id = "test_org"
    instance.request_context.is_storage_configured.return_value = True
    instance.request_context.storage = MagicMock()
    instance.llm_client = MagicMock()
    return instance


@pytest.fixture
def reflexio_no_storage():
    """Reflexio instance where storage is not configured."""
    with patch.object(Reflexio, "__init__", lambda _self: None):
        instance = Reflexio()
    instance.org_id = "test_org"
    instance.request_context = MagicMock()
    instance.request_context.org_id = "test_org"
    instance.request_context.is_storage_configured.return_value = False
    instance.request_context.storage = None
    instance.llm_client = MagicMock()
    return instance
