"""Tests for the in-memory cache functionality."""

import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

from reflexio.client import ReflexioClient
from reflexio.client.cache import InMemoryCache


class TestInMemoryCache:
    """Test cases for InMemoryCache class."""

    def test_basic_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = InMemoryCache()

        # Set a value
        cache.set("test_method", "test_value", param1="value1", param2="value2")

        # Get the value
        result = cache.get("test_method", param1="value1", param2="value2")

        assert result == "test_value"  # noqa: S101

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = InMemoryCache()

        # Get non-existent value
        result = cache.get("test_method", param1="value1")

        assert result is None  # noqa: S101

    def test_cache_with_different_params(self):
        """Test that different parameters create different cache entries."""
        cache = InMemoryCache()

        # Set two values with different parameters
        cache.set("test_method", "value1", param="a")
        cache.set("test_method", "value2", param="b")

        # Get both values
        result1 = cache.get("test_method", param="a")
        result2 = cache.get("test_method", param="b")

        assert result1 == "value1"  # noqa: S101
        assert result2 == "value2"  # noqa: S101

    def test_cache_expiration(self):
        """Test that cache entries expire after TTL."""
        # Use a short TTL for testing (1 second)
        cache = InMemoryCache(ttl_seconds=1)

        # Set a value
        cache.set("test_method", "test_value", param="value")

        # Immediately get the value - should be cached
        result = cache.get("test_method", param="value")
        assert result == "test_value"  # noqa: S101

        # Wait for expiration
        time.sleep(1.1)

        # Get the value again - should be expired
        result = cache.get("test_method", param="value")
        assert result is None  # noqa: S101

    def test_cache_with_datetime(self):
        """Test cache with datetime parameters."""
        cache = InMemoryCache()

        dt1 = datetime(2025, 1, 1, 12, 0, 0)
        dt2 = datetime(2025, 1, 2, 12, 0, 0)

        # Set values with datetime parameters
        cache.set("test_method", "value1", timestamp=dt1)
        cache.set("test_method", "value2", timestamp=dt2)

        # Get values
        result1 = cache.get("test_method", timestamp=dt1)
        result2 = cache.get("test_method", timestamp=dt2)

        assert result1 == "value1"  # noqa: S101
        assert result2 == "value2"  # noqa: S101

    def test_cache_with_none_value(self):
        """Test cache with None as parameter value."""
        cache = InMemoryCache()

        # Set value with None parameter
        cache.set("test_method", "value", param=None)

        # Get value
        result = cache.get("test_method", param=None)

        assert result == "value"  # noqa: S101

    def test_thread_safety(self):
        """Test that cache is thread-safe."""
        cache = InMemoryCache()
        results = []
        errors = []

        def set_and_get(thread_id):
            try:
                # Set value
                cache.set("test_method", f"value_{thread_id}", thread_id=thread_id)

                # Small delay to simulate concurrent access
                time.sleep(0.01)

                # Get value
                result = cache.get("test_method", thread_id=thread_id)
                results.append((thread_id, result))
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=set_and_get, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0  # noqa: S101

        # Verify all threads got their correct values
        assert len(results) == 10  # noqa: S101
        for thread_id, result in results:
            assert result == f"value_{thread_id}"  # noqa: S101


class TestReflexioClientCache:
    """Test cases for cache integration in ReflexioClient."""

    @patch("reflexio.client.client.requests.Session")
    def test_get_profiles_cache_hit(self, mock_session_class):
        """Test that get_profiles returns cached result on cache hit."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "user_profiles": [],
            "msg": None,
        }
        mock_session.request.return_value = mock_response

        # Create client
        client = ReflexioClient(api_key="test_key")

        # First call - should hit API
        request1 = {
            "user_id": "user1",
            "start_time": None,
            "end_time": None,
            "top_k": 30,
        }
        result1 = client.get_profiles(request1)

        # Second call with same parameters - should hit cache
        result2 = client.get_profiles(request1)

        # Verify API was only called once
        assert mock_session.request.call_count == 1  # noqa: S101

        # Verify results are the same
        assert result1.model_dump() == result2.model_dump()  # noqa: S101

    @patch("reflexio.client.client.requests.Session")
    def test_get_profiles_force_refresh(self, mock_session_class):
        """Test that force_refresh bypasses cache."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "user_profiles": [],
            "msg": None,
        }
        mock_session.request.return_value = mock_response

        # Create client
        client = ReflexioClient(api_key="test_key")

        # First call
        request = {
            "user_id": "user1",
            "start_time": None,
            "end_time": None,
            "top_k": 30,
        }
        client.get_profiles(request)

        # Second call with force_refresh - should hit API again
        _result2 = client.get_profiles(request, force_refresh=True)

        # Verify API was called twice
        assert mock_session.request.call_count == 2  # noqa: S101

    @patch("reflexio.client.client.requests.Session")
    def test_get_profiles_different_params(self, mock_session_class):
        """Test that different parameters result in cache miss."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "user_profiles": [],
            "msg": None,
        }
        mock_session.request.return_value = mock_response

        # Create client
        client = ReflexioClient(api_key="test_key")

        # First call
        request1 = {
            "user_id": "user1",
            "start_time": None,
            "end_time": None,
            "top_k": 30,
        }
        client.get_profiles(request1)

        # Second call with different parameters
        request2 = {
            "user_id": "user2",
            "start_time": None,
            "end_time": None,
            "top_k": 30,
        }
        client.get_profiles(request2)

        # Verify API was called twice (cache miss)
        assert mock_session.request.call_count == 2  # noqa: S101

    @patch("reflexio.client.client.requests.Session")
    def test_get_agent_playbooks_cache_hit(self, mock_session_class):
        """Test that get_agent_playbooks returns cached result on cache hit."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "agent_playbooks": [],
            "msg": None,
        }
        mock_session.request.return_value = mock_response

        # Create client
        client = ReflexioClient(api_key="test_key")

        # First call - should hit API
        request1 = {"limit": 100, "playbook_name": "test"}
        result1 = client.get_agent_playbooks(request1)

        # Second call with same parameters - should hit cache
        result2 = client.get_agent_playbooks(request1)

        # Verify API was only called once
        assert mock_session.request.call_count == 1  # noqa: S101

        # Verify results are the same
        assert result1.model_dump() == result2.model_dump()  # noqa: S101

    @patch("reflexio.client.client.requests.Session")
    def test_get_agent_playbooks_force_refresh(self, mock_session_class):
        """Test that force_refresh bypasses cache for get_agent_playbooks."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "agent_playbooks": [],
            "msg": None,
        }
        mock_session.request.return_value = mock_response

        # Create client
        client = ReflexioClient(api_key="test_key")

        # First call
        request = {"limit": 100, "playbook_name": "test"}
        client.get_agent_playbooks(request)

        # Second call with force_refresh - should hit API again
        _result2 = client.get_agent_playbooks(request, force_refresh=True)

        # Verify API was called twice
        assert mock_session.request.call_count == 2  # noqa: S101

    @patch("reflexio.client.client.requests.Session")
    def test_get_agent_playbooks_with_none_request(self, mock_session_class):
        """Test that get_agent_playbooks with None request can be cached."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "agent_playbooks": [],
            "msg": None,
        }
        mock_session.request.return_value = mock_response

        # Create client
        client = ReflexioClient(api_key="test_key")

        # First call with None request
        client.get_agent_playbooks(None)

        # Second call with None request - should hit cache
        client.get_agent_playbooks(None)

        # Verify API was only called once
        assert mock_session.request.call_count == 1  # noqa: S101

    @patch("reflexio.client.client.requests.Session")
    def test_cache_expiration_integration(self, mock_session_class):
        """Test that cache expires after TTL in client."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "user_profiles": [],
            "msg": None,
        }
        mock_session.request.return_value = mock_response

        # Create client with short TTL
        client = ReflexioClient(api_key="test_key")
        client._cache = InMemoryCache(ttl_seconds=1)

        # First call
        request = {
            "user_id": "user1",
            "start_time": None,
            "end_time": None,
            "top_k": 30,
        }
        client.get_profiles(request)

        # Verify cache hit
        client.get_profiles(request)
        assert mock_session.request.call_count == 1  # noqa: S101

        # Wait for expiration
        time.sleep(1.1)

        # Third call after expiration - should hit API again
        client.get_profiles(request)
        assert mock_session.request.call_count == 2  # noqa: S101
