"""Keep user-context opt-out detection internal to retrieval."""

from __future__ import annotations

import inspect

from reflexio.client.client import ReflexioClient
from reflexio.models.api_schema.retriever_schema import (
    SearchUserPlaybookRequest,
    SearchUserProfileRequest,
    UnifiedSearchRequest,
)


def test_search_requests_do_not_expose_user_context_override() -> None:
    for request_type in (
        SearchUserProfileRequest,
        SearchUserPlaybookRequest,
        UnifiedSearchRequest,
    ):
        assert "include_user_context" not in request_type.model_fields


def test_client_search_methods_do_not_expose_user_context_override() -> None:
    for method in (
        ReflexioClient.search_user_profiles,
        ReflexioClient.search_profiles,
        ReflexioClient.search_user_playbooks,
        ReflexioClient.search,
    ):
        assert "include_user_context" not in inspect.signature(method).parameters
