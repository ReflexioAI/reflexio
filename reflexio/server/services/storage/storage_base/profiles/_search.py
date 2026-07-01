from abc import abstractmethod

from reflexio.models.api_schema.domain import (
    Interaction,
    Status,
    UserProfile,
)
from reflexio.models.api_schema.retriever_schema import (
    SearchInteractionRequest,
    SearchUserProfileRequest,
)


class ProfileSearchMixin:
    """Mixin for interaction + profile search methods."""

    @abstractmethod
    def search_interaction(
        self,
        search_interaction_request: SearchInteractionRequest,
        query_embedding: list[float] | None = None,
    ) -> list[Interaction]:
        raise NotImplementedError

    @abstractmethod
    def search_user_profile(
        self,
        search_user_profile_request: SearchUserProfileRequest,
        status_filter: list[Status | None] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[UserProfile]:
        raise NotImplementedError
