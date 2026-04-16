"""Abstract ShareLink storage operations.

Each BaseStorage subclass (SQLite, Supabase, Disk) must implement these methods.
Storage instances are org-scoped, so org_id is not a method parameter.
"""

from abc import abstractmethod

from reflexio.models.api_schema.domain import ShareLink


class ShareLinkMixin:
    """Mixin defining share link CRUD operations on a per-org data storage."""

    @abstractmethod
    def create_share_link(
        self,
        token: str,
        resource_type: str,
        resource_id: str,
        expires_at: int | None,
        created_by_email: str | None,
    ) -> ShareLink:
        """Create a new share link.

        Args:
            token (str): The share token (unique).
            resource_type (str): Type of resource (e.g., "profile", "user_playbook").
            resource_id (str): ID of the resource being shared.
            expires_at (int | None): Optional Unix timestamp of expiration.
            created_by_email (str | None): Optional email of creator.

        Returns:
            ShareLink: The created share link with id and created_at populated.
        """
        raise NotImplementedError

    @abstractmethod
    def get_share_link_by_token(self, token: str) -> ShareLink | None:
        """Look up a share link by its token.

        Args:
            token (str): The share token.

        Returns:
            ShareLink | None: The share link if found, else None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_share_link_by_resource(
        self, resource_type: str, resource_id: str
    ) -> ShareLink | None:
        """Look up an existing share link for a specific resource (for dedup).

        Args:
            resource_type (str): Type of resource.
            resource_id (str): ID of the resource.

        Returns:
            ShareLink | None: The existing share link if any, else None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_share_links(self) -> list[ShareLink]:
        """Return all share links for this org.

        Returns:
            list[ShareLink]: All share links, ordered by created_at ascending.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_share_link(self, link_id: int) -> bool:
        """Delete a share link by ID.

        Args:
            link_id (int): The share link ID.

        Returns:
            bool: True if deleted, False if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all_share_links(self) -> int:
        """Delete all share links for this org.

        Returns:
            int: Number of links deleted.
        """
        raise NotImplementedError
