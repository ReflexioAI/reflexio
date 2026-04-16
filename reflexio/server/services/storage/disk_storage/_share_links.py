"""Disk storage does not support share links (not needed for the disk use case)."""

from reflexio.models.api_schema.domain import ShareLink


class DiskShareLinkMixin:
    """Disk storage does not implement share links."""

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

        Raises:
            NotImplementedError: Share links are not supported with disk storage.
        """
        raise NotImplementedError("Share links are not supported with disk storage")

    def get_share_link_by_token(self, token: str) -> ShareLink | None:
        """Look up a share link by its token.

        Args:
            token (str): The share token.

        Returns:
            ShareLink | None: The share link if found, else None.

        Raises:
            NotImplementedError: Share links are not supported with disk storage.
        """
        raise NotImplementedError("Share links are not supported with disk storage")

    def get_share_link_by_resource(
        self, resource_type: str, resource_id: str
    ) -> ShareLink | None:
        """Look up an existing share link for a specific resource (for dedup).

        Args:
            resource_type (str): Type of resource.
            resource_id (str): ID of the resource.

        Returns:
            ShareLink | None: The existing share link if any, else None.

        Raises:
            NotImplementedError: Share links are not supported with disk storage.
        """
        raise NotImplementedError("Share links are not supported with disk storage")

    def get_share_links(self) -> list[ShareLink]:
        """Return all share links for this org.

        Returns:
            list[ShareLink]: All share links, ordered by created_at ascending.

        Raises:
            NotImplementedError: Share links are not supported with disk storage.
        """
        raise NotImplementedError("Share links are not supported with disk storage")

    def delete_share_link(self, link_id: int) -> bool:
        """Delete a share link by ID.

        Args:
            link_id (int): The share link ID.

        Returns:
            bool: True if deleted, False if not found.

        Raises:
            NotImplementedError: Share links are not supported with disk storage.
        """
        raise NotImplementedError("Share links are not supported with disk storage")

    def delete_all_share_links(self) -> int:
        """Delete all share links for this org.

        Returns:
            int: Number of links deleted.

        Raises:
            NotImplementedError: Share links are not supported with disk storage.
        """
        raise NotImplementedError("Share links are not supported with disk storage")
