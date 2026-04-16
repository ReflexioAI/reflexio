"""SQLite implementation of ShareLinkMixin."""

import time
from typing import Any

from reflexio.models.api_schema.domain import ShareLink

from ._base import SQLiteStorageBase


def _row_to_share_link(row: Any) -> ShareLink:
    """Convert a sqlite3.Row to a ShareLink model.

    Args:
        row: A sqlite3.Row from the share_links table.

    Returns:
        ShareLink: The populated model.
    """
    d = dict(row)
    return ShareLink(
        id=d["id"],
        org_id=d["org_id"],
        token=d["token"],
        resource_type=d["resource_type"],
        resource_id=d["resource_id"],
        created_at=d["created_at"],
        expires_at=d["expires_at"],
        created_by_email=d["created_by_email"],
    )


class SQLiteShareLinkMixin:
    """SQLite-backed share link operations."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    org_id: str
    _execute: Any
    _fetchone: Any
    _fetchall: Any

    @SQLiteStorageBase.handle_exceptions
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
        now = int(time.time())
        cur = self._execute(
            """INSERT INTO share_links
               (org_id, token, resource_type, resource_id, created_at, expires_at, created_by_email)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                self.org_id,
                token,
                resource_type,
                resource_id,
                now,
                expires_at,
                created_by_email,
            ),
        )
        return ShareLink(
            id=cur.lastrowid,
            org_id=self.org_id,
            token=token,
            resource_type=resource_type,
            resource_id=resource_id,
            created_at=now,
            expires_at=expires_at,
            created_by_email=created_by_email,
        )

    @SQLiteStorageBase.handle_exceptions
    def get_share_link_by_token(self, token: str) -> ShareLink | None:
        """Look up a share link by its token.

        Args:
            token (str): The share token.

        Returns:
            ShareLink | None: The share link if found, else None.
        """
        row = self._fetchone(
            "SELECT * FROM share_links WHERE org_id = ? AND token = ?",
            (self.org_id, token),
        )
        return _row_to_share_link(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def get_share_link_by_resource(
        self, resource_type: str, resource_id: str
    ) -> ShareLink | None:
        """Look up an existing share link for a specific resource.

        Args:
            resource_type (str): Type of resource.
            resource_id (str): ID of the resource.

        Returns:
            ShareLink | None: The existing share link if any, else None.
        """
        row = self._fetchone(
            "SELECT * FROM share_links WHERE org_id = ? AND resource_type = ? AND resource_id = ?",
            (self.org_id, resource_type, resource_id),
        )
        return _row_to_share_link(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def get_share_links(self) -> list[ShareLink]:
        """Return all share links for this org.

        Returns:
            list[ShareLink]: All share links, ordered by created_at ascending.
        """
        rows = self._fetchall(
            "SELECT * FROM share_links WHERE org_id = ? ORDER BY created_at ASC",
            (self.org_id,),
        )
        return [_row_to_share_link(row) for row in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_share_link(self, link_id: int) -> bool:
        """Delete a share link by ID.

        Args:
            link_id (int): The share link ID.

        Returns:
            bool: True if deleted, False if not found.
        """
        cur = self._execute(
            "DELETE FROM share_links WHERE org_id = ? AND id = ?",
            (self.org_id, link_id),
        )
        return cur.rowcount > 0

    @SQLiteStorageBase.handle_exceptions
    def delete_all_share_links(self) -> int:
        """Delete all share links for this org.

        Returns:
            int: Number of links deleted.
        """
        cur = self._execute(
            "DELETE FROM share_links WHERE org_id = ?",
            (self.org_id,),
        )
        return cur.rowcount
