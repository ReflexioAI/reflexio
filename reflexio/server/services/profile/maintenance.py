"""Source-authorized profile deadline work on the existing maintenance owner."""

from collections.abc import Callable
from concurrent.futures import CancelledError
from typing import Any

from reflexio.server.services.profile.lifecycle import whole_epoch
from reflexio.server.services.profile.lifecycle_deadlines import (
    NativeLifecycleDeadlineStore,
)


def purge_due_profile_payloads(
    storage: Any,
    now: int,
    project_id: str = "",
    *,
    checkpoint: Callable[[], None] | None = None,
) -> int:
    """Page only due subjects, then revalidate each row's own exact authority.

    One failed subject must not prevent eligible siblings from being maintained.
    Surface the first failure to the existing project maintenance observer after
    processing the other subjects; a failed pass is retried by that same owner.
    """
    whole_epoch(now)
    after = ""
    deleted = 0
    first_failure: Exception | None = None
    while True:
        if checkpoint is not None:
            checkpoint()
        users = storage.profile_lifecycle_due_users(now=now, after_user_id=after)
        if not users:
            if first_failure is not None:
                raise first_failure
            return deleted
        page_after = after
        page_users: set[str] = set()
        for user_id in users:
            if checkpoint is not None:
                checkpoint()
            # SQL owns collation/order. Python ordering can differ for opaque
            # mixed-case or non-ASCII subject IDs, so only reject duplicates.
            if (
                not isinstance(user_id, str)
                or not user_id
                or user_id == page_after
                or user_id in page_users
            ):
                raise ValueError("Profile deadline subject scan did not advance")
            page_users.add(user_id)
            after = user_id
            try:
                results = NativeLifecycleDeadlineStore(
                    storage, user_id, project_id
                ).purge_due(now, checkpoint=checkpoint)
                deleted += sum(result["status"] == "hard_deleted" for result in results)
            except CancelledError:
                raise
            except Exception as error:
                if first_failure is None:
                    first_failure = error

        if after == page_after:
            raise ValueError("Profile deadline subject page did not advance")
