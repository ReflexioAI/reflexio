import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from reflexio.models.api_schema.retriever_schema import (
    SearchInteractionRequest,
    SearchUserProfileRequest,
)
from reflexio.models.api_schema.service_schemas import (
    DeleteUserInteractionRequest,
    DeleteUserProfileRequest,
    Interaction,
    Status,
    UserProfile,
)
from reflexio.server.services.storage.storage_base import matches_status_filter

logger = logging.getLogger(__name__)


class ProfileMixin:
    # ------------------------------------------------------------------
    # Profile methods
    # ------------------------------------------------------------------

    def get_all_profiles(
        self,
        limit: int = 100,
        status_filter: list[Status | None] | None = None,
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]

        with self._lock:
            all_profiles = self._list_entities_recursive(
                self._profiles_dir(), UserProfile
            )

        profiles = [
            p for p in all_profiles if matches_status_filter(p.status, status_filter)
        ]
        profiles.sort(key=lambda x: x.last_modified_timestamp, reverse=True)
        return profiles[:limit]

    def get_user_profile(
        self,
        user_id: str,
        status_filter: list[Status | None] | None = None,
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]

        user_dir = self._user_dir(self._profiles_dir(), user_id)
        with self._lock:
            all_profiles = self._list_entities(user_dir, UserProfile)

        if not all_profiles:
            logger.warning(
                "get_user_profile::User profile not found for user id: %s", user_id
            )
            return []

        return [
            p for p in all_profiles if matches_status_filter(p.status, status_filter)
        ]

    def add_user_profile(self, user_id: str, user_profiles: list[UserProfile]) -> None:
        with self._lock:
            for profile in user_profiles:
                path = self._entity_path(
                    self._user_dir(self._profiles_dir(), user_id),
                    str(profile.profile_id),
                )
                self._write_entity(path, profile)
                self._write_embedding(path, profile.embedding)
        self._trigger_qmd_update()

    def delete_user_profile(self, request: DeleteUserProfileRequest) -> None:
        with self._lock:
            path = self._entity_path(
                self._user_dir(self._profiles_dir(), request.user_id),
                str(request.profile_id),
            )
            if path.exists():
                self._delete_embedding(path)
                path.unlink()

    def update_user_profile_by_id(
        self, user_id: str, profile_id: str, new_profile: UserProfile
    ) -> None:
        with self._lock:
            path = self._entity_path(
                self._user_dir(self._profiles_dir(), user_id),
                profile_id,
            )
            if not path.exists():
                logger.warning(
                    "update_user_profile_by_id::User profile not found for user id: %s",
                    user_id,
                )
                return
            self._write_entity(path, new_profile)
            self._write_embedding(path, new_profile.embedding)

    def delete_all_profiles_for_user(self, user_id: str) -> None:
        with self._lock:
            user_dir = self._user_dir(self._profiles_dir(), user_id)
            if user_dir.exists():
                shutil.rmtree(user_dir)
                user_dir.mkdir(parents=True, exist_ok=True)

    def delete_all_profiles(self) -> None:
        """Delete all profiles across all users."""
        with self._lock:
            self._clear_dir(self._profiles_dir())

    def count_all_profiles(self) -> int:
        with self._lock:
            profiles_dir = self._profiles_dir()
            if not profiles_dir.exists():
                return 0
            return len(self._scan_entities(profiles_dir, recursive=True))

    def update_all_profiles_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        user_ids: list[str] | None = None,
    ) -> int:
        with self._lock:
            updated_count = 0
            profiles_dir = self._profiles_dir()
            if not profiles_dir.exists():
                return 0

            for profile_path in self._scan_entities(profiles_dir, recursive=True):
                # Extract user_id from path: profiles/{user_id}/{profile_id}
                user_id = profile_path.parent.name
                if user_ids is not None and user_id not in user_ids:
                    continue

                profile_obj = self._read_entity(profile_path, UserProfile)

                status_matches = False
                if old_status is None or (
                    hasattr(old_status, "value") and old_status.value is None
                ):
                    if profile_obj.status is None:
                        status_matches = True
                elif (
                    isinstance(old_status, Status) and profile_obj.status == old_status
                ):
                    status_matches = True

                if status_matches:
                    profile_obj.status = new_status
                    profile_obj.last_modified_timestamp = int(
                        datetime.now(UTC).timestamp()
                    )
                    self._write_entity(profile_path, profile_obj)
                    self._write_embedding(profile_path, profile_obj.embedding)
                    updated_count += 1

            logger.info(
                "Updated %s profiles from %s to %s",
                updated_count,
                old_status,
                new_status,
            )
            return updated_count

    def delete_all_profiles_by_status(self, status: Status) -> int:
        with self._lock:
            deleted_count = 0
            profiles_dir = self._profiles_dir()
            if not profiles_dir.exists():
                return 0

            for profile_path in self._scan_entities(profiles_dir, recursive=True):
                profile_obj = self._read_entity(profile_path, UserProfile)
                if isinstance(status, Status) and profile_obj.status == status:
                    self._delete_embedding(profile_path)
                    profile_path.unlink()
                    deleted_count += 1

            logger.info("Deleted %s profiles with status %s", deleted_count, status)
            return deleted_count

    def get_user_ids_with_status(self, status: Status | None) -> list[str]:
        with self._lock:
            profiles_dir = self._profiles_dir()
            if not profiles_dir.exists():
                return []

            user_ids_with_status: list[str] = []
            # Iterate user directories
            for user_dir in sorted(profiles_dir.iterdir()):
                if not user_dir.is_dir():
                    continue
                for profile_path in self._scan_entities(user_dir):
                    profile_obj = self._read_entity(profile_path, UserProfile)
                    status_matches = False
                    if status is None or (
                        hasattr(status, "value") and status.value is None
                    ):
                        if profile_obj.status is None:
                            status_matches = True
                    elif isinstance(status, Status) and profile_obj.status == status:
                        status_matches = True

                    if status_matches:
                        user_ids_with_status.append(user_dir.name)
                        break
            return user_ids_with_status

    def delete_profiles_by_ids(self, profile_ids: list[str]) -> int:
        if not profile_ids:
            return 0
        profile_id_set = set(profile_ids)
        with self._lock:
            deleted_count = 0
            for p in self._scan_entities(self._profiles_dir(), recursive=True):
                profile = self._read_entity(p, UserProfile)
                if profile.profile_id in profile_id_set:
                    self._delete_embedding(p)
                    p.unlink()
                    deleted_count += 1
            return deleted_count

    # ------------------------------------------------------------------
    # Interaction methods
    # ------------------------------------------------------------------

    def get_all_interactions(self, limit: int = 100) -> list[Interaction]:
        with self._lock:
            interactions = self._list_entities_recursive(
                self._interactions_dir(), Interaction
            )
        interactions.sort(key=lambda x: x.created_at, reverse=True)
        return interactions[:limit]

    def get_user_interaction(self, user_id: str) -> list[Interaction]:
        user_dir = self._user_dir(self._interactions_dir(), user_id)
        with self._lock:
            interactions = self._list_entities(user_dir, Interaction)
        if not interactions:
            logger.warning(
                "get_user_interaction::User interaction not found for user id: %s",
                user_id,
            )
        return interactions

    def add_user_interaction(self, user_id: str, interaction: Interaction) -> None:
        with self._lock:
            if interaction.interaction_id == 0:
                interaction.interaction_id = self._next_id(self._interactions_dir())

            path = self._entity_path(
                self._user_dir(self._interactions_dir(), user_id),
                str(interaction.interaction_id),
            )
            self._write_entity(path, interaction)
            self._write_embedding(path, interaction.embedding)

    def add_user_interactions_bulk(
        self, user_id: str, interactions: list[Interaction]
    ) -> None:
        """Add multiple user interactions at once.

        Args:
            user_id (str): The user ID
            interactions (list[Interaction]): List of interactions to add
        """
        if not interactions:
            return

        with self._lock:
            next_id = self._next_id(self._interactions_dir())
            for interaction in interactions:
                if interaction.interaction_id == 0:
                    interaction.interaction_id = next_id
                    next_id += 1
                path = self._entity_path(
                    self._user_dir(self._interactions_dir(), user_id),
                    str(interaction.interaction_id),
                )
                self._write_entity(path, interaction)
                self._write_embedding(path, interaction.embedding)
        self._trigger_qmd_update()

    def delete_user_interaction(self, request: DeleteUserInteractionRequest) -> None:
        with self._lock:
            path = self._entity_path(
                self._user_dir(self._interactions_dir(), request.user_id),
                str(request.interaction_id),
            )
            if path.exists():
                self._delete_embedding(path)
                path.unlink()

    def delete_all_interactions_for_user(self, user_id: str) -> None:
        with self._lock:
            user_dir = self._user_dir(self._interactions_dir(), user_id)
            if user_dir.exists():
                shutil.rmtree(user_dir)
                user_dir.mkdir(parents=True, exist_ok=True)

    def delete_all_interactions(self) -> None:
        """Delete all interactions across all users."""
        with self._lock:
            self._clear_dir(self._interactions_dir())

    def count_all_interactions(self) -> int:
        with self._lock:
            return len(self._scan_entities(self._interactions_dir(), recursive=True))

    def delete_oldest_interactions(self, count: int) -> int:
        if count <= 0:
            return 0

        with self._lock:
            interactions_dir = self._interactions_dir()

            # Collect all interactions with their paths
            all_entries: list[tuple[Path, Interaction]] = []
            for p in self._scan_entities(interactions_dir, recursive=True):
                interaction = self._read_entity(p, Interaction)
                all_entries.append((p, interaction))

            if not all_entries:
                return 0

            # Sort by created_at (oldest first)
            all_entries.sort(key=lambda x: x[1].created_at or 0)

            # Delete oldest N
            to_delete = all_entries[:count]
            for path, _ in to_delete:
                self._delete_embedding(path)
                path.unlink()

            return len(to_delete)

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def search_interaction(
        self,
        search_interaction_request: SearchInteractionRequest,
        query_embedding: list[float] | None = None,  # noqa: ARG002
    ) -> list[Interaction]:
        query = search_interaction_request.query
        user_id = search_interaction_request.user_id

        # Try QMD search when a query is provided
        if query:
            qmd_results = self._qmd.search(
                query,
                mode=search_interaction_request.search_mode,
                top_k=search_interaction_request.top_k or 100,
            )
            if qmd_results:
                interactions_dir = self._interactions_dir()
                user_dir = self._user_dir(interactions_dir, user_id)
                matched: list[Interaction] = []
                for r in qmd_results:
                    filepath = Path(r.filepath)
                    # Only include files under this user's interactions dir
                    if not filepath.is_relative_to(user_dir):
                        continue
                    if not filepath.exists():
                        continue
                    interaction = self._read_entity(filepath, Interaction)
                    # Post-filter by request_id
                    if (
                        search_interaction_request.request_id
                        and interaction.request_id
                        != search_interaction_request.request_id
                    ):
                        continue
                    # Post-filter by time range
                    if (
                        search_interaction_request.start_time
                        and interaction.created_at
                        < search_interaction_request.start_time.timestamp()
                    ):
                        continue
                    if (
                        search_interaction_request.end_time
                        and interaction.created_at
                        > search_interaction_request.end_time.timestamp()
                    ):
                        continue
                    matched.append(interaction)
                return matched

        # Fallback: Python substring matching
        interactions = self.get_user_interaction(user_id)
        if search_interaction_request.request_id:
            interactions = [
                i
                for i in interactions
                if i.request_id == search_interaction_request.request_id
            ]
        if query:
            interactions = [i for i in interactions if query in i.content]
        if search_interaction_request.start_time:
            interactions = [
                i
                for i in interactions
                if i.created_at >= search_interaction_request.start_time.timestamp()
            ]
        if search_interaction_request.end_time:
            interactions = [
                i
                for i in interactions
                if i.created_at <= search_interaction_request.end_time.timestamp()
            ]
        return interactions

    def search_user_profile(
        self,
        search_user_profile_request: SearchUserProfileRequest,
        status_filter: list[Status | None] | None = None,
        query_embedding: list[float] | None = None,  # noqa: ARG002
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]

        query = search_user_profile_request.query
        user_id = search_user_profile_request.user_id

        # Try QMD search when a query is provided
        if query:
            qmd_results = self._qmd.search(
                query,
                mode=search_user_profile_request.search_mode,
                top_k=search_user_profile_request.top_k or 10,
            )
            if qmd_results:
                profiles_dir = self._profiles_dir()
                user_dir = self._user_dir(profiles_dir, user_id)
                matched: list[UserProfile] = []
                for r in qmd_results:
                    filepath = Path(r.filepath)
                    # Only include files under this user's profiles dir
                    if not filepath.is_relative_to(user_dir):
                        continue
                    if not filepath.exists():
                        continue
                    profile = self._read_entity(filepath, UserProfile)
                    # Post-filter by status
                    if not matches_status_filter(profile.status, status_filter):
                        continue
                    # Post-filter by generated_from_request_id
                    if (
                        search_user_profile_request.generated_from_request_id
                        and profile.generated_from_request_id
                        != search_user_profile_request.generated_from_request_id
                    ):
                        continue
                    # Post-filter by time range
                    if (
                        search_user_profile_request.start_time
                        and profile.last_modified_timestamp
                        < search_user_profile_request.start_time.timestamp()
                    ):
                        continue
                    if (
                        search_user_profile_request.end_time
                        and profile.last_modified_timestamp
                        > search_user_profile_request.end_time.timestamp()
                    ):
                        continue
                    matched.append(profile)
                if search_user_profile_request.top_k:
                    matched = matched[: search_user_profile_request.top_k]
                return matched

        # Fallback: Python substring matching
        user_profiles = self.get_user_profile(user_id, status_filter=status_filter)
        if search_user_profile_request.generated_from_request_id:
            user_profiles = [
                p
                for p in user_profiles
                if p.generated_from_request_id
                == search_user_profile_request.generated_from_request_id
            ]
        if query:
            user_profiles = [p for p in user_profiles if query in p.content]
        if search_user_profile_request.start_time:
            user_profiles = [
                p
                for p in user_profiles
                if p.last_modified_timestamp
                >= search_user_profile_request.start_time.timestamp()
            ]
        if search_user_profile_request.end_time:
            user_profiles = [
                p
                for p in user_profiles
                if p.last_modified_timestamp
                <= search_user_profile_request.end_time.timestamp()
            ]
        if search_user_profile_request.top_k:
            user_profiles = user_profiles[: search_user_profile_request.top_k]

        return user_profiles
