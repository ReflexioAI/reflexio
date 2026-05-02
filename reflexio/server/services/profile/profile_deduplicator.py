"""
Profile deduplication service that merges duplicate profiles from multiple extractors
and against existing profiles in the database using hybrid search and LLM.
"""

import logging
import os
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from reflexio.models.api_schema.retriever_schema import SearchUserProfileRequest
from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.deduplication_utils import (
    BaseDeduplicator,
    format_dedup_timestamp,
    parse_item_id,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileTimeToLive,
    calculate_expiration_timestamp,
)

logger = logging.getLogger(__name__)


# Backward-compat alias — existing unit tests import this name from this
# module. Delegates to the shared helper in deduplication_utils.
_format_profile_timestamp = format_dedup_timestamp


# Canonical prefix emitted by the extractor for forget/delete requests. The
# dedup LLM routes matching NEW profiles into `deletions`; any fallback path
# that skips the LLM step must strip these markers before returning so they
# are never persisted as facts.
_DELETION_MARKER_PREFIX = "Requested removal of"


def _strip_deletion_markers(
    profiles: list[UserProfile],
) -> list[UserProfile]:
    """
    Drop profiles whose content is a canonical deletion marker.

    Used on fallback paths (LLM error, unexpected response type, empty dedup
    output) to prevent "Requested removal of …" markers emitted by the
    extractor from being persisted as regular profile facts when the dedup
    LLM step is skipped or yields no deletions. Persisting such markers would
    recreate the exact zombie-profile failure mode the deletion-directive
    channel was introduced to eliminate.

    Args:
        profiles (list[UserProfile]): Profiles to filter.

    Returns:
        list[UserProfile]: Profiles with deletion markers removed.
    """
    return [
        p
        for p in profiles
        if not (p.content or "").lstrip().startswith(_DELETION_MARKER_PREFIX)
    ]


# ===============================
# Profile-specific Pydantic Output Schemas for LLM
# ===============================


class ProfileDuplicateGroup(BaseModel):
    """
    Represents a group of duplicate profiles across NEW and EXISTING sets.

    Attributes:
        item_ids: List of item IDs matching prompt format (e.g., 'NEW-0', 'EXISTING-1')
        merged_content: The consolidated profile content combining information from all duplicates
        merged_time_to_live: The chosen time_to_live for the merged profile
        reasoning: Brief explanation of why these profiles are duplicates and how they were merged
    """

    item_ids: list[str] = Field(
        description="IDs of items in this group matching prompt format (e.g., 'NEW-0', 'EXISTING-1')"
    )
    merged_content: str = Field(
        description="Consolidated profile content combining all duplicate information"
    )
    merged_time_to_live: str = Field(
        description="Time to live for merged profile: one_day, one_week, one_month, one_quarter, one_year, infinity"
    )
    reasoning: str = Field(description="Brief explanation of the merge decision")

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class ProfileDeletionDirective(BaseModel):
    """
    Represents a NEW profile that is a meta-request to forget an EXISTING fact.

    Used when the user explicitly asks the system to erase a previously-stored
    profile (e.g. "forget that I like X"). Unlike a duplicate group, a deletion
    directive removes the matched EXISTING profile(s) without writing any merged
    or replacement profile — the NEW directive is consumed, not retained.

    Attributes:
        new_id: ID of the NEW profile that expresses the deletion directive (e.g. 'NEW-0')
        existing_ids: IDs of EXISTING profiles to delete without replacement (e.g. ['EXISTING-0'])
        reasoning: Brief explanation of why this was classified as a deletion directive
            rather than a fact update
    """

    new_id: str = Field(
        description="ID of the NEW profile that is a deletion directive (e.g. 'NEW-0')"
    )
    existing_ids: list[str] = Field(
        description="IDs of EXISTING profiles to delete without replacement (e.g. ['EXISTING-0'])"
    )
    reasoning: str = Field(
        description="Brief explanation of the deletion classification"
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class ProfileDeduplicationOutput(BaseModel):
    """
    Output schema for profile deduplication with NEW/EXISTING format.

    Attributes:
        duplicate_groups: List of duplicate groups to merge
        unique_ids: List of IDs of unique NEW profiles (e.g., 'NEW-2')
        deletions: List of deletion directives — NEW profiles that are pure
            meta-requests to erase an EXISTING profile. Both the NEW and the
            matched EXISTING profile(s) are removed; no merged replacement is
            produced.
    """

    duplicate_groups: list[ProfileDuplicateGroup] = Field(
        default=[], description="Groups of duplicate profiles that should be merged"
    )
    unique_ids: list[str] = Field(
        default=[],
        description="IDs of unique NEW profiles (e.g., 'NEW-2')",
    )
    deletions: list[ProfileDeletionDirective] = Field(
        default=[],
        description=(
            "NEW profiles that are pure deletion directives (the user asked to "
            "forget/remove a stored fact). Both the NEW and matched EXISTING "
            "profiles are removed; no merged replacement is written."
        ),
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class ProfileDeduplicator(BaseDeduplicator):
    """
    Deduplicates new profiles against each other and against existing profiles
    in the database using hybrid search (vector + FTS) and LLM-based merging.

    Follows the same pattern as PlaybookDeduplicator.
    """

    DEDUPLICATION_PROMPT_ID = "profile_deduplication"

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
    ):
        """
        Initialize the profile deduplicator.

        Args:
            request_context: Request context with storage and prompt manager
            llm_client: Unified LLM client for LLM calls
        """
        super().__init__(request_context, llm_client)

    def _get_prompt_id(self) -> str:
        """Get the prompt ID for profile deduplication."""
        return self.DEDUPLICATION_PROMPT_ID

    def _get_item_count_key(self) -> str:
        """Get the key name for item count in prompt variables."""
        return "new_profile_count"

    def _get_items_key(self) -> str:
        """Get the key name for items in prompt variables."""
        return "new_profiles"

    def _get_output_schema_class(self) -> type[BaseModel]:
        """Get the profile-specific output schema with NEW/EXISTING format."""
        return ProfileDeduplicationOutput

    def _format_items_for_prompt(self, profiles: list[UserProfile]) -> str:
        """
        Format profiles list for LLM prompt with NEW-N prefix.

        Args:
            profiles: List of profiles

        Returns:
            Formatted string representation
        """
        return self._format_profiles_with_prefix(profiles, "NEW")

    def _format_profiles_with_prefix(
        self, profiles: list[UserProfile], prefix: str
    ) -> str:
        """
        Format profiles with a given prefix (NEW or EXISTING).

        Args:
            profiles: List of profiles to format
            prefix: Prefix string for indices

        Returns:
            Formatted string
        """
        if not profiles:
            return "(None)"
        lines = []
        for idx, profile in enumerate(profiles):
            ttl = (
                profile.profile_time_to_live.value
                if profile.profile_time_to_live
                else "unknown"
            )
            source = profile.source or "unknown"
            modified_date = _format_profile_timestamp(profile.last_modified_timestamp)
            lines.append(
                f'[{prefix}-{idx}] Content: "{profile.content}" | TTL: {ttl} | Source: {source} | Last Modified: {modified_date}'
            )
        return "\n".join(lines)

    def _format_new_and_existing_for_prompt(
        self,
        new_profiles: list[UserProfile],
        existing_profiles: list[UserProfile],
    ) -> tuple[str, str]:
        """
        Format new and existing profiles for the deduplication prompt.

        Args:
            new_profiles: New profiles to deduplicate
            existing_profiles: Existing profiles from the database

        Returns:
            Tuple of (new_profiles_text, existing_profiles_text)
        """
        new_text = self._format_profiles_with_prefix(new_profiles, "NEW")
        existing_text = self._format_profiles_with_prefix(existing_profiles, "EXISTING")
        return new_text, existing_text

    def _retrieve_existing_profiles(
        self,
        new_profiles: list[UserProfile],
        user_id: str,
    ) -> list[UserProfile]:
        """
        Retrieve existing profiles from the database using hybrid search.

        For each new profile, uses its profile_content as the query with
        pre-computed embeddings for vector search.

        Args:
            new_profiles: List of new profiles to search against
            user_id: User ID to scope the search

        Returns:
            Deduplicated list of existing UserProfile objects from the database
        """
        storage = self.request_context.storage

        # Collect profile content strings for embedding
        query_texts = []
        for profile in new_profiles:
            text = profile.content
            if text and text.strip():
                query_texts.append(text.strip())

        if not query_texts:
            return []

        # Batch-generate embeddings
        try:
            embeddings = self.client.get_embeddings(
                query_texts, dimensions=EMBEDDING_DIMENSIONS
            )
        except Exception as e:
            logger.warning("Failed to generate embeddings for dedup search: %s", e)
            embeddings = [None] * len(query_texts)

        # Search for each new profile
        seen_ids: set[str] = set()
        existing_profiles: list[UserProfile] = []

        for i, query_text in enumerate(query_texts):
            try:
                results = storage.search_user_profile(  # type: ignore[reportOptionalMemberAccess]
                    SearchUserProfileRequest(
                        query=query_text,
                        user_id=user_id,
                        top_k=10,
                        threshold=0.4,
                    ),
                    status_filter=[None],  # Only current profiles
                    query_embedding=embeddings[i],
                )
                for profile in results:
                    if profile.profile_id and profile.profile_id not in seen_ids:
                        seen_ids.add(profile.profile_id)
                        existing_profiles.append(profile)
            except Exception as e:  # noqa: PERF203
                logger.warning(
                    "Failed to search existing profiles for query %d: %s", i, e
                )

        logger.info(
            "Retrieved %d unique existing profiles for deduplication",
            len(existing_profiles),
        )
        return existing_profiles

    def deduplicate(
        self,
        new_profiles: list[UserProfile],
        user_id: str,
        request_id: str,
    ) -> tuple[list[UserProfile], list[str], list[UserProfile]]:
        """
        Deduplicate profiles across extractors and against existing profiles in DB.

        Args:
            new_profiles: List of new UserProfile objects from extractors
            request_id: Request ID for context
            user_id: User ID to scope the existing profile search

        Returns:
            Tuple of (deduplicated profiles, existing profile IDs to delete, superseded existing profiles)
        """
        # Check if mock mode is enabled
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            logger.info("Mock mode: skipping deduplication")
            return new_profiles, [], []

        if not new_profiles:
            return [], [], []

        # Retrieve existing profiles via hybrid search
        existing_profiles = self._retrieve_existing_profiles(new_profiles, user_id)

        # Format for prompt
        new_text, existing_text = self._format_new_and_existing_for_prompt(
            new_profiles, existing_profiles
        )

        # Build and call LLM
        prompt = self.request_context.prompt_manager.render_prompt(
            self._get_prompt_id(),
            {
                "new_profile_count": len(new_profiles),
                "new_profiles": new_text,
                "existing_profile_count": len(existing_profiles),
                "existing_profiles": existing_text,
            },
        )

        output_schema_class = self._get_output_schema_class()

        try:
            from reflexio.server.services.service_utils import (
                log_llm_messages,
                log_model_response,
            )

            log_llm_messages(
                logger, "Profile deduplication", [{"role": "user", "content": prompt}]
            )

            response = self.client.generate_chat_response(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format=output_schema_class,
            )

            log_model_response(logger, "Deduplication response", response)

            if not isinstance(response, ProfileDeduplicationOutput):
                logger.warning(
                    "Unexpected response type from deduplication LLM: %s",
                    type(response),
                )
                return _strip_deletion_markers(new_profiles), [], []

            dedup_output = response
        except Exception as e:
            logger.error("Failed to identify duplicates: %s", str(e))
            return _strip_deletion_markers(new_profiles), [], []

        if not dedup_output.duplicate_groups and not dedup_output.deletions:
            logger.info("No duplicate or deletion actions for request %s", request_id)
            return _strip_deletion_markers(new_profiles), [], []

        logger.info(
            "Found %d duplicate profile groups and %d deletion directives for request %s",
            len(dedup_output.duplicate_groups),
            len(dedup_output.deletions),
            request_id,
        )

        # Build deduplicated result
        return self._build_deduplicated_results(
            new_profiles=new_profiles,
            existing_profiles=existing_profiles,
            dedup_output=dedup_output,
            user_id=user_id,
            request_id=request_id,
        )

    def _build_deduplicated_results(
        self,
        new_profiles: list[UserProfile],
        existing_profiles: list[UserProfile],
        dedup_output: ProfileDeduplicationOutput,
        user_id: str,
        request_id: str,
    ) -> tuple[list[UserProfile], list[str], list[UserProfile]]:
        """
        Build the deduplicated profile list from LLM output.

        Args:
            new_profiles: Flattened list of new profiles
            existing_profiles: List of existing profiles from DB
            dedup_output: LLM deduplication output
            user_id: User ID
            request_id: Request ID

        Returns:
            Tuple of (profiles ready to save, existing profile IDs to delete, superseded existing profiles)
        """
        handled_new_indices: set[int] = set()
        result_profiles: list[UserProfile] = []
        existing_ids_to_delete: list[str] = []
        seen_delete_ids: set[str] = set()
        superseded_profiles: list[UserProfile] = []

        now_ts = int(datetime.now(UTC).timestamp())

        # Process deletion directives first. A directive is a NEW profile that
        # is a meta-request to forget an EXISTING profile. Both the NEW and the
        # matched EXISTING profile(s) are removed with no merged replacement.
        self._apply_deletion_directives(
            dedup_output.deletions,
            new_profiles=new_profiles,
            existing_profiles=existing_profiles,
            handled_new_indices=handled_new_indices,
            existing_ids_to_delete=existing_ids_to_delete,
            seen_delete_ids=seen_delete_ids,
            superseded_profiles=superseded_profiles,
        )

        # Process duplicate groups
        for group in dedup_output.duplicate_groups:
            group_new_indices: list[int] = []
            group_existing_indices: list[int] = []

            for item_id in group.item_ids:
                parsed = parse_item_id(item_id)
                if parsed is None:
                    continue
                prefix, idx = parsed
                if prefix == "NEW":
                    group_new_indices.append(idx)
                elif prefix == "EXISTING":
                    group_existing_indices.append(idx)

            # Reject groups that overlap with profiles already consumed by a
            # deletion directive. Merging such a group would write a
            # replacement profile containing content the user asked to forget.
            conflicting_new = [i for i in group_new_indices if i in handled_new_indices]
            conflicting_existing = [
                i
                for i in group_existing_indices
                if 0 <= i < len(existing_profiles)
                and existing_profiles[i].profile_id
                and existing_profiles[i].profile_id in seen_delete_ids
            ]
            if conflicting_new or conflicting_existing:
                logger.warning(
                    "Skipping duplicate group %s: overlaps with deletion "
                    "directives (NEW indices=%s, EXISTING indices=%s)",
                    group.item_ids,
                    conflicting_new,
                    conflicting_existing,
                )
                continue

            # Mark NEW indices as handled only after the overlap check passes.
            for idx in group_new_indices:
                handled_new_indices.add(idx)

            # Collect existing profile IDs to delete and their profiles for changelog (deduplicated)
            for eidx in group_existing_indices:
                self._mark_existing_for_deletion(
                    f"EXISTING-{eidx}",
                    existing_profiles,
                    existing_ids_to_delete,
                    seen_delete_ids,
                    superseded_profiles,
                )

            # Get template from first NEW profile in group (for metadata)
            template_profile: UserProfile | None = None
            if group_new_indices:
                first_new_idx = group_new_indices[0]
                if 0 <= first_new_idx < len(new_profiles):
                    template_profile = new_profiles[first_new_idx]

            if template_profile is None:
                logger.warning("Could not find template profile for group, skipping")
                continue

            # Merge custom_features from all NEW profiles in group
            group_new_profiles = [
                new_profiles[i] for i in group_new_indices if 0 <= i < len(new_profiles)
            ]
            merged_custom_features = self._merge_custom_features(group_new_profiles)

            # Merge extractor_names from all NEW profiles in group
            merged_extractor_names = self._merge_extractor_names(group_new_profiles)

            # Determine TTL
            try:
                ttl = ProfileTimeToLive(group.merged_time_to_live)
            except ValueError:
                ttl = template_profile.profile_time_to_live
                logger.warning(
                    "Invalid TTL '%s' from LLM, using template TTL '%s'",
                    group.merged_time_to_live,
                    ttl.value,
                )

            merged_profile = UserProfile(
                profile_id=str(uuid.uuid4()),
                user_id=user_id,
                content=group.merged_content,
                last_modified_timestamp=now_ts,
                generated_from_request_id=request_id,
                profile_time_to_live=ttl,
                expiration_timestamp=calculate_expiration_timestamp(now_ts, ttl),
                custom_features=merged_custom_features,
                source=template_profile.source,
                status=template_profile.status,
                extractor_names=merged_extractor_names,
            )
            result_profiles.append(merged_profile)

        # Add unique NEW profiles
        for uid in dedup_output.unique_ids:
            parsed = parse_item_id(uid)
            if parsed is None:
                continue
            prefix, idx = parsed
            if (
                prefix == "NEW"
                and idx not in handled_new_indices
                and 0 <= idx < len(new_profiles)
            ):
                result_profiles.append(new_profiles[idx])
                handled_new_indices.add(idx)

        # Safety fallback: add any NEW profiles not mentioned by LLM
        for idx, profile in enumerate(new_profiles):
            if idx not in handled_new_indices:
                logger.warning(
                    "New profile at index %d was not handled by LLM, adding as-is",
                    idx,
                )
                result_profiles.append(profile)

        return result_profiles, existing_ids_to_delete, superseded_profiles

    def _apply_deletion_directives(
        self,
        directives: list[ProfileDeletionDirective],
        *,
        new_profiles: list[UserProfile],
        existing_profiles: list[UserProfile],
        handled_new_indices: set[int],
        existing_ids_to_delete: list[str],
        seen_delete_ids: set[str],
        superseded_profiles: list[UserProfile],
    ) -> None:
        """
        Apply deletion directives in place: consume the NEW profile and mark matched
        EXISTING profile(s) for deletion without producing a merged replacement.

        A directive is a NEW profile whose content is a meta-request to forget an
        EXISTING profile (e.g. "Requested removal of interest in X from stored
        profiles"). The NEW is suppressed from the result set and the matched
        EXISTING rows are added to the deletion list.

        Args:
            directives: Deletion directives from the LLM.
            new_profiles: Flat list of NEW profiles (indexed by NEW-N id).
            existing_profiles: List of EXISTING profiles (indexed by EXISTING-M id).
            handled_new_indices: Set of NEW indices already accounted for; this
                method adds the consumed directive indices to it.
            existing_ids_to_delete: Output list of profile IDs to delete; this
                method appends to it.
            seen_delete_ids: Set used to deduplicate IDs across all deletion paths.
            superseded_profiles: Output list of deleted profile objects for the
                changelog; this method appends to it.
        """
        for directive in directives:
            self._consume_new_index(
                directive.new_id, len(new_profiles), handled_new_indices
            )
            for eid in directive.existing_ids:
                self._mark_existing_for_deletion(
                    eid,
                    existing_profiles,
                    existing_ids_to_delete,
                    seen_delete_ids,
                    superseded_profiles,
                )
            logger.info(
                "Profile deletion directive %s -> delete %s: %s",
                directive.new_id,
                directive.existing_ids,
                directive.reasoning,
            )

    @staticmethod
    def _consume_new_index(
        new_id: str, new_profile_count: int, handled_new_indices: set[int]
    ) -> None:
        """Mark a NEW-N id as handled so the safety fallback does not re-add it."""
        parsed = parse_item_id(new_id)
        if parsed is None:
            return
        prefix, idx = parsed
        if prefix == "NEW" and 0 <= idx < new_profile_count:
            handled_new_indices.add(idx)

    @staticmethod
    def _mark_existing_for_deletion(
        existing_id: str,
        existing_profiles: list[UserProfile],
        existing_ids_to_delete: list[str],
        seen_delete_ids: set[str],
        superseded_profiles: list[UserProfile],
    ) -> None:
        """Resolve an EXISTING-N id to a profile_id and queue it for deletion."""
        parsed = parse_item_id(existing_id)
        if parsed is None:
            return
        prefix, idx = parsed
        if prefix != "EXISTING" or not (0 <= idx < len(existing_profiles)):
            return
        pid = existing_profiles[idx].profile_id
        if pid and pid not in seen_delete_ids:
            seen_delete_ids.add(pid)
            existing_ids_to_delete.append(pid)
            superseded_profiles.append(existing_profiles[idx])

    def _merge_custom_features(self, profiles: list[UserProfile]) -> dict | None:
        """
        Merge custom_features from multiple profiles.

        Args:
            profiles: List of profiles to merge custom_features from

        Returns:
            Merged custom_features dict or None if no custom_features
        """
        merged = {}
        for profile in profiles:
            if profile.custom_features:
                merged.update(profile.custom_features)

        return merged or None

    def _merge_extractor_names(self, profiles: list[UserProfile]) -> list[str] | None:
        """
        Merge extractor_names from multiple profiles, preserving order and removing duplicates.

        Args:
            profiles: List of profiles to merge extractor_names from

        Returns:
            Merged list of unique extractor names or None if no extractor_names
        """
        seen: set[str] = set()
        merged: list[str] = []
        for profile in profiles:
            if profile.extractor_names:
                for name in profile.extractor_names:
                    if name not in seen:
                        seen.add(name)
                        merged.append(name)
        return merged or None
