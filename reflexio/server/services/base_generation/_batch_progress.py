"""Batch-progress + rerun orchestration for ``BaseGenerationService`` (Tier-1b decomposition).

``BatchProgressMixin`` holds the shared batch-with-progress driver
(``_run_batch_with_progress``) plus the ``run_rerun`` public template method and the
overridable rerun hooks subclasses implement (``_get_rerun_user_ids``,
``_build_rerun_request_params``, ``_create_run_request_for_item``,
``_create_rerun_response``, ``_get_generated_count``, ``_pre_process_rerun``).

``_run_batch_with_progress`` is both subclass-called (``self._run_batch_with_progress``
in the profile/playbook manual-run paths, resolved via MRO) and instance-patched in
tests (``patch.object(svc, "_run_batch_with_progress")``, which is location-agnostic).
It is the sole WRITER of ``self._is_batch_mode`` (read by the base ``run``); that field
is initialised on the base ``__init__`` and stubbed annotation-only below. ``run_rerun``
is PUBLIC (dispatched via ``run_method="run_rerun"`` from ``lib/_generation.py``) so its
signature is byte-identical. Method bodies are moved VERBATIM from the former monolithic
``base_generation_service.py``.
"""

import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from reflexio.server.services.operation_state_utils import OperationStateManager

logger = logging.getLogger(__name__)

TRequest = TypeVar("TRequest")


class BatchProgressMixin(Generic[TRequest]):  # noqa: UP046
    """Batch-with-progress driver + rerun template method and hooks.

    Mixed into ``BaseGenerationService`` ahead of the other mixins and ``ABC``. Every
    ``self.`` call resolves either to a sibling hook on this mixin or, via MRO, to a
    base-owned method (``run``, ``_create_state_manager``, ``_get_base_service_name``)
    or the base-initialised ``_is_batch_mode`` field — the annotation-only / type-only
    stubs below give pyright the types without introducing shared class-level state.
    """

    # Annotation-only stub for the base-owned field this driver WRITES (init'd in the
    # base ``__init__``). NEVER assign here — a class-level default would be a
    # shared-state footgun on a mixin.
    _is_batch_mode: bool

    if TYPE_CHECKING:
        # Abstract on the base ABC (stays there per SINK-2); declared type-only so
        # pyright can resolve the ``self._get_base_service_name()`` call. No runtime
        # attribute is added, so ``__abstractmethods__`` is unaffected.
        def _get_base_service_name(self) -> str: ...
        # Concrete on the base (the FIFO-drain orchestrator + lock/queue helper);
        # resolved via MRO. Declared type-only so pyright can resolve the calls.
        def run(self, request: TRequest) -> None: ...
        def _create_state_manager(self) -> OperationStateManager: ...

    # ===============================
    # Batch with progress (shared by rerun + manual)
    # ===============================

    def _run_batch_with_progress(
        self,
        user_ids: list[str],
        request: TRequest,
        request_params: dict,
        state_manager: OperationStateManager,
    ) -> tuple[int, int]:
        """Run a batch of users with progress tracking.

        Shared logic for both run_rerun() and run_manual_regular().
        Initializes progress, processes each user, and finalizes.
        Checks for cancellation before each user.

        Args:
            user_ids: List of user IDs to process
            request: The original request object
            request_params: Parameters dict for progress state
            state_manager: OperationStateManager instance

        Returns:
            Tuple of (users_processed, total_generated)
        """
        total_users = len(user_ids)
        self._is_batch_mode = True

        # Initialize progress
        state_manager.initialize_progress(
            total_users=total_users,
            request_params=request_params,
        )

        try:
            # Process each user
            users_processed = 0
            processed_user_ids: list[str] = []
            for user_id in user_ids:
                # Check for cancellation before starting next user
                if state_manager.is_cancellation_requested():
                    logger.info(
                        "Cancellation requested for %s, stopping after %d/%d users",
                        self._get_base_service_name(),
                        users_processed,
                        total_users,
                    )
                    state_manager.mark_cancelled()
                    return users_processed, self._get_generated_count(
                        request, processed_user_ids=processed_user_ids
                    )

                state_manager.set_current_item(user_id)

                try:
                    run_request = self._create_run_request_for_item(user_id, request)
                    self.run(run_request)
                    users_processed += 1
                    processed_user_ids.append(user_id)

                    state_manager.update_progress(
                        item_id=user_id,
                        count=0,  # Extractors collect their own data
                        success=True,
                        total_users=total_users,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process user %s for %s: %s",
                        user_id,
                        self._get_base_service_name(),
                        str(e),
                    )
                    state_manager.update_progress(
                        item_id=user_id,
                        count=0,
                        success=False,
                        total_users=total_users,
                        error=str(e),
                    )
                    continue

            # Get generated count and finalize
            total_generated = self._get_generated_count(
                request, processed_user_ids=processed_user_ids
            )
            state_manager.finalize_progress(users_processed, total_generated)

            return users_processed, total_generated
        finally:
            self._is_batch_mode = False

    # ===============================
    # Rerun methods (optional - override to enable rerun functionality)
    # ===============================

    def _get_rerun_user_ids(self, request: TRequest) -> list[str]:
        """Get user IDs to process during rerun.

        Override this method to enable rerun functionality for the service.
        Returns a list of user IDs that have interactions matching the request filters.
        Each extractor collects its own data using its configured window_size.

        Args:
            request: The rerun request object

        Returns:
            List of user IDs to process
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _build_rerun_request_params(self, request: TRequest) -> dict:
        """Build request params dict for operation state tracking.

        Override this method to enable rerun functionality for the service.

        Args:
            request: The rerun request object

        Returns:
            Dictionary of request parameters for state tracking
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _create_run_request_for_item(self, user_id: str, request: TRequest) -> TRequest:
        """Create the request object to pass to self.run() for a single user.

        Override this method to enable rerun functionality for the service.
        Each extractor collects its own data using its configured window_size.

        Args:
            user_id: The user ID to process
            request: The original rerun request object

        Returns:
            A request object suitable for self.run()
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _create_rerun_response(self, success: bool, msg: str, count: int) -> Any:
        """Create the rerun response object.

        Override this method to enable rerun functionality for the service.

        Args:
            success: Whether the operation succeeded
            msg: Status message
            count: Number of items generated

        Returns:
            A response object (e.g., RerunProfileGenerationResponse)
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _get_generated_count(
        self,
        request: TRequest,
        processed_user_ids: list[str] | None = None,
    ) -> int:
        """Get the count of generated items (profiles or playbooks) after rerun.

        Override this method to enable rerun functionality for the service.

        Args:
            request: The rerun request object (for filtering)
            processed_user_ids: List of user IDs that were successfully processed
                in the batch. Provided by _run_batch_with_progress so overrides
                don't need to handle user_id=None from batch requests.

        Returns:
            Number of items generated during rerun
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _pre_process_rerun(self, request: TRequest) -> None:  # noqa: B027
        """Hook called before processing rerun items.

        Override in subclasses to perform cleanup or preparation before rerun.
        Default implementation does nothing.

        Args:
            request: The rerun request object
        """

    def run_rerun(self, request: TRequest) -> Any:
        """Run the rerun workflow for the service.

        This template method orchestrates the rerun process:
        1. Check for existing in-progress operations
        2. Get user IDs to process
        3. Pre-process hook
        4. Run batch with progress tracking
        5. Return response

        Child classes must implement the hook methods to enable rerun functionality:
        - _get_rerun_user_ids()
        - _build_rerun_request_params()
        - _create_run_request_for_item()
        - _create_rerun_response()

        Args:
            request: The rerun request object

        Returns:
            A response object with success status, message, and count
        """
        state_manager = self._create_state_manager()

        try:
            # 1. Check for existing in-progress operation
            error = state_manager.check_in_progress()
            if error:
                return self._create_rerun_response(False, error, 0)

            # 2. Get user IDs to process
            user_ids = self._get_rerun_user_ids(request)
            if not user_ids:
                return self._create_rerun_response(
                    False, "No interactions found matching the specified filters", 0
                )

            # 3. Pre-process hook (e.g., delete existing pending items)
            self._pre_process_rerun(request)

            # 4. Run batch with progress tracking
            users_processed, total_generated = self._run_batch_with_progress(
                user_ids=user_ids,
                request=request,
                request_params=self._build_rerun_request_params(request),
                state_manager=state_manager,
            )

            msg = f"Completed for {users_processed} user(s)"
            return self._create_rerun_response(True, msg, total_generated)

        except Exception as e:
            state_manager.mark_progress_failed(str(e))
            return self._create_rerun_response(
                False,
                f"Failed to run {self._get_base_service_name()}: {str(e)}",
                0,
            )
