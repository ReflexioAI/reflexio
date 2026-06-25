"""Reflection mixin: post-publish critique-and-revise of cited memories."""

from __future__ import annotations

from reflexio.lib._base import STORAGE_NOT_CONFIGURED_MSG, ReflexioBase
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionResult,
    ReflectionServiceRequest,
)


class ReflectionMixin(ReflexioBase):
    """Expose ``run_reflection`` on the Reflexio facade."""

    def run_reflection(
        self,
        request: ReflectionServiceRequest | dict,
    ) -> ReflectionResult:
        """Run one reflection pass for the given request.

        Best-effort: returns a ``ReflectionResult`` describing what
        happened. Storage must be configured; raises otherwise so the
        caller (the publish-time hook) can swallow it via its own
        ``try/except`` wrapper.

        Args:
            request (ReflectionServiceRequest | dict): The reflection
                request. Dicts are coerced.

        Returns:
            ReflectionResult: Counts of cited / considered / replaced /
                skipped / failed decisions for logging and tests.

        Raises:
            ValueError: If storage is not configured.
        """
        if not self._is_storage_configured():
            raise ValueError(STORAGE_NOT_CONFIGURED_MSG)
        if isinstance(request, dict):
            request = ReflectionServiceRequest(**request)

        # Local import keeps the heavy service module out of facade
        # cold-start when reflection is disabled.
        from reflexio.server.services.reflection.service import (
            ReflectionService,
        )

        service = ReflectionService(
            request_context=self.request_context,
            llm_client=self.llm_client,
        )
        return service.run(request)
