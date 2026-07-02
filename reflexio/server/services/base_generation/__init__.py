"""Sub-mixins composed into ``BaseGenerationService`` (Tier-1b decomposition).

Each mixin holds a cohesive bucket of the ``BaseGenerationService`` behaviour and
is composed into the base via MRO in
``reflexio.server.services.base_generation_service``. The residual base keeps the
ABC skeleton (all abstract methods), ``__init__`` (per-run field inits), the ``run``
FIFO drain, ``_run_generation`` billing emit-ordering, and the lock/queue helpers.
"""

from reflexio.server.services.base_generation._config_filter import (
    ConfigFilterMixin,
)
from reflexio.server.services.base_generation._extraction_lifecycle import (
    ExtractionRunLifecycleMixin,
)
from reflexio.server.services.base_generation._should_run import (
    ShouldRunPrecheckMixin,
)
from reflexio.server.services.base_generation._status_change import (
    StatusChangeMixin,
    StatusChangeOperation,
)
from reflexio.server.services.base_generation._usage_billing import (
    UsageBillingMixin,
)

__all__ = [
    "ConfigFilterMixin",
    "ExtractionRunLifecycleMixin",
    "ShouldRunPrecheckMixin",
    "StatusChangeMixin",
    "StatusChangeOperation",
    "UsageBillingMixin",
]
