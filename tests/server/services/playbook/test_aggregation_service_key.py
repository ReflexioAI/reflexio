from reflexio.server.extensions import get_service, register_service, reset_services
from reflexio.server.services.playbook.aggregation_prompt_processing import (
    AGGREGATION_PROMPT_PROCESSOR,
)


def test_unset_is_none_then_resolves_after_register() -> None:
    reset_services()
    assert get_service(AGGREGATION_PROMPT_PROCESSOR) is None
    sentinel = object()
    register_service(AGGREGATION_PROMPT_PROCESSOR, sentinel)  # type: ignore[arg-type]
    assert get_service(AGGREGATION_PROMPT_PROCESSOR) is sentinel
