"""One cadence owns new automatic admission, including forced publications."""

from types import SimpleNamespace

import pytest

from reflexio.models.api_schema.domain.entities import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.models.config_schema import Config
from reflexio.models.extraction_window import get_extractor_window_params
from reflexio.server.services.base_generation._config_filter import ConfigFilterMixin
from reflexio.server.services.durable_learning.admission import extraction_admission

pytestmark = pytest.mark.unit


def test_deferred_execution_exports_keep_the_existing_worker_identities():
    from reflexio.server.services import durable_learning
    from reflexio.server.services.durable_learning import scheduler, worker

    assert durable_learning.DurableLearningWorker is worker.DurableLearningWorker
    assert (
        durable_learning.DurableLearningScheduler is scheduler.DurableLearningScheduler
    )
    assert (
        durable_learning.maybe_start_durable_learning
        is scheduler.maybe_start_durable_learning
    )


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("stalled", [False, True])
def test_zero_cadence_blocks_new_profile_work_even_under_force_or_finalization(
    force, finished, stalled
):
    config = Config(storage_config=None, profile_stride_size=0, profile_window_size=23)
    request = PublishUserInteractionRequest(
        user_id="synthetic",
        session_id="synthetic",
        interaction_data_list=[InteractionData(content="Synthetic profile admission")],
        force_extraction=force,
        session_finished=finished,
    )
    policy = extraction_admission(config, request, stalled=stalled)
    assert policy["profile"] == {"eligible": False}
    assert get_extractor_window_params(config, "profile").window_size == 23
    assert policy["playbook"]["eligible"] is (not stalled)


class _ProfileFilter(ConfigFilterMixin):
    def __init__(self, *, stride, automatic, force):
        self.root_config = Config(storage_config=None, profile_stride_size=stride)
        self.service_config = SimpleNamespace(
            auto_run=automatic, force_extraction=force
        )

    def _get_extractor_state_service_name(self):
        return "profile_extractor"

    def _get_extraction_window(self):
        return get_extractor_window_params(self.root_config, "profile")

    @property
    def storage(self):
        raise AssertionError("Eligibility must be settled before reading storage")


@pytest.mark.parametrize("force", [False, True])
def test_direct_automatic_service_zero_gate_precedes_force_and_storage(force):
    service = _ProfileFilter(stride=0, automatic=True, force=force)
    assert (
        service._filter_config_by_stride(service.root_config.profile_extractor_config)
        is None
    )


@pytest.mark.parametrize("stride", [0, 3])
@pytest.mark.parametrize("force", [False, True])
def test_manual_service_keeps_config_and_never_consumes_automatic_stride(stride, force):
    service = _ProfileFilter(stride=stride, automatic=False, force=force)
    extractor = service.root_config.profile_extractor_config
    assert service._filter_config_by_stride(extractor) is extractor


def test_positive_forced_cadence_keeps_existing_force_behavior():
    service = _ProfileFilter(stride=3, automatic=True, force=True)
    extractor = service.root_config.profile_extractor_config
    assert service._filter_config_by_stride(extractor) is extractor


@pytest.mark.parametrize("manual", [False, True, None])
def test_current_runtime_refuses_retired_switch_instead_of_ignoring_it(manual):
    with pytest.raises(ValueError, match="manual_trigger"):
        Config.model_validate(
            {
                "storage_config": None,
                "profile_stride_size": 0,
                "profile_extractor_config": {
                    "extraction_definition_prompt": "Synthetic profile policy",
                    "manual_trigger": manual,
                },
            }
        )
