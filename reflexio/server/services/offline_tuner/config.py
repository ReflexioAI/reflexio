from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from reflexio.models.config_schema import Config
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name

# The level a tuner PROBLEM line emits at -- a failed attempt, a conflict, a
# bootstrap that did not reconcile, a refusal. WARNING is deliberate, and it is
# ONLY for those.
#
# ROUTINE LINES DO NOT USE IT. The tick-outcome line fires on every sweep tick
# and most ticks are an ``ineligible_*`` answer nobody should act on. Routine
# tuner lines (tick outcomes, attempt answers, bootstrap/bridge counts,
# retry-eligible notes) go to the operational channel instead --
# ``get_ops_logger("offline_tuner")`` at INFO -- which ``configure_logging``
# pins to INFO in every profile. Which outcomes count as a problem is
# ``sweep_progress.is_problem_outcome``.
#
# Why a problem line cannot simply be INFO on its module logger: the
# production logging profile sets the root logger to WARNING (capped by
# ``REFLEXIO_LOG_LEVEL``) with one INFO stream handler on stdout, so an INFO
# line on a module logger is silenced by the root LEVEL. For a deployment whose
# operator can read only the process log, a line that is not in the log did
# not happen. Raising the HTTP server's ``--log-level`` does not fix this: it
# configures the server's own loggers and touches neither the root logger nor
# this one. So the level travels on the record instead. Problem lines are
# low-volume, so WARNING costs almost nothing.
TUNER_OUTCOME_LOG_LEVEL = logging.WARNING


class OfflineTunerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


def generation_model_available(config: Config) -> bool:
    """Check whether the same generation role used by the tuner can resolve."""
    try:
        return bool(
            resolve_model_name(
                ModelRole.GENERATION,
                llm_config=config.llm_config,
                api_key_config=config.api_key_config,
            )
        )
    except RuntimeError:
        return False


def sanitize_platform_offline_tuner_config(
    offline_tuner_config: OfflineTunerConfig,
    *,
    shared_config: Config,
) -> OfflineTunerConfig:
    if not offline_tuner_config.enabled:
        return offline_tuner_config
    if generation_model_available(shared_config):
        return offline_tuner_config

    return offline_tuner_config.model_copy(update={"enabled": False})
