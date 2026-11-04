from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from reflexio.models.config_schema import Config
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.site_var.site_var_manager import SiteVarManager

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
    """Whether a generation model can actually be RESOLVED for this org.

    ASK THE QUESTION GENERATION ASKS. This gate exists so tuning is not switched
    on where no model could run it, and the only honest test of that is the
    resolver the pipeline itself uses. ``resolve_model_name`` has four branches
    in priority order -- org override, the embedding special case, the
    ``llm_model_setting`` site var, then AUTO-DETECT FROM AVAILABLE PROVIDER
    KEYS -- and an earlier version of this function reimplemented only the
    first and third.

    Missing the auto-detect branch is not a corner case; it can be the only
    branch a deployment resolves through. A shipped ``llm_model_setting`` may
    carry an empty ``default_generation_model_name`` while an organization has
    no ``llm_config`` at all and still generates through its provider keys.
    Both of the old branches were false for such an org, so
    ``sanitize_platform_offline_tuner_config`` silently flipped ``enabled``
    back to False and tuning could not be switched on by anyone.

    That failure was invisible from the outside: the save returns 200, the
    capability assert has already passed, and the only signal is the echoed
    ``enabled: false``. Reimplementing a resolution chain rather than calling it
    is what allowed the two copies to disagree, so this now calls it.

    ``RuntimeError`` is the resolver's documented "no API keys are available and
    no override is set", which is exactly this function's False -- and it is
    caught narrowly so an unexpected failure still surfaces rather than being
    read as a definitive "no model".

    Args:
        config (Config): The organization's shared configuration.

    Returns:
        bool: True when a generation model resolves for this organization.
    """
    model_setting = SiteVarManager().get_site_var("llm_model_setting")
    raw_site_var = (
        model_setting.get("default_generation_model_name")
        if isinstance(model_setting, dict)
        else None
    )
    # NORMALIZE EXACTLY AS THE EXECUTION PATH DOES. `get_site_var` returns
    # parsed JSON, so a malformed `llm_model_setting` can carry any type here.
    # `open_world_generation_client` drops every non-string value before
    # resolving, and a truthy non-string (a JSON number, say) is returned
    # verbatim by `resolve_model_name`'s site-var branch -- so without this the
    # gate would admit `True` for a value the run then refuses to resolve,
    # recreating the admit/execute disagreement this function exists to end.
    site_var_value = raw_site_var if isinstance(raw_site_var, str) else None
    config_override = (
        config.llm_config.generation_model_name
        if config.llm_config is not None
        else None
    )
    try:
        return bool(
            resolve_model_name(
                ModelRole.GENERATION,
                site_var_value=site_var_value,
                config_override=config_override,
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
