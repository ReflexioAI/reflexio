from dataclasses import dataclass


@dataclass(frozen=True)
class PlaybookServiceConstants:
    PLAYBOOK_EXTRACTORS_CONFIG_NAME = "user_playbook_extractor_configs"
    # ===============================
    # prompt ids
    # ===============================
    PLAYBOOK_SHOULD_GENERATE_PROMPT_ID = "playbook_should_generate"
    PLAYBOOK_EXTRACTION_CONTEXT_PROMPT_ID = "playbook_extraction_context"
    PLAYBOOK_EXTRACTION_PROMPT_ID = "playbook_extraction_main"
    PLAYBOOK_EXTRACTION_CONTEXT_INCREMENTAL_PROMPT_ID = (
        "playbook_extraction_context_incremental"
    )
    PLAYBOOK_EXTRACTION_INCREMENTAL_PROMPT_ID = "playbook_extraction_main_incremental"
    PLAYBOOK_AGGREGATION_PROMPT_ID = "playbook_aggregation"

    # ===============================
    # expert content prompt ids
    # ===============================
    PLAYBOOK_SHOULD_GENERATE_EXPERT_PROMPT_ID = "playbook_should_generate_expert"
    PLAYBOOK_EXTRACTION_CONTEXT_EXPERT_PROMPT_ID = "playbook_extraction_context_expert"
    PLAYBOOK_EXTRACTION_EXPERT_PROMPT_ID = "playbook_extraction_main_expert"

    # ===============================
    # agent success evaluation prompt ids
    # ===============================
    AGENT_SUCCESS_EVALUATION_SHOULD_EVALUATE_PROMPT_ID = (
        "agent_success_evaluation_should_evaluate"
    )
    AGENT_SUCCESS_EVALUATION_PROMPT_ID = "agent_success_evaluation"
