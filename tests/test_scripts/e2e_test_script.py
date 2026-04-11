import logging

from reflexio.lib.reflexio_lib import InteractionData, Reflexio
from reflexio.models.api_schema.retriever_schema import (
    GetInteractionsRequest,
    GetUserProfilesRequest,
)
from reflexio.models.api_schema.service_schemas import UserActionType
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigTest,
)
from reflexio.server import OPENAI_API_KEY

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)

import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

user_id = "your_user_id2"


def sample_publish_interaction(am: Reflexio):
    am.publish_interaction(
        request_id="your_request_id",  # for attribution for profile sources
        user_id=user_id,
        interaction_requests=[
            InteractionData(
                role="User",  # agent, student or other that you can define to match your prompt
                content="Hey, this is Kiera Cooper. I am calling with Fundera. How have you been?",
                user_action=UserActionType.NONE,  # Optional, defaults to NONE
                user_action_description="",  # Optional, defaults to empty string
                interacted_image_url="",  # Optional, defaults to empty string
            )
        ],
        source="conversation",  # optional, mark interaction if needed. can be used for filtering when search user profiles
    )


def sample_publish_playbook(am: Reflexio):
    am.publish_interaction(
        request_id="your_request_id",  # for attribution for profile sources
        user_id=user_id,
        agent_version="1.0.0",
        interaction_requests=[
            InteractionData(
                role="User",  # agent, student or other that you can define to match your prompt
                content="I don't like the way you talked to me",
                user_action=UserActionType.NONE,  # Optional, defaults to NONE
                user_action_description="",  # Optional, defaults to empty string
                interacted_image_url="",  # Optional, defaults to empty string
            )
        ],
        source="conversation",  # optional, mark interaction if needed. can be used for filtering when search user profiles
    )


if __name__ == "__main__":
    agentic_mem = Reflexio("3", "/Users/yilu/repos/reflexio/data")

    config = Config(
        storage_config=agentic_mem.request_context.configurator.config.storage_config,
        storage_config_test=StorageConfigTest.SUCCEEDED,
        agent_context_prompt="this is a sales call between two people",
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extraction_definition_prompt="name of the person, intent of the conversation, and the topic of the conversation",
                context_prompt="this is a sales call between two people",
            )
        ],
        user_playbook_extractor_configs=None,
    )
    agentic_mem.request_context.configurator.set_config(config)
    print(
        f"agentic_mem.request_context.configurator.config: {agentic_mem.request_context.configurator.config}"
    )

    # something that contains user profile
    sample_publish_interaction(agentic_mem)

    # something that contains user playbook content
    sample_publish_playbook(agentic_mem)

    interactions = agentic_mem.get_interactions(
        request=GetInteractionsRequest(
            user_id=user_id,
        )
    )
    print(f"interactions: {interactions}")

    profiles = agentic_mem.get_profiles(
        request=GetUserProfilesRequest(
            user_id=user_id,
        )
    )
    print(f"profiles: {profiles}")
