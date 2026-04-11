"""
Utils for service layer
"""

import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from reflexio.cli.log_format import LLM_IO_LOG_FILE, next_llm_entry_id
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    UserActionType,
)
from reflexio.server import LLM_PROMPT_LEVEL
from reflexio.server.prompt.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

# Custom log level for model responses (between INFO=20 and WARNING=30)
# Already registered in server/__init__.py; import the numeric constant only.
MODEL_RESPONSE_LEVEL = 25


def log_model_response(
    target_logger: logging.Logger, label: str, response: Any
) -> None:
    """
    Log an LLM model response. Full response goes to the log file at LLM_PROMPT level.
    A one-line summary goes to the console at MODEL_RESPONSE level.

    Args:
        target_logger (logging.Logger): The logger instance to use
        label (str): Descriptive label for the response (e.g. "Profile updates model response")
        response (Any): The model response to log
    """
    entry_id = next_llm_entry_id()
    # Full response to llm_io.log only (level 15 < INFO 20, so console ignores it)
    target_logger.log(
        LLM_PROMPT_LEVEL,
        "[#%d] %s: %s",
        entry_id,
        label,
        response,
        extra={"entry_id": entry_id, "label": label},
    )
    # One-line summary to console
    response_type = type(response).__name__
    target_logger.log(
        MODEL_RESPONSE_LEVEL,
        "[MODEL] %s: %s — %s [#%d]",
        label,
        response_type,
        LLM_IO_LOG_FILE,
        entry_id,
    )


def log_llm_messages(
    target_logger: logging.Logger,
    label: str,
    messages: list[dict[str, Any]],
) -> None:
    """
    Log LLM prompt messages. Full content goes to the log file at LLM_PROMPT level.
    A one-line summary goes to the console at INFO level.

    Args:
        target_logger (logging.Logger): The logger instance to use
        label (str): Descriptive label (e.g. "Profile extraction")
        messages (list[dict[str, Any]]): The LLM messages to log
    """
    entry_id = next_llm_entry_id()
    # Full messages to llm_io.log only
    formatted = format_messages_for_logging(messages)
    target_logger.log(
        LLM_PROMPT_LEVEL,
        "%s messages:\n%s",
        label,
        formatted,
        extra={"entry_id": entry_id, "label": label},
    )
    # Summary to console
    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
    target_logger.info(
        "[LLM] %s: %d msgs, ~%d chars — %s [#%d]",
        label,
        len(messages),
        total_chars,
        LLM_IO_LOG_FILE,
        entry_id,
    )


@dataclass
class PromptConfig:
    """Configuration for a prompt to be rendered.

    Attributes:
        prompt_id: The ID of the prompt template to render
        variables: Dictionary of variables to pass to the prompt template
    """

    prompt_id: str
    variables: dict[str, any]  # type: ignore[reportGeneralTypeIssues]


@dataclass
class MessageConstructionConfig:
    """Configuration for constructing LLM messages from interactions.

    Attributes:
        prompt_manager: The prompt manager to use for rendering prompts
        system_prompt_config: Configuration for the system message (before interactions)
        user_prompt_config: Configuration for the user message (after interactions)
    """

    prompt_manager: PromptManager
    system_prompt_config: PromptConfig | None = None
    user_prompt_config: PromptConfig | None = None


def format_interactions_to_history_string(interactions: list[Interaction]) -> str:
    """
    Format a list of interactions into a single string representing the interaction history.

    Each interaction is formatted as:
    - Text content: "{role}: {content}"
    - User actions: "{role}: {action} {action_description}"

    Args:
        interactions (list[Interaction]): List of interactions to format

    Returns:
        str: A formatted string representing the interaction history, with interactions separated by newlines.
             Returns empty string if no interactions are provided.

    Example:
        >>> interactions = [
        ...     Interaction(role="user", content="I love sushi", user_action=UserActionType.NONE),
        ...     Interaction(role="user", content="", user_action=UserActionType.CLICK, user_action_description="menu item")
        ... ]
        >>> result = format_interactions_to_history_string(interactions)
        >>> print(result)
        user: I love sushi
        user: click menu item
    """
    formatted_interactions = []
    for interaction in interactions:
        # Add text content with tools_used prefix if present
        if interaction.content:
            if interaction.tools_used:
                tool_prefix = " ".join(
                    f"[used tool: {t.tool_name}({json.dumps(t.tool_data)})]"
                    for t in interaction.tools_used
                )
                formatted_interactions.append(
                    f"{interaction.role}: ```{tool_prefix} {interaction.content}```"
                )
            else:
                formatted_interactions.append(
                    f"{interaction.role}: ```{interaction.content}```"
                )

        # Add user action
        if interaction.user_action != UserActionType.NONE:
            formatted_interactions.append(
                f"{interaction.role}: ```{interaction.user_action.value} {interaction.user_action_description}```"
            )

    return "\n".join(formatted_interactions)


def format_sessions_to_history_string(
    sessions: list[RequestInteractionDataModel],
) -> str:
    """
    Format interactions grouped by session into a string.

    All RequestInteractionDataModel objects with the same session_id are consolidated
    under a single header. Within each consolidated group, interactions are ordered by
    their id in ascending order (smaller to bigger).

    Args:
        sessions (list[RequestInteractionDataModel]): List of request interaction data models to format

    Returns:
        str: A formatted string with interactions grouped by session.
             Returns empty string if no sessions are provided.

    Example:
        >>> # Given sessions with interactions (multiple requests in same session)
        >>> result = format_sessions_to_history_string(sessions)
        >>> print(result)
        === Session: session_1 ===
        user: Hello, I need help
        assistant: How can I assist you?
        user: Thanks for the help
        assistant: You're welcome!

        === Session: session_2 ===
        user: I love sushi
        assistant: That's great!
    """
    if not sessions:
        return ""

    # Group all RequestInteractionDataModel objects by their session_id
    grouped_by_name: dict[str, list[RequestInteractionDataModel]] = {}
    for request_interaction in sessions:
        if request_interaction.session_id not in grouped_by_name:
            grouped_by_name[request_interaction.session_id] = []
        grouped_by_name[request_interaction.session_id].append(request_interaction)

    # Sort each group's requests by created_at timestamp
    for group_name in grouped_by_name:
        grouped_by_name[group_name] = sorted(
            grouped_by_name[group_name], key=lambda g: g.request.created_at
        )

    # Sort group names by the earliest request timestamp in each group
    sorted_group_names = sorted(
        grouped_by_name.keys(),
        key=lambda name: grouped_by_name[name][0].request.created_at,
    )

    formatted_groups = []
    for group_name in sorted_group_names:
        # Format header with session name
        group_header = f"=== Session: {group_name} ==="

        # Combine all interactions from all requests in this session
        all_interactions = []
        for request_interaction in grouped_by_name[group_name]:
            all_interactions.extend(request_interaction.interactions)

        # Sort interactions by id ascending (smaller to bigger)
        all_interactions = sorted(all_interactions, key=lambda i: i.interaction_id)

        # Format combined interactions
        group_interactions = format_interactions_to_history_string(all_interactions)

        # Combine header and interactions
        formatted_groups.append(f"{group_header}\n{group_interactions}")

    return "\n\n".join(formatted_groups)


def extract_interactions_from_request_interaction_data_models(
    request_interaction_data_models: list[RequestInteractionDataModel],
) -> list[Interaction]:
    """
    Extract a flat list of interactions from request interaction groups.

    This is useful for backward compatibility with services that still expect
    a flat list of interactions.

    Args:
        request_interaction_data_models (list[RequestInteractionDataModel]): List of request interaction data models

    Returns:
        list[Interaction]: Flat list of all interactions from all groups
    """
    interactions = []
    for request_interaction_data_model in request_interaction_data_models:
        interactions.extend(request_interaction_data_model.interactions)
    return interactions


def construct_messages_from_interactions(
    interactions: list[Interaction],
    config: MessageConstructionConfig,
) -> list[dict]:
    """
    Construct a list of LLM messages from interactions with custom prompts.

    This function creates a structured message sequence:
    1. Optional system message (using system_prompt_config)
    2. Single user message containing:
       - Formatted interactions (text content and user actions)
       - User prompt (if configured)
       - Last interaction's image (if present)

    Args:
        interactions: List of interactions to convert into messages
        config: Configuration for message construction including prompt configs

    Returns:
        list[dict]: List of messages ready to be sent to the LLM API.
            Each message is a dict with 'role' and 'content' keys.

    Example:
        >>> system_config = PromptConfig(
        ...     prompt_id="profile_update_instruction_start",
        ...     variables={"agent_context_prompt": "...", "context_prompt": "..."}
        ... )
        >>> user_config = PromptConfig(
        ...     prompt_id="profile_update_main",
        ...     variables={"extraction_definition_prompt": "...", "existing_profiles": "..."}
        ... )
        >>> config = MessageConstructionConfig(
        ...     prompt_manager=prompt_manager,
        ...     system_prompt_config=system_config,
        ...     user_prompt_config=user_config
        ... )
        >>> messages = construct_messages_from_interactions(interactions, config)
    """
    messages = []

    # Add system message if configured
    if config.system_prompt_config:
        system_content = config.prompt_manager.render_prompt(
            config.system_prompt_config.prompt_id,
            config.system_prompt_config.variables,
        )
        messages.append({"role": "system", "content": system_content})

    # Build combined user message content
    combined_content = []

    # Add user prompt if configured
    if config.user_prompt_config:
        user_prompt_content = config.prompt_manager.render_prompt(
            config.user_prompt_config.prompt_id,
            config.user_prompt_config.variables,
        )
        combined_content.append({"type": "text", "text": user_prompt_content})

    # Add last interaction's image (if present)
    if interactions:
        last_interaction = interactions[-1]

        # Add image URL (priority over base64)
        if last_interaction.interacted_image_url:
            combined_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": last_interaction.interacted_image_url},
                }
            )
        # Add base64-encoded image if no URL
        elif last_interaction.image_encoding:
            combined_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{last_interaction.image_encoding}"
                    },
                }
            )

    # Add combined user message if there's any content
    if combined_content:
        # Flatten to plain string when all blocks are text-only (no images).
        # Content-block format combined with structured output (response_format)
        # causes significantly slower OpenAI API responses and timeouts.
        all_text = all(block.get("type") == "text" for block in combined_content)
        if all_text:
            content = "\n\n".join(block["text"] for block in combined_content)
        else:
            content = combined_content
        messages.append({"role": "user", "content": content})

    return messages


def extract_json_from_string(text: str) -> dict:
    """
    Extract JSON from a string, handling both JSON-style and Python-style booleans.

    This function attempts to extract JSON from text in the following order:
    1. From code blocks (```json...```)
    2. From content between first { and last }

    It also handles Python-style boolean values (True/False/None) by converting
    them to JSON-style (true/false/null) before parsing, and falls back to
    Python literal parsing for dict-like responses that use single quotes.

    Args:
        text (str): string to extract JSON from

    Returns:
        dict: JSON object, or empty dict if parsing fails
    """

    def normalize_json_string(json_str: str) -> str:
        """Convert Python-style syntax to JSON-friendly syntax.

        Handles:
        - Python booleans (True/False/None) -> JSON (true/false/null)
        """
        # Replace Python boolean/null values with JSON equivalents
        # Use word boundaries to avoid replacing parts of strings
        json_str = re.sub(r"\bTrue\b", "true", json_str)
        json_str = re.sub(r"\bFalse\b", "false", json_str)
        json_str = re.sub(r"\bNone\b", "null", json_str)

        return json_str  # noqa: RET504

    def fix_unescaped_inner_quotes(json_str: str) -> str:
        """
        Attempt to fix common cases where apostrophes are returned as double quotes.

        Many LLMs occasionally substitute `customer's` with `customer"s`, leaving the JSON
        invalid because the double quote isn't escaped. We treat double quotes that sit
        between two word characters as apostrophes.
        """
        return re.sub(r"(?<=\w)\"(?=\w)", "'", json_str)

    def parse_json_candidate(json_str: str) -> tuple[dict | None, str | None]:
        """
        Try to parse a JSON candidate string using multiple strategies.

        Strategies:
        1. Direct json.loads
        2. json.loads after normalizing Python syntax
        3. ast.literal_eval as a fallback for Python-style dicts
        """
        candidates = [json_str]
        last_error: str | None = None

        normalized_json_str = normalize_json_string(json_str)
        if normalized_json_str != json_str:
            candidates.append(normalized_json_str)

        repaired_json_str = fix_unescaped_inner_quotes(normalized_json_str)
        if repaired_json_str not in candidates:
            candidates.append(repaired_json_str)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed, None
            except json.JSONDecodeError as err:  # noqa: PERF203
                last_error = str(err)
                continue

        for candidate in candidates:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed, None
            except (ValueError, SyntaxError) as err:  # noqa: PERF203
                last_error = str(err)
                continue

        return None, last_error

    # Pattern to match content between triple backticks and json keyword
    # Allow optional whitespace around json keyword and content
    pattern = r"```json\s*(.*?)\s*```"

    # Find the match using regex
    match = re.search(pattern, text, re.DOTALL)

    if match:
        json_str = match.group(1)
        parsed_json, error_message = parse_json_candidate(json_str)
        if parsed_json is not None:
            return parsed_json
        if error_message:
            logger.warning("Failed to parse JSON from code block: %s", error_message)

    # Try to find JSON content between first { and last }
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx : end_idx + 1]
        parsed_json, error_message = parse_json_candidate(json_str)
        if parsed_json is not None:
            return parsed_json
        if error_message:
            logger.warning("Failed to parse JSON from braces: %s", error_message)

    return {}


def format_messages_for_logging(messages: list[dict[str, Any]]) -> str:
    """
    Format messages for logging with proper newlines in text content.

    Args:
        messages: List of message dictionaries with role and content

    Returns:
        str: Formatted string representation of messages with newlines preserved
    """
    formatted_parts = []
    for i, msg in enumerate(messages):
        formatted_parts.append(f"Message {i + 1}:")
        formatted_parts.append(f"  role: {msg.get('role', 'unknown')}")
        content = msg.get("content", "")

        if isinstance(content, str):
            # Simple string content - preserve newlines
            formatted_parts.append("  content:")
            # Indent each line of content
            formatted_parts.extend(f"    {line}" for line in content.split("\n"))
        elif isinstance(content, list):
            # Multimodal content (list of objects)
            formatted_parts.append("  content:")
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "unknown")
                    if item_type == "text":
                        text_content = item.get("text", "")
                        formatted_parts.append(f"    type: {item_type}")
                        formatted_parts.append("    text:")
                        # Indent each line of text
                        formatted_parts.extend(
                            f"      {line}" for line in text_content.split("\n")
                        )
                    else:
                        # For non-text content, use JSON representation
                        formatted_parts.append(f"    {json.dumps(item, indent=4)}")
                else:
                    formatted_parts.append(f"    {json.dumps(item, indent=4)}")
        else:
            # Fallback to JSON for other types
            formatted_parts.append(f"  content: {json.dumps(content, indent=4)}")

        formatted_parts.append("")  # Empty line between messages

    return "\n".join(formatted_parts)


# Example usage
if __name__ == "__main__":
    test_string = """'Based on the existing profiles (which are currently empty) and the new interaction indicating a preference for sushi, the update would involve adding a new profile that reflects this preference. Here's the JSON format for the updates:\n\n```json\n{\n    "add_profile": ["I like sushi"],\n    "delete_profile": []\n}\n```'"""

    result = extract_json_from_string(test_string)
    print("Extracted JSON:", result)
