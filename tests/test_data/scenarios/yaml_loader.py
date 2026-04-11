"""Load YAML conversation scenarios and convert to Reflexio InteractionData.

Provides a thin conversion layer between the human-readable YAML format
(used for hand-authored test conversations) and the InteractionData schema
expected by the Reflexio publish API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from reflexio import InteractionData, ToolUsed

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent


def load_scenario(path: Path) -> dict[str, Any]:
    """Parse a YAML scenario file.

    Args:
        path (Path): Path to the YAML scenario file

    Returns:
        dict[str, Any]: Parsed scenario data with participants, tools, and conversations
    """
    return yaml.safe_load(path.read_text())


def build_interactions(
    conversation: dict[str, Any],
    participants: dict[str, dict[str, str]],
) -> list[InteractionData]:
    """Convert a conversation block from YAML into InteractionData objects.

    Maps speaker keys to their participant role and converts tool_calls
    into ToolUsed objects compatible with the Reflexio API.

    Args:
        conversation (dict[str, Any]): A single user's conversation block containing 'turns'
        participants (dict[str, dict[str, str]]): Participant definitions mapping key to role/display_name

    Returns:
        list[InteractionData]: Interaction objects ready for publish_interaction()
    """
    interactions: list[InteractionData] = []

    for turn in conversation.get("turns", []):
        speaker_key = turn["speaker"]
        participant = participants.get(speaker_key, {})
        role = participant.get("role", "User")
        content = turn.get("content", "")

        tool_calls = turn.get("tool_calls")
        if tool_calls:
            tools_used = [
                ToolUsed(
                    tool_name=tc["name"],
                    tool_data={
                        "input": tc.get("input", {}),
                        "output": (
                            json.dumps(tc["output"])
                            if isinstance(tc.get("output"), dict)
                            else str(tc.get("output", ""))
                        ),
                    },
                )
                for tc in tool_calls
            ]
            interactions.append(
                InteractionData(role=role, content=content, tools_used=tools_used)
            )
        else:
            interactions.append(InteractionData(role=role, content=content))

    return interactions


def get_user_conversations(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract user conversations from a scenario, excluding the agent.

    Args:
        scenario (dict[str, Any]): Parsed scenario data

    Returns:
        dict[str, dict[str, Any]]: Mapping of user key to their conversation block
    """
    participants = scenario.get("participants", {})
    conversations = scenario.get("conversations", {})
    return {
        key: conv
        for key, conv in conversations.items()
        if participants.get(key, {}).get("role") != "Assistant"
    }


def make_user_id(scenario_id: str, user_key: str) -> str:
    """Generate a deterministic user_id from scenario ID and user key.

    Args:
        scenario_id (str): The scenario identifier (e.g., 'customer-support')
        user_key (str): The user's key in the participants map (e.g., 'priya')

    Returns:
        str: A user ID like 'multiuser-customer-support-priya'
    """
    return f"multiuser-{scenario_id}-{user_key}"


def scenario_to_jsonl(scenario: dict[str, Any], output_dir: Path) -> list[Path]:
    """Export a scenario's conversations to JSONL files for debugging.

    Args:
        scenario (dict[str, Any]): Parsed scenario data
        output_dir (Path): Directory to write JSONL files into

    Returns:
        list[Path]: Paths to the written JSONL files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    participants = scenario.get("participants", {})
    scenario_id = scenario.get("id", "unknown")
    paths: list[Path] = []

    for user_key, conv in get_user_conversations(scenario).items():
        file_path = output_dir / f"{scenario_id}_{user_key}.jsonl"
        lines: list[str] = []
        for i, turn in enumerate(conv.get("turns", []), start=1):
            speaker_key = turn["speaker"]
            participant = participants.get(speaker_key, {})
            role_raw = participant.get("role", "User")
            role = "customer" if role_raw == "User" else "agent"

            entry: dict[str, Any] = {
                "turn": i,
                "role": role,
                "content": turn.get("content", ""),
                "labels": [],
            }

            if tool_calls := turn.get("tool_calls"):
                entry["tool_interactions"] = [
                    {
                        "tool_call_id": f"call_{scenario_id}_{user_key}_{i}_{j}",
                        "function_name": tc["name"],
                        "arguments": tc.get("input", {}),
                        "result": tc.get("output", {}),
                    }
                    for j, tc in enumerate(tool_calls)
                ]

            lines.append(json.dumps(entry))

        file_path.write_text("\n".join(lines) + "\n")
        paths.append(file_path)
        logger.info("Exported %s (%d turns)", file_path.name, len(lines))

    return paths
