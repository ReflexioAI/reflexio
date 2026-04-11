"""Export YAML scenarios to JSON files publishable via the Reflexio CLI.

Usage:
    # Export all e2e scenarios
    python tests/test_data/scenarios/export_json.py --tier e2e

    # Export all eval scenarios
    python tests/test_data/scenarios/export_json.py --tier eval

    # Export a specific scenario
    python tests/test_data/scenarios/export_json.py --file tests/test_data/scenarios/e2e/customer_support.yaml

    # Custom output directory
    python tests/test_data/scenarios/export_json.py --tier e2e --output /tmp/scenarios
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from yaml_loader import (
    build_interactions,
    get_user_conversations,
    load_scenario,
    make_user_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
E2E_DIR = _THIS_DIR / "e2e"
EVAL_DIR = _THIS_DIR / "eval"
DEFAULT_OUTPUT = _THIS_DIR / "output"


def export_scenario(scenario_path: Path, output_dir: Path) -> list[Path]:
    """Export a single YAML scenario to one JSON file per user.

    Args:
        scenario_path (Path): Path to the YAML scenario file
        output_dir (Path): Directory to write JSON files into

    Returns:
        list[Path]: Paths to the written JSON files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario(scenario_path)
    scenario_id = scenario.get("id", scenario_path.stem)
    participants = scenario.get("participants", {})
    paths: list[Path] = []

    for user_key, conv in get_user_conversations(scenario).items():
        interactions = build_interactions(conv, participants)
        user_id = make_user_id(scenario_id, user_key)

        payload = {
            "user_id": user_id,
            "source": conv.get("source", scenario_id),
            "agent_version": conv.get("agent_version", "v1.0"),
            "session_id": conv.get("session_id", f"{scenario_id}-{user_key}"),
            "interactions": [
                {"role": i.role, "content": i.content}
                | (
                    {
                        "tools_used": [
                            {"tool_name": t.tool_name, "tool_data": t.tool_data}
                            for t in i.tools_used
                        ]
                    }
                    if i.tools_used
                    else {}
                )
                for i in interactions
            ],
        }

        file_path = output_dir / f"{scenario_id}_{user_key}.json"
        file_path.write_text(json.dumps(payload, indent=2) + "\n")
        paths.append(file_path)
        logger.info("  %s: %d turns -> %s", user_key, len(interactions), file_path.name)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YAML scenarios to JSON files for the Reflexio CLI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tier",
        choices=["e2e", "eval", "all"],
        help="Export all scenarios in a tier",
    )
    group.add_argument(
        "--file",
        type=Path,
        help="Export a single YAML scenario file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if args.file:
        if not args.file.exists():
            logger.error("File not found: %s", args.file)
            return
        logger.info("Exporting %s", args.file.name)
        export_scenario(args.file, args.output)
    else:
        tiers = ["e2e", "eval"] if args.tier == "all" else [args.tier]
        for tier in tiers:
            tier_dir = E2E_DIR if tier == "e2e" else EVAL_DIR
            scenarios = sorted(tier_dir.glob("*.yaml"))
            logger.info("Exporting %d %s scenario(s)", len(scenarios), tier)
            for path in scenarios:
                export_scenario(path, args.output / tier)

    logger.info("Done. Publish with: reflexio publish --file <json_file> --wait")


if __name__ == "__main__":
    main()
