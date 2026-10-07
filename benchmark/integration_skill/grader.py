"""Independent behavioral checks, run against the completed application."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reflexio import ReflexioClient
from reflexio.models.api_schema.domain.entities import PublishUserInteractionRequest


def search_result(seed: int, state: str = "populated") -> dict[str, Any]:
    base = {"agent_version": "support@1", "request_id": "seed-request"}
    data = {
        "success": state != "failure",
        "profiles": [
            {
                "profile_id": f"profile-{seed}",
                "user_id": "u",
                "last_modified_timestamp": 1,
                "generated_from_request_id": "seed-request",
                "content": f"profile-content-{seed}",
            }
        ],
        "user_playbooks": [
            {**base, "user_playbook_id": seed, "content": f"user-content-{seed}"},
            {**base, "user_playbook_id": seed + 1000, "content": "[discard]irrelevant"},
        ],
        "agent_playbooks": [
            {**base, "agent_playbook_id": seed, "content": f"agent-content-{seed}"}
        ],
    }
    if state == "empty":
        for key in ("profiles", "user_playbooks", "agent_playbooks"):
            data[key] = []
    return data


class Capture:
    def __init__(self, barrier: threading.Barrier | None = None):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self.barrier = barrier

    def start(self, seed: int, state: str, identities: dict) -> None:
        self.local.record = {
            "seed": seed,
            "state": state,
            "identities": identities.copy(),
            "search": [],
            "publish": [],
            "model": [],
            "events": [],
        }
        with self.lock:
            self.records.append(self.local.record)

    def post(self, path: str, *, json: dict, timeout: float = 5) -> dict:  # noqa: ARG002
        record = self.local.record
        if path == "/api/search":
            record["events"].append("search")
            record["search"].append(json)
            if record["state"] == "exception":
                raise TimeoutError("synthetic search timeout")
            return search_result(record["seed"], record["state"])
        if path != "/api/publish_interaction":
            raise AssertionError(f"unexpected endpoint: {path}")
        record["events"].append("publish")
        record["publish"].append(json)
        return {
            "success": True,
            "request_id": f"request-{record['seed']}",
            "warnings": [],
        }

    def generate(self, message: str, context: str) -> str:
        self.local.record["events"].append("model")
        self.local.record["model"].append((message, context))
        if self.barrier:
            self.barrier.wait(timeout=10)
        return f"response:{message}"

    def diagnostic(self, message: str) -> None:
        pass


def client_for(kind: str, capture: Capture) -> Any:
    if kind == "http_app":
        return capture
    client = ReflexioClient(api_key="fixture-only", url_endpoint="http://localhost:1")
    client._make_request = lambda method, endpoint, headers=None, **kw: capture.post(
        endpoint, json=kw.get("json", {})
    )
    return client


def check_record(record: dict) -> None:
    seed, state = record["seed"], record["state"]
    assert record["events"] == ["search", "model", "publish"], (
        "lifecycle/duplicate calls"
    )
    message, context = record["model"][0]
    assert message == f"message-{seed}"
    expected = []
    if state == "populated":
        for marker in (
            f"profile-content-{seed}",
            f"user-content-{seed}",
            f"agent-content-{seed}",
        ):
            assert marker in context, f"required context lost: {marker}"
        actual_markers = set(re.findall(r"(?:profile|user|agent)-content-\d+", context))
        expected_markers = {
            f"{kind}-content-{seed}" for kind in ("profile", "user", "agent")
        }
        assert actual_markers == expected_markers, "foreign learning context injected"
        assert "[discard]" not in context, "discarded candidate injected"
        expected = [
            ("profile", f"profile-{seed}"),
            ("user_playbook", str(seed)),
            ("agent_playbook", str(seed)),
        ]
    else:
        assert not context, "context leaked into empty/failed search"
    body = record["publish"][0]
    search = record["search"][0]
    for key in ("user_id", "session_id", "source", "agent_version"):
        assert body[key] == search[key] == record["identities"][key], (
            f"identity mismatch: {key}"
        )
    request = PublishUserInteractionRequest.model_validate(body)
    assert len(request.interaction_data_list) == 2, "changed interaction count"
    user, agent = request.interaction_data_list
    assert user.content == message and agent.content == f"response:{message}"
    assert agent.role in ("Agent", "Assistant")
    assert not user.retrieved_learnings, "references attached to user interaction"
    raw = body["interaction_data_list"][1].get("retrieved_learnings", [])
    assert all(isinstance(item["learning_id"], str) for item in raw), (
        "IDs must be strings"
    )
    refs = [(ref.kind, ref.learning_id) for ref in agent.retrieved_learnings]
    assert sorted(refs) == sorted(expected), (
        f"references: expected {expected}, got {refs}"
    )


def grade(workspace: Path, kind: str) -> dict:
    sys.path.insert(0, str(workspace))
    spec = importlib.util.spec_from_file_location(
        "graded_application", workspace / "app.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = {}
    for scenario in (
        "populated",
        "empty",
        "failure",
        "exception",
        "consecutive",
        "concurrent",
    ):
        try:
            capture = Capture(
                threading.Barrier(2) if scenario == "concurrent" else None
            )
            client = client_for(kind, capture)

            def turn(
                seed: int,
                state: str = "populated",
                capture=capture,
                client=client,
                scenario=scenario,
            ) -> None:
                identities = {
                    "user_id": f"benchmark-user-{scenario}",
                    "session_id": f"benchmark-session-{scenario}",
                    "source": f"benchmark-{scenario}",
                    "agent_version": f"support@{scenario}",
                }
                capture.start(seed, state, identities)
                response = module.handle_turn(
                    client,
                    capture,
                    user_message=f"message-{seed}",
                    **identities,
                )
                assert response == f"response:message-{seed}", "user response changed"

            if scenario == "concurrent":
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(turn, seed) for seed in (117, 218)]
                    for future in futures:
                        future.result(timeout=15)
            elif scenario == "consecutive":
                turn(117)
                turn(218)
                turn(319, "empty")
            else:
                turn(117, scenario)
            for record in capture.records:
                check_record(record)
            results[scenario] = {"passed": True}
        except Exception as exc:
            results[scenario] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "passed": all(item["passed"] for item in results.values()),
        "checks": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("kind", choices=("python_app", "http_app"))
    args = parser.parse_args()
    try:
        result = grade(args.workspace, args.kind)
    except Exception as exc:
        result = {
            "passed": False,
            "checks": {
                "application_load": {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            },
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
