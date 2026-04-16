"""
Evaluation runner for Reflexio-OpenClaw integration.

Usage:
    uv run python -m reflexio.integrations.openclaw.eval.runner

    # Or directly:
    uv run python open_source/reflexio/reflexio/integrations/openclaw/eval/runner.py

Starts a local Reflexio server (SQLite) in a temp directory, runs each scenario
from dataset.json, and reports pass/fail per scenario with timing.

Exit code 0 if all scenarios pass, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_DATASET_PATH = _THIS_DIR / "dataset.json"

# How long to wait for server readiness on startup
_SERVER_READY_TIMEOUT_S = 30
# Polling interval for wait_extraction
_POLL_INTERVAL_S = 2


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of a single step execution."""

    action: str
    passed: bool
    message: str = ""


@dataclass
class ScenarioResult:
    """Aggregate result for one scenario."""

    scenario_id: str
    category: str
    passed: bool
    elapsed_s: float
    steps: list[StepResult] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Find an available TCP port by binding to port 0 and releasing.

    Returns:
        int: An available port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, timeout_s: float = _SERVER_READY_TIMEOUT_S) -> bool:
    """Poll the server health endpoint until it responds or timeout expires.

    Args:
        base_url (str): Base URL of the server (e.g. "http://127.0.0.1:9123").
        timeout_s (float): Maximum seconds to wait.

    Returns:
        bool: True if server became ready within timeout, False otherwise.
    """
    import requests as _requests

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = _requests.get(f"{base_url}/health", timeout=2)  # noqa: S113
            if resp.ok:
                return True
        except _requests.RequestException:
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Helpers for server management
# ---------------------------------------------------------------------------


class _TmpDirProxy:
    """Thin wrapper that exposes a ``name`` attribute for an already-created directory.

    Used to hand an existing ``TemporaryDirectory`` context-manager path to
    ``EvalRunner._start_server`` without creating a second temp directory.

    Args:
        path (str): Absolute path to the temporary directory.
    """

    def __init__(self, path: str) -> None:
        self.name = path


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Orchestrates scenario execution against a local Reflexio server.

    Args:
        dataset_path (Path): Path to the JSON dataset file.
        scenario_ids (list[str] | None): If provided, only run these scenarios.
    """

    def __init__(
        self,
        dataset_path: Path = _DATASET_PATH,
        scenario_ids: list[str] | None = None,
    ) -> None:
        self._dataset_path = dataset_path
        self._scenario_ids = scenario_ids
        self._server_proc: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._base_url = ""
        self._port = 0
        self._tmp_dir: _TmpDirProxy | None = None
        self._client: Any = None
        # State shared across steps within a scenario
        self._last_search_results: Any = None
        self._last_exit_code: int | None = None

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        """Start a local Reflexio server with SQLite storage."""
        if self._tmp_dir is None:
            raise RuntimeError("_start_server called before _tmp_dir was set")
        data_dir = Path(self._tmp_dir.name)
        self._port = _find_free_port()
        self._base_url = f"http://127.0.0.1:{self._port}"

        env = {
            **os.environ,
            "REFLEXIO_STORAGE": "sqlite",
            "REFLEXIO_DATA_DIR": str(data_dir),
        }

        self._server_proc = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "uvicorn",
                "reflexio.server.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self._port),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not _wait_for_server(self._base_url):
            self._server_proc.terminate()
            raise RuntimeError(
                f"Server did not become ready within {_SERVER_READY_TIMEOUT_S}s"
            )

        # Re-create client pointing at the new server
        from reflexio import ReflexioClient

        self._client = ReflexioClient(url_endpoint=self._base_url, api_key="")

    def _stop_server(self) -> None:
        """Stop the running server process."""
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None

    def _server_is_running(self) -> bool:
        """Check whether the server is currently responding.

        Returns:
            bool: True if healthy, False otherwise.
        """
        return _wait_for_server(self._base_url, timeout_s=3)

    # ------------------------------------------------------------------
    # Action handlers — each returns (passed, message)
    # ------------------------------------------------------------------

    def _action_publish_interaction(self, params: dict) -> tuple[bool, str]:
        """Publish a conversation to the server.

        Args:
            params (dict): Step params with user_id, agent_version, interactions.

        Returns:
            tuple[bool, str]: (success, message)
        """
        user_id: str = params["user_id"]
        agent_version: str = params.get("agent_version", "openclaw-agent")
        raw_interactions: list[dict] = params["interactions"]

        from reflexio.models.api_schema.service_schemas import InteractionData

        interactions = [
            InteractionData(
                role=turn["role"],
                content=turn["content"],
            )
            for turn in raw_interactions
        ]

        resp = self._client.publish_interaction(
            user_id=user_id,
            interactions=interactions,
            agent_version=agent_version,
            wait_for_response=True,
            force_extraction=True,
        )
        return True, f"published → {resp.message}"

    def _action_wait_extraction(self, params: dict) -> tuple[bool, str]:
        """Poll until at least one user playbook appears or timeout.

        Args:
            params (dict): Step params with optional timeout_s.

        Returns:
            tuple[bool, str]: (success, message)
        """
        timeout_s: float = float(params.get("timeout_s", 60))
        user_id: str | None = params.get("user_id")
        deadline = time.monotonic() + timeout_s

        # We just need the server to have finished background work.
        # The publish_interaction call above used wait_for_response=True,
        # so extraction should already be complete — a short sleep guards
        # against any final async flush.
        while time.monotonic() < deadline:
            resp = self._client.get_user_playbooks(user_id=user_id)
            if resp.user_playbooks:
                return True, f"extraction done ({len(resp.user_playbooks)} playbooks)"
            time.sleep(_POLL_INTERVAL_S)

        return False, f"no playbooks appeared within {timeout_s}s"

    def _action_verify_playbook_exists(self, params: dict) -> tuple[bool, str]:
        """Verify a user playbook matching given criteria exists.

        Args:
            params (dict): Step params; optional trigger_contains, content_contains, user_id.

        Returns:
            tuple[bool, str]: (success, message)
        """
        user_id: str | None = params.get("user_id")
        trigger_contains: str | None = params.get("trigger_contains")
        content_contains: str | None = params.get("content_contains")

        resp = self._client.get_user_playbooks(user_id=user_id)
        playbooks = resp.user_playbooks

        for pb in playbooks:
            trigger_text = (pb.trigger or "").lower()
            content_text = (pb.content or "").lower()

            if (
                trigger_contains
                and trigger_contains.lower() not in trigger_text + content_text
            ):
                continue
            if content_contains and content_contains.lower() not in content_text:
                continue
            return True, f"found matching playbook: {pb.content[:80]!r}"

        criteria = []
        if trigger_contains:
            criteria.append(f"trigger~{trigger_contains!r}")
        if content_contains:
            criteria.append(f"content~{content_contains!r}")
        return (
            False,
            f"no playbook matched {', '.join(criteria)} among {len(playbooks)} playbooks",
        )

    def _action_seed_user_playbook(self, params: dict) -> tuple[bool, str]:
        """Directly add a user playbook (seed for search/aggregation tests).

        Args:
            params (dict): user_id, agent_version, content, and optional structured fields.

        Returns:
            tuple[bool, str]: (success, message)
        """
        from reflexio.models.api_schema.service_schemas import UserPlaybook

        user_id: str = params["user_id"]
        agent_version: str = params.get("agent_version", "openclaw-agent")
        content: str = params["content"]

        pb = UserPlaybook(
            user_id=user_id,
            agent_version=agent_version,
            request_id=f"eval-seed-{uuid.uuid4().hex[:8]}",
            content=content,
            trigger=params.get("trigger"),
            rationale=params.get("rationale"),
        )

        resp = self._client.add_user_playbook(user_playbooks=[pb])
        return True, f"seeded playbook → {resp.message}"

    def _action_search(self, params: dict) -> tuple[bool, str]:
        """Run a unified search and store results for subsequent verify steps.

        Args:
            params (dict): query and optional user_id.

        Returns:
            tuple[bool, str]: (success, message)
        """
        query: str = params["query"]
        user_id: str | None = params.get("user_id")

        resp = self._client.search(query=query, user_id=user_id)
        self._last_search_results = resp

        total = (
            len(resp.user_playbooks or [])
            + len(resp.agent_playbooks or [])
            + len(resp.profiles or [])
        )
        return True, f"search returned {total} results"

    def _action_verify_result_contains(self, params: dict) -> tuple[bool, str]:
        """Assert that at least one search result contains the expected text.

        Args:
            params (dict): field (content|trigger) and contains string.

        Returns:
            tuple[bool, str]: (success, message)
        """
        field_name: str = params.get("field", "content")
        needle: str = params["contains"].lower()

        results = self._last_search_results
        if results is None:
            return False, "no search results stored — run 'search' action first"

        all_texts = _collect_field_values(results, field_name)
        if any(needle in t.lower() for t in all_texts):
            return True, f"found {needle!r} in {field_name}"
        return (
            False,
            f"{needle!r} not found in any {field_name} value; got: {all_texts[:3]}",
        )

    def _action_verify_result_not_contains(self, params: dict) -> tuple[bool, str]:
        """Assert that no search result contains the unwanted text.

        Args:
            params (dict): field and contains string that must NOT appear.

        Returns:
            tuple[bool, str]: (success, message)
        """
        field_name: str = params.get("field", "content")
        needle: str = params["contains"].lower()

        results = self._last_search_results
        if results is None:
            return False, "no search results stored — run 'search' action first"

        all_texts = _collect_field_values(results, field_name)
        if any(needle in t.lower() for t in all_texts):
            return False, f"unwanted {needle!r} found in {field_name}"
        return True, f"{needle!r} correctly absent from results"

    def _action_run_aggregation(self, params: dict) -> tuple[bool, str]:
        """Trigger playbook aggregation via the client.

        Args:
            params (dict): agent_version, optional wait (bool).

        Returns:
            tuple[bool, str]: (success, message)
        """
        agent_version: str = params.get("agent_version", "openclaw-agent")
        wait: bool = bool(params.get("wait", True))

        resp = self._client.run_playbook_aggregation(
            agent_version=agent_version,
            wait_for_response=wait,
        )
        msg = resp.message if wait and resp is not None else "aggregation queued"
        return True, msg

    def _action_verify_has_agent_playbooks(self, _params: dict) -> tuple[bool, str]:
        """Assert the last search returned at least one agent playbook.

        Args:
            _params (dict): Unused.

        Returns:
            tuple[bool, str]: (success, message)
        """
        results = self._last_search_results
        if results is None:
            return False, "no search results stored — run 'search' action first"

        agent_pbs = results.agent_playbooks or []
        if agent_pbs:
            return True, f"{len(agent_pbs)} agent playbook(s) returned"
        return False, "no agent playbooks in search results"

    def _action_verify_agent_playbook_count(self, params: dict) -> tuple[bool, str]:
        """Check that agent playbooks for a given version satisfy count and content criteria.

        Args:
            params (dict): agent_version, max_expected count, optional content_contains.

        Returns:
            tuple[bool, str]: (success, message)
        """
        agent_version: str = params.get("agent_version", "openclaw-agent")
        max_expected: int = int(params.get("max_expected", 9999))
        content_contains: str | None = params.get("content_contains")

        resp = self._client.get_agent_playbooks(
            agent_version=agent_version, force_refresh=True
        )
        pbs = resp.agent_playbooks or []

        if content_contains:
            pbs = [
                p for p in pbs if content_contains.lower() in (p.content or "").lower()
            ]

        if len(pbs) > max_expected:
            return (
                False,
                f"expected ≤{max_expected} agent playbooks matching {content_contains!r}, got {len(pbs)}",
            )
        return True, f"{len(pbs)} agent playbook(s) — within limit of {max_expected}"

    def _action_run_cli(self, params: dict) -> tuple[bool, str]:
        """Run a CLI command as a subprocess, storing exit code.

        Args:
            params (dict): command list.

        Returns:
            tuple[bool, str]: (success, message)
        """
        cmd: list[str] = params["command"]
        env = {
            **os.environ,
            "REFLEXIO_URL": self._base_url,
        }
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)  # noqa: S603
        self._last_exit_code = result.returncode
        return True, f"exit={result.returncode}"

    def _action_verify_exit_code(self, params: dict) -> tuple[bool, str]:
        """Assert the last CLI command exited with the expected code.

        Args:
            params (dict): expected exit code.

        Returns:
            tuple[bool, str]: (success, message)
        """
        expected: int = int(params.get("expected", 0))
        if self._last_exit_code is None:
            return False, "no CLI command has run yet"
        if self._last_exit_code == expected:
            return True, f"exit code {self._last_exit_code} == {expected}"
        return False, f"exit code {self._last_exit_code} != {expected}"

    def _action_run_cli_expect_failure(self, params: dict) -> tuple[bool, str]:
        """Run a CLI command that is expected to fail, but must not crash.

        Args:
            params (dict): command list; should_not_crash bool.

        Returns:
            tuple[bool, str]: (success, message)
        """
        cmd: list[str] = params["command"]
        env = {
            **os.environ,
            "REFLEXIO_URL": self._base_url,
        }
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)  # noqa: S603
            self._last_exit_code = result.returncode
            # Non-zero exit is expected; what we care about is no Python traceback crash
            crashed = "Traceback (most recent call last):" in (result.stderr or "")
            if crashed:
                return False, "command produced an unhandled traceback"
            return (
                True,
                f"command completed (exit={result.returncode}) without crashing",
            )
        except Exception as exc:
            return False, f"subprocess raised exception: {exc}"

    def _action_stop_server(self, _params: dict) -> tuple[bool, str]:
        """Stop the Reflexio server.

        Args:
            _params (dict): Unused.

        Returns:
            tuple[bool, str]: (success, message)
        """
        self._stop_server()
        return True, "server stopped"

    def _action_start_server(self, _params: dict) -> tuple[bool, str]:
        """Start the Reflexio server.

        Args:
            _params (dict): Unused.

        Returns:
            tuple[bool, str]: (success, message)
        """
        self._start_server()
        return True, "server started"

    def _action_verify_server_running(self, _params: dict) -> tuple[bool, str]:
        """Assert the server is currently responding.

        Args:
            _params (dict): Unused.

        Returns:
            tuple[bool, str]: (success, message)
        """
        if self._server_is_running():
            return True, "server is responding"
        return False, "server did not respond"

    # ------------------------------------------------------------------
    # Dispatch table
    # ------------------------------------------------------------------

    def _dispatch(self, action: str, params: dict) -> tuple[bool, str]:
        """Route an action name to its handler.

        Args:
            action (str): Action identifier from the dataset.
            params (dict): Parameters for the action.

        Returns:
            tuple[bool, str]: (passed, message)
        """
        handlers = {
            "publish_interaction": self._action_publish_interaction,
            "wait_extraction": self._action_wait_extraction,
            "verify_playbook_exists": self._action_verify_playbook_exists,
            "seed_user_playbook": self._action_seed_user_playbook,
            "search": self._action_search,
            "verify_result_contains": self._action_verify_result_contains,
            "verify_result_not_contains": self._action_verify_result_not_contains,
            "run_aggregation": self._action_run_aggregation,
            "verify_has_agent_playbooks": self._action_verify_has_agent_playbooks,
            "verify_agent_playbook_count": self._action_verify_agent_playbook_count,
            "run_cli": self._action_run_cli,
            "verify_exit_code": self._action_verify_exit_code,
            "run_cli_expect_failure": self._action_run_cli_expect_failure,
            "stop_server": self._action_stop_server,
            "start_server": self._action_start_server,
            "verify_server_running": self._action_verify_server_running,
        }

        handler = handlers.get(action)
        if handler is None:
            return False, f"unknown action: {action!r}"
        return handler(params)

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------

    def _run_scenario(self, scenario: dict) -> ScenarioResult:
        """Execute a single scenario and return its result.

        Args:
            scenario (dict): A scenario dict from the dataset.

        Returns:
            ScenarioResult: Aggregated result with per-step outcomes.
        """
        scenario_id: str = scenario["id"]
        category: str = scenario.get("category", "")
        steps: list[dict] = scenario.get("steps", [])

        # Reset per-scenario state
        self._last_search_results = None
        self._last_exit_code = None

        start = time.monotonic()
        step_results: list[StepResult] = []
        scenario_passed = True

        for step in steps:
            action: str = step.get("action", "")
            params: dict = step.get("params", {})

            try:
                passed, message = self._dispatch(action, params)
            except Exception as exc:
                passed, message = False, f"exception: {exc}"

            step_results.append(
                StepResult(action=action, passed=passed, message=message)
            )

            if not passed:
                scenario_passed = False
                break  # stop on first failure

        elapsed = time.monotonic() - start
        return ScenarioResult(
            scenario_id=scenario_id,
            category=category,
            passed=scenario_passed,
            elapsed_s=elapsed,
            steps=step_results,
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> list[ScenarioResult]:
        """Load the dataset, start the server, run all scenarios, and clean up.

        Returns:
            list[ScenarioResult]: Results for every scenario that was run.
        """
        dataset = json.loads(_DATASET_PATH.read_text())
        scenarios: list[dict] = dataset["scenarios"]

        if self._scenario_ids:
            scenarios = [s for s in scenarios if s["id"] in self._scenario_ids]

        results: list[ScenarioResult] = []

        with tempfile.TemporaryDirectory(prefix="reflexio_eval_") as tmp_dir:
            self._tmp_dir = _TmpDirProxy(tmp_dir)

            try:
                self._start_server()
            except RuntimeError as exc:
                # If the server failed to start, all scenarios fail
                results.extend(
                    ScenarioResult(
                        scenario_id=s["id"],
                        category=s.get("category", ""),
                        passed=False,
                        elapsed_s=0.0,
                        error=str(exc),
                    )
                    for s in scenarios
                )
                return results

            try:
                for scenario in scenarios:
                    result = self._run_scenario(scenario)
                    results.append(result)
            finally:
                self._stop_server()

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_field_values(search_response: Any, field_name: str) -> list[str]:
    """Collect all values for a given field from a unified search response.

    Args:
        search_response: A UnifiedSearchViewResponse object.
        field_name (str): The field name to extract (e.g. "content", "trigger").

    Returns:
        list[str]: All non-empty string values found across all result lists.
    """
    values: list[str] = []

    def _extract(items: list | None) -> None:
        if not items:
            return
        for item in items:
            val = getattr(item, field_name, None)
            if val:
                values.append(str(val))

    _extract(getattr(search_response, "user_playbooks", None))
    _extract(getattr(search_response, "agent_playbooks", None))
    _extract(getattr(search_response, "profiles", None))
    return values


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(results: list[ScenarioResult]) -> None:
    """Print a human-readable pass/fail report to stdout.

    Args:
        results (list[ScenarioResult]): Scenario results to display.
    """
    total = len(results)
    passed_count = sum(1 for r in results if r.passed)

    print()
    print("=" * 70)
    print(f"  Reflexio-OpenClaw Integration Eval  ({passed_count}/{total} passed)")
    print("=" * 70)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"\n  [{status}] {result.scenario_id}  ({result.category})  {result.elapsed_s:.1f}s"
        )

        for step in result.steps:
            step_mark = "✓" if step.passed else "✗"
            print(f"         {step_mark} {step.action}: {step.message}")

        if result.error:
            print(f"         ! {result.error}")

    print()
    print(f"  Result: {passed_count}/{total} scenarios passed")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the evaluation suite and return an exit code.

    Returns:
        int: 0 if all scenarios passed, 1 otherwise.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Reflexio-OpenClaw integration evaluation runner"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_ids",
        metavar="ID",
        help="Run only the named scenario(s) (repeatable)",
    )
    args = parser.parse_args()

    runner = EvalRunner(scenario_ids=args.scenario_ids)
    results = runner.run()
    _print_report(results)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
