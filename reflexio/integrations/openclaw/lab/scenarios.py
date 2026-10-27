"""Golden scenarios for the OpenClaw plugin, run against a live lab.

Prerequisite: ``lab.sh up`` (stub mode). Run from the Reflexio repo root:

    uv run --no-sync python reflexio/integrations/openclaw/lab/scenarios.py

Each scenario drives the real ``openclaw`` CLI and asserts on what landed in
Reflexio or in the request the model received -- never on an exit code alone,
because a plugin that fails open exits 0 whether or not it did anything.
Exits non-zero if any scenario fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reflexio import ReflexioClient

_THIS_DIR = Path(__file__).resolve().parent
LAB_SH = _THIS_DIR / "lab.sh"
PLUGIN_ID = "reflexio-openclaw-smart"


def _lab_env() -> dict[str, str]:
    out = subprocess.run(
        ["bash", str(LAB_SH), "env"], check=True, capture_output=True, text=True
    ).stdout
    env = os.environ.copy()
    for match in re.finditer(r'(\w+)="([^"]*)"', out):
        env[match.group(1)] = match.group(2)
    return env


ENV = _lab_env()
# lab.sh owns the lab location (and its OPENCLAW_LAB_DIR override); read it
# back rather than resolving it a second time here.
LAB = Path(ENV["OPENCLAW_LAB_DIR"])
PORTS = dict(line.split("=", 1) for line in (LAB / "ports.env").read_text().split())
BACKEND = f"http://127.0.0.1:{PORTS['BACKEND_PORT']}/"
CLIENT = ReflexioClient(url_endpoint=BACKEND)


def openclaw(
    *args: str, env: dict[str, str] | None = None, timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["openclaw", *args],
        env=env or ENV,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def turn(
    message: str, *, local: bool = True, env: dict[str, str] | None = None
) -> dict[str, Any]:
    args = ["agent", "--agent", "main", "-m", message, "--json"]
    proc = openclaw(*(["agent", "--local", *args[1:]] if local else args), env=env)
    assert proc.returncode == 0, f"turn failed: {proc.stderr[-500:]}"
    return json.loads(proc.stdout)


def reply_text(envelope: dict[str, Any]) -> str:
    return "".join(p.get("text") or "" for p in envelope.get("payloads") or [])


def interactions() -> list[dict[str, Any]]:
    response = CLIENT.get_all_interactions(limit=200)
    return [i.model_dump() for i in response.interactions]


def published_turn(user_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the (User, Assistant) interactions published for one prompt."""
    for _ in range(20):
        rows = interactions()
        users = [r for r in rows if r["role"] == "User" and r["content"] == user_text]
        if users:
            user = users[0]
            assistants = [
                r
                for r in rows
                if r["role"] == "Assistant" and r["request_id"] == user["request_id"]
            ]
            assert assistants, f"no Assistant row for request {user['request_id']}"
            return user, assistants[0]
        time.sleep(0.5)
    raise AssertionError(f"nothing published for prompt {user_text!r}")


def backend_hits(path: str) -> int:
    """Successful (2xx) requests to ``path`` in the backend access log."""
    log = (LAB / "backend.log").read_text(errors="replace")
    return len(re.findall(rf'"POST {re.escape(path)} HTTP/[\d.]+" 2\d\d', log))


def llm_requests() -> list[dict[str, Any]]:
    log = LAB / "llm-requests.jsonl"
    return (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )


def expected_user_id(workspace: str) -> str:
    """Python mirror of identity.ts, so the two languages are checked against each other."""
    path = str(Path(workspace).resolve())
    return f"openclaw:main:{Path(path).name.lower()}-{hashlib.sha256(path.encode()).hexdigest()[:8]}"


# --------------------------------------------------------------------------- scenarios


def s_load() -> None:
    proc = openclaw("plugins", "inspect", PLUGIN_ID, "--runtime", "--json")
    plugin = json.loads(proc.stdout)["plugin"]
    assert plugin["status"] == "loaded", plugin["status"]
    assert plugin["hookCount"] == 3, plugin["hookCount"]
    assert plugin["commands"] == ["reflexio"], plugin["commands"]


STATE: dict[str, str] = {}


def s_publish() -> None:
    prompt = f"Remember I deploy on Fridays ({uuid.uuid4().hex[:8]})."
    envelope = turn(prompt)
    user, assistant = published_turn(prompt)
    workspace = str(LAB / "state" / "workspace")
    assert user["user_id"] == expected_user_id(workspace), user["user_id"]
    assert re.fullmatch(r"openclaw-[0-9a-f]{32}", user["request_id"]), user[
        "request_id"
    ]
    assert assistant["content"] == reply_text(envelope)
    STATE["user_id"] = user["user_id"]


def s_inject_and_attribute() -> None:
    marker = f"SEED-{uuid.uuid4().hex[:8]}"
    profile_id = f"lab-{marker.lower()}"
    CLIENT.add_user_profile(
        [
            {
                "profile_id": profile_id,
                "user_id": STATE["user_id"],
                "content": f"{marker}: the user wants answers in exactly one sentence.",
                "last_modified_timestamp": int(time.time()),
                "generated_from_request_id": "lab-seed",
            }
        ]
    )
    before = len(llm_requests())
    prompt = f"How do plants make food? ({marker.lower()})"
    turn(prompt)
    sent = json.dumps(llm_requests()[before:])
    assert marker in sent, "seeded preference never reached the model's prompt"
    _, assistant = published_turn(prompt)
    learned = assistant.get("retrieved_learnings") or []
    assert any(
        r["learning_id"] == profile_id and r["kind"] == "profile" for r in learned
    ), learned
    # Citations are not on the interaction view; the dashboard's per-rule
    # stats are the read path that aggregates them.
    stats = CLIENT._make_request(
        "POST", "/api/get_playbook_application_stats", json={"days_back": 1}
    )
    applied = {
        row.get("real_id"): row.get("applied_count") for row in stats.get("stats", [])
    }
    assert applied.get(profile_id, 0) >= 1, applied


def s_duplicate_publish_is_recognized() -> None:
    """A re-sent request_id must read as delivered to the plugin; otherwise a
    publish that committed after a client timeout is retried forever. The
    server replays an identical retry as a success; client.ts ALREADY_ADMITTED
    covers older servers that refused it instead."""
    wire = json.loads(
        (_THIS_DIR.parent / "plugin/tests/fixtures/wire-requests.json").read_text()
    )
    body = {
        **wire["/api/publish_interaction"],
        "request_id": f"openclaw-lab-{uuid.uuid4().hex}",
        "user_id": STATE["user_id"],
        "session_id": "lab-duplicate",
    }
    first = CLIENT._make_request("POST", "/api/publish_interaction", json=body)
    assert first.get("success") is not False, first
    second = CLIENT._make_request("POST", "/api/publish_interaction", json=body)
    assert second.get("success") is True, second
    assert second.get("replayed") is True, second


def s_failed_backend_is_harmless() -> None:
    with socket.socket() as probe:  # a port that was free a moment ago
        probe.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{probe.getsockname()[1]}/"
    entry = f"plugins.entries.{PLUGIN_ID}.config.url"
    openclaw("config", "set", entry, dead)
    searches = backend_hits("/api/search")
    try:
        proc = openclaw(
            "agent",
            "--local",
            "--agent",
            "main",
            "-m",
            f"Still fine? ({uuid.uuid4().hex[:6]})",
            "--json",
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        assert reply_text(json.loads(proc.stdout)) == "Four.", (
            "turn changed while Reflexio was down"
        )
        assert "[reflexio] search failed" in proc.stderr, "outage was not reported"
        assert backend_hits("/api/search") == searches
    finally:
        openclaw("config", "set", entry, BACKEND)


def s_recursion_guard() -> None:
    searches = backend_hits("/api/search")
    turn(
        f"Internal extraction call ({uuid.uuid4().hex[:6]})",
        env={**ENV, "OPENCLAW_SMART_INTERNAL": "1"},
    )
    assert backend_hits("/api/search") == searches, (
        "plugin searched during an internal run"
    )


def s_session_end_finalizes() -> None:
    subprocess.run(
        ["bash", str(LAB_SH), "gateway"], check=True, env=ENV, capture_output=True
    )
    try:
        finalized = backend_hits("/api/finalize_session")
        turn(f"A gateway turn ({uuid.uuid4().hex[:6]})", local=False)
        turn("/new", local=False)
        for _ in range(20):
            if backend_hits("/api/finalize_session") > finalized:
                return
            time.sleep(0.5)
        raise AssertionError("/new did not finalize the Reflexio session")
    finally:
        # A running Gateway owns the state dir and makes --local turns refuse.
        subprocess.run(
            ["bash", str(LAB_SH), "gateway", "stop"],
            check=True,
            env=ENV,
            capture_output=True,
        )


MARKER_WORD = "Cheers"


def _learned(user_id: str) -> list[dict[str, Any]]:
    hits = CLIENT.search(
        query=f"end every answer with {MARKER_WORD}", user_id=user_id, top_k=10
    )
    items = [*hits.profiles, *hits.user_playbooks]
    return [
        i.model_dump()
        for i in items
        if MARKER_WORD.lower() in (i.content or "").lower()
    ]


def _gateway_turn(message: str, session_key: str) -> dict[str, Any]:
    """One turn through the lab Gateway in an explicit session.

    A real-model lab drives turns through a Gateway: Reflexio's extraction runs
    `openclaw infer` concurrently, and two embedded (--local) OpenClaw
    processes contend for the state database. Returns the run result
    (``payloads`` / ``meta``), which the Gateway envelope nests under
    ``result``.
    """
    proc = openclaw(
        "agent",
        "--agent",
        "main",
        "--session-key",
        session_key,
        "-m",
        message,
        "--json",
    )
    assert proc.returncode == 0, f"turn failed: {proc.stderr[-500:]}"
    envelope = json.loads(proc.stdout)
    assert envelope.get("status") == "ok", envelope
    return envelope["result"]


def s_real_learning_loop() -> None:
    """A real model learns a stated preference and applies it in a new session.

    Extraction runs through Reflexio's `openclaw` provider, i.e. `openclaw
    infer` on the lab's own login, so this also proves that provider end to
    end. Each phase uses a fresh session key, so the later session cannot see
    the instruction in its own history: only Reflexio can carry it across.
    """
    workspace = str(LAB / "state" / "workspace")
    user_id = expected_user_id(workspace)
    teach, apply = (f"agent:main:lab-{uuid.uuid4().hex[:8]}" for _ in range(2))
    subprocess.run(
        ["bash", str(LAB_SH), "gateway"], check=True, env=ENV, capture_output=True
    )
    try:
        first = _gateway_turn(
            f"From now on, always end every answer with the exact word '{MARKER_WORD}'. "
            "What is the capital of France?",
            teach,
        )
        _gateway_turn("Thanks. What is 3 + 3?", teach)
        CLIENT.finalize_session(
            user_id=user_id, session_id=first["meta"]["agentMeta"]["sessionId"]
        )
        deadline = time.monotonic() + 600
        while not (learned := _learned(user_id)):
            assert time.monotonic() < deadline, (
                "no learning extracted within 10 minutes"
            )
            time.sleep(10)
        ids = {str(i.get("profile_id") or i.get("user_playbook_id")) for i in learned}

        prompt = f"Name the largest planet. ({uuid.uuid4().hex[:6]})"
        result = _gateway_turn(prompt, apply)
        _, assistant = published_turn(prompt)
        retrieved = {
            r["learning_id"] for r in assistant.get("retrieved_learnings") or []
        }
        assert ids & retrieved, f"learned {ids} not injected (retrieved {retrieved})"
        reply = reply_text(result)
        assert MARKER_WORD.lower() in reply.lower(), (
            f"reply ignored the learning: {reply!r}"
        )
        print(f"      learned: {[i.get('content') for i in learned]}")
        print(f"      new-session reply: {reply!r}")
    finally:
        subprocess.run(
            ["bash", str(LAB_SH), "gateway", "stop"],
            check=True,
            env=ENV,
            capture_output=True,
        )


REAL_SCENARIOS: list[tuple[str, Callable[[], None]]] = [
    ("load", s_load),
    ("real model learns and applies a preference", s_real_learning_loop),
]

SCENARIOS: list[tuple[str, Callable[[], None]]] = [
    ("load", s_load),
    ("publish", s_publish),
    ("inject + attribute", s_inject_and_attribute),
    ("duplicate publish is recognized", s_duplicate_publish_is_recognized),
    ("backend down is harmless", s_failed_backend_is_harmless),
    ("recursion guard", s_recursion_guard),
    ("session_end finalizes", s_session_end_finalizes),
]


def main() -> int:
    failed = 0
    # A real-model lab has no stub to inspect, so it runs its own set.
    scenarios = REAL_SCENARIOS if PORTS.get("LAB_MODE") == "real" else SCENARIOS
    for name, scenario in scenarios:
        started = time.monotonic()
        try:
            scenario()
            print(f"PASS  {name}  ({time.monotonic() - started:.1f}s)")
        except Exception as exc:  # noqa: BLE001 — report every scenario
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"{len(scenarios) - failed}/{len(scenarios)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
