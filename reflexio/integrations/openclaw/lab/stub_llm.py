"""Deterministic OpenAI-compatible chat endpoint for the OpenClaw lab.

Records every request body to a JSONL file so scenarios can assert on exactly
what OpenClaw sent the model, and answers with a fixed reply. When the prompt
carries injected ``[oc:…]`` tags in its latest user message, the reply ends
with a citation marker for them, so the plugin's citation path is exercised
without a real model. Older messages are ignored: OpenClaw replays earlier
turns' injections in the history, and citing those would attribute stale tags.

Usage: python stub_llm.py PORT REQUEST_LOG
"""

from __future__ import annotations

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPLY = "Four."
_TAG = re.compile(r"\[oc:([ps]-[0-9a-f]{6})\]")


# OpenClaw appends a host-authored runtime-context message after the user's
# own; it carries no injected learnings. 2026.10.10+ and the older carrier
# form (2026.9.x) spell it differently.
_RUNTIME_CONTEXT_MARKERS = (
    "OpenClaw runtime context",
    "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>",
)


def _is_runtime_context(message: dict) -> bool:
    content = message.get("content")
    text = content if isinstance(content, str) else json.dumps(content)
    return any(marker in text[:80] for marker in _RUNTIME_CONTEXT_MARKERS)


def _reply_for(body: dict) -> str:
    users = [
        m
        for m in body.get("messages", [])
        if m.get("role") == "user" and not _is_runtime_context(m)
    ]
    tags = _TAG.findall(json.dumps(users[-1].get("content"))) if users else []
    if not tags:
        return REPLY
    noun = "learning" if len(tags) == 1 else "learnings"
    return f"{REPLY}\n✨ {len(tags)} Reflexio {noun} applied [oc:{','.join(tags)}]"


class Handler(BaseHTTPRequestHandler):
    log_path: Path

    def log_message(self, *_args: object) -> None:
        """Keep the lab's stdout clean."""

    def _json(self, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, reply: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {
            "id": "stub",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "stub-model",
        }
        first = {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": reply},
                    "finish_reason": None,
                }
            ],
        }
        last = {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        for chunk in (first, last):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        self._json(
            {"object": "list", "data": [{"id": "stub-model", "object": "model"}]}
        )

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"path": self.path, "body": body}) + "\n")
        reply = _reply_for(body)
        if body.get("stream"):
            self._stream(reply)
            return
        self._json(
            {
                "id": "stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "stub-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )


def main() -> None:
    port, log_path = int(sys.argv[1]), Path(sys.argv[2])
    Handler.log_path = log_path
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
