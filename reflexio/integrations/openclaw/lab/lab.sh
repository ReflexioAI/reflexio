#!/usr/bin/env bash
# Isolated local lab for the Reflexio OpenClaw plugin.
#
#   lab.sh up [--real [MODEL]]
#                        install OpenClaw, start Reflexio + stub LLM, install the
#                        packed plugin. --real uses your Codex subscription for
#                        the agent and for Reflexio's extraction instead of the
#                        stub (MODEL default openai/gpt-5.6-luna; see README.md).
#   lab.sh reinstall     rebuild, repack and reinstall the plugin only
#   lab.sh turn MESSAGE  run one embedded agent turn, print the JSON envelope
#   lab.sh gateway [stop] start (or stop) a Gateway, needed for session_end
#                        (e.g. via /new). While it runs, --local turns refuse.
#   lab.sh oc ARGS...    run any openclaw command against the lab state
#   lab.sh env           print the exports for using the lab from a shell
#   lab.sh down [--wipe] stop the lab's processes (and delete its directory)
#
# Nothing here touches ~/.openclaw, ~/.reflexio or a shared backend port: the
# OpenClaw install, state, Reflexio HOME/data and every port live under
# $OPENCLAW_LAB_DIR (default: ${TMPDIR:-/tmp}/reflexio-openclaw-lab).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"   # lab/ -> open_source/reflexio
LAB="${OPENCLAW_LAB_DIR:-${TMPDIR:-/tmp}/reflexio-openclaw-lab}"
OC_VERSION="${OPENCLAW_LAB_VERSION:-2026.10.6}"
NODE_MIN="24.22.1"
PLUGIN_ID="reflexio-openclaw-smart"

die() { echo "lab: $*" >&2; exit 1; }
log() { echo "lab: $*" >&2; }

node_ok() {
  command -v node >/dev/null && node -e '
    const [a,b,c]=process.versions.node.split(".").map(Number), [x,y,z]=process.argv[1].split(".").map(Number);
    process.exit(a>x||(a===x&&(b>y||(b===y&&c>=z)))?0:1)' "$NODE_MIN"
}

ensure_node() {
  node_ok && return
  if [ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "${NVM_DIR:-$HOME/.nvm}/nvm.sh" >/dev/null
    nvm use --silent "$NODE_MIN" >/dev/null 2>&1 || nvm install "$NODE_MIN" >/dev/null
    nvm use --silent "$NODE_MIN" >/dev/null
  fi
  node_ok || die "OpenClaw $OC_VERSION needs Node >= $NODE_MIN (found $(node -v 2>/dev/null || echo none))"
}

lab_env() {
  export PATH="$LAB/prefix/bin:$(dirname "$(command -v node)"):$PATH"
  # Every lab process gets the lab's HOME: OpenClaw keeps some state (audit
  # records, skill installs) under HOME rather than OPENCLAW_STATE_DIR. The uv
  # and npm download caches stay shared; they hold packages, not user state,
  # and a fresh cache would re-download several GB per lab.
  if [ "$HOME" != "$LAB/home" ]; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$(uv cache dir 2>/dev/null)}"
    export npm_config_cache="${npm_config_cache:-$HOME/.npm}"
    export HOME="$LAB/home"
  fi
  mkdir -p "$HOME"
  export OPENCLAW_STATE_DIR="$LAB/state" OPENCLAW_CONFIG_PATH="$LAB/state/openclaw.json"
  export OPENCLAW_GATEWAY_TOKEN=lab-token
  # shellcheck disable=SC1091
  [ -f "$LAB/ports.env" ] && . "$LAB/ports.env"
  return 0
}

free_port() { python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }

# Ports are picked once per lab and reused, so check them again before each
# start: attaching to whatever else now listens there would test the wrong server.
# Bind with SO_REUSEADDR, as the services themselves do, so connections the
# previous run left in TIME_WAIT do not read as taken; a live listener still does.
require_free_port() {
  python3 -c 'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1])))' "$1" 2>/dev/null \
    || die "port $1 is taken by another process; run 'lab.sh down --wipe' to pick new ports"
}

wait_http() {
  for _ in $(seq 1 60); do curl -sf "$1" >/dev/null && return 0; sleep 1; done
  die "timed out waiting for $1"
}

install_openclaw() {
  if [ "$("$LAB/prefix/bin/openclaw" --version 2>/dev/null | awk '{print $2}')" = "$OC_VERSION" ]; then
    return
  fi
  log "installing openclaw@$OC_VERSION into $LAB/prefix"
  npm install -g --prefix "$LAB/prefix" "openclaw@$OC_VERSION" --no-fund --no-audit --loglevel=error
}

start_backend() {
  log "starting Reflexio backend on 127.0.0.1:$BACKEND_PORT (sqlite, $LAB/data)"
  require_free_port "$BACKEND_PORT"
  require_free_port "$EMBED_PORT"
  mkdir -p "$LAB/home" "$LAB/data" "$LAB/logs"
  local -a extra_env=(MOCK_LLM_RESPONSE=true OPENAI_API_KEY=sk-placeholder-mock)
  if [ "${MODE:-stub}" = real ]; then
    # Extraction runs through the lab's OpenClaw (and so the same Codex login).
    extra_env=(OPENCLAW_SMART_USE_LOCAL_CLI=1 "OPENCLAW_BIN=$LAB/prefix/bin/openclaw"
      "OPENCLAW_STATE_DIR=$LAB/state" "OPENCLAW_CONFIG_PATH=$LAB/state/openclaw.json")
  fi
  # Run from the lab directory (not the repo root) so no developer .env is
  # loaded, and give the colocated embedding service its own port instead of
  # the shared default. Redirect outside the subshell and exec into the
  # service, so no shell is left holding this script's stdout open.
  (cd "$LAB" && exec env HOME="$LAB/home" LOCAL_STORAGE_PATH="$LAB/data" \
    REFLEXIO_LOG_DIR="$LAB/logs" "${extra_env[@]}" setsid uv run --no-sync --project "$REPO_ROOT" \
    reflexio services start --only backend --backend-port "$BACKEND_PORT" \
    --embedding-port "$EMBED_PORT" --no-reload --storage sqlite) \
    </dev/null >"$LAB/backend.log" 2>&1 &
  echo $! >"$LAB/backend.pid"
  wait_http "http://127.0.0.1:$BACKEND_PORT/health"
}

start_stub() {
  log "starting stub LLM on 127.0.0.1:$STUB_PORT"
  require_free_port "$STUB_PORT"
  setsid python3 -I "$HERE/stub_llm.py" "$STUB_PORT" "$LAB/llm-requests.jsonl" </dev/null >"$LAB/stub.log" 2>&1 &
  echo $! >"$LAB/stub.pid"
  wait_http "http://127.0.0.1:$STUB_PORT/v1/models"
}

write_config() {
  python3 - "$OPENCLAW_CONFIG_PATH" "${MODE:-stub}" "$STUB_PORT" "$GATEWAY_PORT" <<'PY'
import json, sys
from pathlib import Path
path, mode, port = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cfg = json.loads(path.read_text()) if path.exists() else {}
if mode == "stub":
    cfg.setdefault("models", {})["mode"] = "merge"
    cfg["models"].setdefault("providers", {})["stub"] = {
        "baseUrl": f"http://127.0.0.1:{port}/v1", "apiKey": "stub", "api": "openai-completions",
        "models": [{"id": "stub-model", "name": "stub", "reasoning": False, "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 128000, "maxTokens": 4096}],
    }
    cfg.setdefault("agents", {}).setdefault("defaults", {})["model"] = {"primary": "stub/stub-model"}
gateway = cfg.setdefault("gateway", {})
gateway.update({"mode": "local", "port": int(sys.argv[4]), "bind": "loopback"})
gateway["auth"] = {"mode": "token", "token": "lab-token"}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(cfg, indent=2))
PY
}

cmd_gateway() {
  log "starting Gateway on 127.0.0.1:$GATEWAY_PORT"
  require_free_port "$GATEWAY_PORT"
  setsid openclaw gateway --port "$GATEWAY_PORT" --bind loopback </dev/null >"$LAB/gateway.log" 2>&1 &
  echo $! >"$LAB/gateway.pid"
  for _ in $(seq 1 90); do openclaw health >/dev/null 2>&1 && return 0; sleep 2; done
  die "Gateway did not become healthy; see $LAB/gateway.log"
}

install_plugin() {
  log "building the SDK wheel and installing its bundled plugin through setup"
  # Import the built wheel outside the source checkout. This is the customer
  # installation door, including bundle compilation and host config migration.
  uv build --wheel --clear --out-dir "$LAB/wheel" "$REPO_ROOT" >/dev/null
  local wheel
  wheel=$(find "$LAB/wheel" -name '*.whl' -print -quit)
  rm -rf "$LAB/sdk"
  uv pip install --no-deps --target "$LAB/sdk" "$wheel" >/dev/null
  (cd "$LAB" && PYTHONPATH="$LAB/sdk" "$REPO_ROOT/.venv/bin/python" - "$LAB/plugin.env" "$BACKEND_PORT" <<'PYINSTALL'
import sys
from pathlib import Path
from reflexio.cli.commands import setup_cmd

env_path = Path(sys.argv[1])
env_path.write_text(f"REFLEXIO_URL=http://127.0.0.1:{sys.argv[2]}/\n")
assert Path(setup_cmd.__file__).is_relative_to(Path.cwd() / "sdk")
if not setup_cmd._install_openclaw_integration(env_path):
    raise SystemExit("SDK plugin installation did not load all hooks")
PYINSTALL
  )
}

sign_in_openai() {
  # The lab gets its OWN ChatGPT/Codex login, via device code. Importing
  # ~/.codex would share a rotating refresh token with the Codex CLI, and a
  # refresh here could sign that CLI out.
  if openclaw models auth list --provider openai --json 2>/dev/null \
    | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("profiles") else 1)'; then
    return
  fi
  log "sign in to ChatGPT/Codex for this lab: open the URL below and enter the code"
  openclaw models auth login --provider openai --device-code
}

configure_real_extraction() {
  log "pointing Reflexio extraction at openclaw/$REAL_MODEL"
  (cd "$REPO_ROOT" && uv run --no-sync python - "http://127.0.0.1:$BACKEND_PORT/" "openclaw/$REAL_MODEL" <<'PY'
import sys
from reflexio import ReflexioClient
from reflexio.models.config_schema import LLMConfig
client = ReflexioClient(url_endpoint=sys.argv[1])
config = client.get_config()
model = sys.argv[2]
config.llm_config = LLMConfig(
    generation_model_name=model, should_run_model_name=model, pre_retrieval_model_name=model
)
client.set_config(config)
PY
  )
  openclaw config set agents.defaults.model.primary "$REAL_MODEL" >/dev/null
}

cmd_up() {
  MODE=stub
  if [ "${1:-}" = "--real" ]; then MODE=real; REAL_MODEL="${2:-openai/gpt-5.6-luna}"; fi
  ensure_node
  mkdir -p "$LAB"
  install_openclaw
  if [ ! -f "$LAB/ports.env" ]; then
    printf 'BACKEND_PORT=%s\nEMBED_PORT=%s\nSTUB_PORT=%s\nGATEWAY_PORT=%s\nLAB_MODE=%s\n' \
      "$(free_port)" "$(free_port)" "$(free_port)" "$(free_port)" "$MODE" >"$LAB/ports.env"
  fi
  lab_env
  if [ -z "${EMBED_PORT:-}" ]; then echo "EMBED_PORT=$(free_port)" >>"$LAB/ports.env"; lab_env; fi
  [ "$LAB_MODE" = "$MODE" ] || die "lab exists in $LAB_MODE mode; run 'lab.sh down --wipe' first"
  write_config
  [ "$MODE" = real ] && sign_in_openai
  [ "$MODE" = stub ] && start_stub
  start_backend
  [ "$MODE" = real ] && configure_real_extraction
  install_plugin
  log "up ($MODE): backend http://127.0.0.1:$BACKEND_PORT, state $LAB/state"
}

stop_service() {
  local pidfile="$LAB/$1.pid" pattern
  [ -f "$pidfile" ] || return 0
  case "$1" in
    backend) pattern="reflexio services start" ;;
    stub) pattern="stub_llm.py" ;;
    gateway) pattern="openclaw" ;;
  esac
  local pid; pid="$(cat "$pidfile")"
  rm -f "$pidfile"
  # A stale pid file can name a recycled pid; only signal our own process.
  if ps -o args= -p "$pid" 2>/dev/null | grep -qF "$pattern"; then
    # Each service runs in its own process group (setsid): the backend
    # launcher spawns uvicorn workers, so stop the whole group.
    kill -TERM -- "-$pid" 2>/dev/null || true
  fi
}

cmd_down() {
  for name in gateway backend stub; do stop_service "$name"; done
  lab_env
  if [ -n "${BACKEND_PORT:-}" ]; then
    for _ in $(seq 1 20); do curl -sf "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null || break; sleep 1; done
  fi
  if [ "${1:-}" = "--wipe" ]; then rm -rf "$LAB"; log "wiped $LAB"; else log "stopped"; fi
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  down) shift; cmd_down "$@" ;;
  reinstall) ensure_node; lab_env; install_plugin ;;
  turn) shift; ensure_node; lab_env; openclaw agent --local --agent main -m "$*" --json ;;
  gateway)
    ensure_node; lab_env
    if [ "${2:-}" = stop ]; then
      stop_service gateway
      for _ in $(seq 1 30); do openclaw health >/dev/null 2>&1 || exit 0; sleep 1; done
      die "Gateway still answering after stop"
    fi
    cmd_gateway ;;
  oc) shift; ensure_node; lab_env; openclaw "$@" ;;
  env) ensure_node; lab_env
    echo "export PATH=\"$PATH\" OPENCLAW_STATE_DIR=\"$OPENCLAW_STATE_DIR\" OPENCLAW_CONFIG_PATH=\"$OPENCLAW_CONFIG_PATH\""
    echo "export OPENCLAW_GATEWAY_TOKEN=\"$OPENCLAW_GATEWAY_TOKEN\" HOME=\"$HOME\""
    echo "export UV_CACHE_DIR=\"$UV_CACHE_DIR\" npm_config_cache=\"$npm_config_cache\""
    echo "export REFLEXIO_URL=\"http://127.0.0.1:${BACKEND_PORT:-?}/\" OPENCLAW_LAB_DIR=\"$LAB\"" ;;
  *) sed -n '2,17p' "$0" >&2; exit 2 ;;
esac
