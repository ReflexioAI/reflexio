#!/usr/bin/env bash
# Run once on plugin install. Syncs the Python env and flips on the
# openclaw LiteLLM provider in reflexio's .env so extraction works with no
# external API key.
#
# On failure, writes the reason to ~/.openclaw-smart/install-failed so
# hook_entry.sh can short-circuit and surface a user-visible message
# instead of silently no-op'ing every session.
#
# Dashboard: openclaw-smart shares claude-smart's dashboard at
# http://localhost:3001. We do not build a dashboard here.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_lib.sh
. "$HERE/_lib.sh"
openclaw_smart_source_login_path
openclaw_smart_prepend_astral_bins

PLUGIN_ROOT="$(cd "$HERE/.." && pwd)"

MARKER_DIR="$HOME/.openclaw-smart"
FAILURE_MARKER="$MARKER_DIR/install-failed"
SUCCESS_MARKER="$MARKER_DIR/install-complete"
INSTALL_LOCK="$MARKER_DIR/install.lock"
INSTALL_REAP_LOCK="$MARKER_DIR/install.lock.reap"
mkdir -p "$MARKER_DIR"

remove_stale_install_lock() {
  local expected current

  expected="$1"
  if ! mkdir "$INSTALL_REAP_LOCK" 2>/dev/null; then
    sleep 1
    return 0
  fi
  current="$(cat "$INSTALL_LOCK" 2>/dev/null || true)"
  if [ "$current" = "$expected" ]; then
    rm -f "$INSTALL_LOCK"
  fi
  rmdir "$INSTALL_REAP_LOCK" 2>/dev/null || true
}

acquire_install_lock() {
  local lock_pid

  if command -v flock >/dev/null 2>&1; then
    exec 9>"$INSTALL_LOCK"
    if flock 9; then
      return 0
    fi
    # flock unexpectedly failed (rare; usually a malformed lockfile fd
    # rather than a busy lock). Fall through to the portable lockfile
    # path below instead of bailing out — the installer must still run.
    echo "[openclaw-smart] flock failed; falling back to lockfile serialization" >&2
  fi

  while ! ( set -C; printf '%s\n' "$$" > "$INSTALL_LOCK" ) 2>/dev/null; do
    lock_pid="$(cat "$INSTALL_LOCK" 2>/dev/null || true)"
    case "$lock_pid" in
      ''|*[!0-9]*)
        remove_stale_install_lock "$lock_pid"
        ;;
      *)
        if kill -0 "$lock_pid" 2>/dev/null; then
          sleep 1
        else
          remove_stale_install_lock "$lock_pid"
        fi
        ;;
    esac
  done
  trap '[ "$(cat "$INSTALL_LOCK" 2>/dev/null || true)" = "$$" ] && rm -f "$INSTALL_LOCK" || true' EXIT
}

# Serialize concurrent installer runs (SessionStart hook + slash-command
# self-heal can both invoke this script). Wait for the active installer
# rather than returning early, otherwise callers can re-check uv before
# the first install has finished and report a false missing-dependency error.
acquire_install_lock

rm -f "$FAILURE_MARKER"

write_failure() {
  local reason
  reason="$1"
  printf '%s\n' "$reason" > "$FAILURE_MARKER"
  rm -f "$SUCCESS_MARKER"
  echo "[openclaw-smart] install failed: $reason" >&2
  echo ''
  exit 0
}

fingerprint_file() {
  local path
  path="$1"
  if [ -f "$path" ]; then
    cksum "$path" 2>/dev/null | awk '{print $1 ":" $2}'
  else
    printf 'missing\n'
  fi
}

install_fingerprint() {
  printf 'plugin_root=%s\n' "$PLUGIN_ROOT"
  printf 'smart_install=%s\n' "$(fingerprint_file "$HERE/smart-install.sh")"
  printf 'pyproject=%s\n' "$(fingerprint_file "$PLUGIN_ROOT/pyproject.toml")"
  printf 'uv_lock=%s\n' "$(fingerprint_file "$PLUGIN_ROOT/uv.lock")"
  # Resolved python interpreter — catches a system upgrade (3.12.4 → 3.12.5)
  # that would otherwise let install_complete return true against a venv
  # built against a now-deleted interpreter.
  if command -v uv >/dev/null 2>&1; then
    printf 'python=%s\n' "$(uv python find 3.12 2>/dev/null || echo missing)"
  else
    printf 'python=no-uv\n'
  fi
}

install_complete() {
  [ -f "$SUCCESS_MARKER" ] || return 1
  [ "$(cat "$SUCCESS_MARKER" 2>/dev/null || true)" = "$(install_fingerprint)" ] || return 1
  command -v uv >/dev/null 2>&1 || return 1
  [ -d "$PLUGIN_ROOT/.venv" ] || return 1
  [ -f "$HOME/.reflexio/.env" ] || return 1
  grep -q '^OPENCLAW_SMART_USE_LOCAL_CLI=' "$HOME/.reflexio/.env" || return 1
  grep -q '^OPENCLAW_SMART_USE_LOCAL_EMBEDDING=' "$HOME/.reflexio/.env" || return 1
  return 0
}

write_success_marker() {
  install_fingerprint > "$SUCCESS_MARKER"
}

preflight_supported_runtime_platform() {
  local os_name machine darwin_major
  os_name="$(uname -s 2>/dev/null || echo unknown)"
  machine="$(uname -m 2>/dev/null || echo unknown)"
  case "$os_name" in
    Darwin*)
      if [ "$machine" != "arm64" ]; then
        write_failure "openclaw-smart currently supports Apple Silicon macOS 14+ only; Intel Mac is not supported because native ML wheels are unavailable."
      fi
      darwin_major="$(uname -r 2>/dev/null | awk -F. '{print $1}')"
      case "$darwin_major" in
        ''|*[!0-9]*)
          write_failure "openclaw-smart could not determine the macOS version; Apple Silicon macOS 14+ is required."
          ;;
      esac
      if [ "$darwin_major" -lt 23 ]; then
        write_failure "openclaw-smart currently supports macOS 14+ on Apple Silicon; macOS 13 and older are not supported because native ML wheels are unavailable."
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      case "$machine" in
        x86_64|amd64) : ;;
        *)
          write_failure "openclaw-smart currently supports Windows x64 only; Windows ARM is not supported because native ML wheels are unavailable."
          ;;
      esac
      ;;
    Linux*)
      : # Existing Linux installs remain supported when package wheels are available.
      ;;
    *)
      write_failure "openclaw-smart currently supports Apple Silicon macOS 14+, Windows x64, and Linux for vanilla installs."
      ;;
  esac
}

preflight_supported_runtime_platform

if install_complete; then
  echo ''
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[openclaw-smart] uv not found — installing from astral.sh..." >&2
  # The astral.sh bash installer downloads a zip and unzips it. On
  # Windows-flavoured bash (Git Bash / MSYS) the bundled `unzip` corrupts
  # the Windows uv binary (bad CRC on the inflated uv.exe), leaving the
  # install half-finished. Use the official PowerShell installer
  # (install.ps1) on Windows, which writes uv.exe to ~/.local/bin
  # natively — same destination the bash installer targets on POSIX, so
  # openclaw_smart_prepend_astral_bins picks it up uniformly afterwards.
  if openclaw_smart_is_windows; then
    if ! command -v powershell >/dev/null 2>&1; then
      write_failure "uv install needs PowerShell on Windows but powershell is not on PATH — install uv manually from https://docs.astral.sh/uv/"
    fi
    if ! powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >&2; then
      write_failure "uv install via PowerShell failed — install manually from https://docs.astral.sh/uv/"
    fi
  else
    UV_INSTALLER="$MARKER_DIR/uv-install.sh"
    if ! openclaw_smart_download https://astral.sh/uv/install.sh "$UV_INSTALLER"; then
      write_failure "uv installer download failed — install manually from https://docs.astral.sh/uv/"
    fi
    if ! sh "$UV_INSTALLER" >&2; then
      write_failure "uv install failed — install manually from https://docs.astral.sh/uv/"
    fi
  fi
  openclaw_smart_prepend_astral_bins
  if ! command -v uv >/dev/null 2>&1; then
    UV_FOUND=""
    for candidate in "$HOME/.local/bin/uv" "$HOME/.local/bin/uv.exe" "$HOME/.cargo/bin/uv" "$HOME/bin/uv"; do
      if [ -x "$candidate" ]; then
        UV_FOUND="$candidate"
        break
      fi
    done
    if [ -n "$UV_FOUND" ]; then
      write_failure "uv installed at $UV_FOUND — add its parent directory to PATH in your shell rc"
    else
      write_failure "uv install reported success but binary not found — install manually from https://docs.astral.sh/uv/"
    fi
  fi
fi

cd "$PLUGIN_ROOT"
echo "[openclaw-smart] running uv sync..." >&2
if ! uv sync --locked --python 3.12 --quiet >&2; then
  write_failure "uv sync failed in $PLUGIN_ROOT — run 'uv sync --locked --python 3.12' there to diagnose"
fi

# Compile the TS shim to ./dist/index.js. openClaw 2026.5.12+ requires
# compiled JS at the path declared in package.json's openclaw.extensions.
# Source checkouts ship dist/ via the publisher; in dev mode (or when a
# user installs from a fresh git clone) we (re)build here so the loader
# can find ./dist/index.js. Skip silently when npm or the script is
# missing; the dashboard/install banner will surface the resulting load
# failure to the user.
if [ ! -f "$PLUGIN_ROOT/dist/index.js" ] && command -v npm >/dev/null 2>&1; then
  if [ -f "$PLUGIN_ROOT/package.json" ]; then
    echo "[openclaw-smart] compiling TS shim to dist/..." >&2
    (cd "$PLUGIN_ROOT" && npm install --silent && npm run build --silent) >&2 || \
      echo "[openclaw-smart] WARNING: npm install / build failed; openClaw may refuse to load the plugin" >&2
  fi
fi

# Reflexio's CLI reads ~/.reflexio/.env (see reflexio/cli/env_loader.py);
# append our two opt-in flags there so `reflexio services start` picks
# them up regardless of which directory the user runs it from. Keep
# claude-smart's flags intact if it is also installed.
REFLEXIO_ENV="$HOME/.reflexio/.env"
mkdir -p "$(dirname "$REFLEXIO_ENV")"
touch "$REFLEXIO_ENV"
if ! grep -q '^OPENCLAW_SMART_USE_LOCAL_CLI=' "$REFLEXIO_ENV"; then
  printf '\n# Route reflexio generation through the local openClaw CLI\nOPENCLAW_SMART_USE_LOCAL_CLI=1\n' >> "$REFLEXIO_ENV"
  echo "[openclaw-smart] appended OPENCLAW_SMART_USE_LOCAL_CLI=1 to $REFLEXIO_ENV" >&2
fi
if ! grep -q '^OPENCLAW_SMART_USE_LOCAL_EMBEDDING=' "$REFLEXIO_ENV"; then
  printf '# Use the in-process ONNX embedder (chromadb) — no API key for semantic search\nOPENCLAW_SMART_USE_LOCAL_EMBEDDING=1\n' >> "$REFLEXIO_ENV"
  echo "[openclaw-smart] appended OPENCLAW_SMART_USE_LOCAL_EMBEDDING=1 to $REFLEXIO_ENV" >&2
fi

if ! command -v openclaw >/dev/null 2>&1; then
  echo "[openclaw-smart] WARNING: 'openclaw' CLI not on PATH — reflexio extractors will have no LLM until it's installed" >&2
fi

# Point ~/.reflexio/openclaw-plugin-root at this install so slash commands
# can reference one stable short path regardless of which openClaw
# marketplace loaded us.
if ! bash "$HERE/ensure-plugin-root.sh" "$PLUGIN_ROOT"; then
  echo "[openclaw-smart] WARNING: failed to set ~/.reflexio/openclaw-plugin-root symlink — slash commands may not resolve" >&2
fi

write_success_marker
echo "[openclaw-smart] install complete. Backend auto-starts on session start." >&2
echo ''
