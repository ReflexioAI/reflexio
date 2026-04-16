#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage: reflexio-write.sh <type> <slug> [<ttl>] [--body <str> | --body-file <path>] [--supersedes <id1,id2,...>]

  <type>    profile | playbook
  <slug>    kebab-case, e.g. diet-vegetarian
  <ttl>     required for profile: one_day | one_week | one_month | one_quarter | one_year | infinity
  --body | --body-file | stdin  body content
  --supersedes  comma-separated IDs whose files this supersedes

Environment:
  WORKSPACE  filesystem root where .reflexio/ lives (defaults to pwd)
EOF
}

mkid() {
  local type="${1:-}"
  local prefix
  case "$type" in
    profile)  prefix="prof" ;;
    playbook) prefix="pbk"  ;;
    *) echo "mkid: unknown type '$type'" >&2; return 2 ;;
  esac
  local suffix
  suffix=$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom 2>/dev/null | head -c 4 || true)
  printf '%s_%s\n' "$prefix" "$suffix"
}

validate_slug() {
  local slug="${1:-}"
  if [[ -z "$slug" ]]; then
    echo "validate-slug: empty" >&2
    return 3
  fi
  if [[ ! "$slug" =~ ^[a-z0-9][a-z0-9-]{0,47}$ ]]; then
    echo "validate-slug: invalid format: $slug" >&2
    return 3
  fi
  return 0
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

case "$1" in
  mkid)
    shift
    mkid "$@"
    exit $?
    ;;
  validate-slug)
    shift
    validate_slug "$@"
    exit $?
    ;;
  profile|playbook)
    echo "not implemented" >&2
    exit 1
    ;;
  *)
    usage
    exit 2
    ;;
esac
