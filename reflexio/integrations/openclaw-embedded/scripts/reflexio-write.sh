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

# Compute expiration ISO date given TTL enum
compute_expires() {
  local ttl="$1"
  local created="$2"  # ISO-8601 timestamp, e.g. 2026-04-16T14:20:00Z
  local created_date="${created%%T*}"   # YYYY-MM-DD
  case "$ttl" in
    one_day)     date -u -j -f "%Y-%m-%d" -v+1d "$created_date" "+%Y-%m-%d" 2>/dev/null \
                   || date -u -d "$created_date + 1 day" "+%Y-%m-%d" ;;
    one_week)    date -u -j -f "%Y-%m-%d" -v+7d "$created_date" "+%Y-%m-%d" 2>/dev/null \
                   || date -u -d "$created_date + 7 days" "+%Y-%m-%d" ;;
    one_month)   date -u -j -f "%Y-%m-%d" -v+1m "$created_date" "+%Y-%m-%d" 2>/dev/null \
                   || date -u -d "$created_date + 1 month" "+%Y-%m-%d" ;;
    one_quarter) date -u -j -f "%Y-%m-%d" -v+3m "$created_date" "+%Y-%m-%d" 2>/dev/null \
                   || date -u -d "$created_date + 3 months" "+%Y-%m-%d" ;;
    one_year)    date -u -j -f "%Y-%m-%d" -v+1y "$created_date" "+%Y-%m-%d" 2>/dev/null \
                   || date -u -d "$created_date + 1 year" "+%Y-%m-%d" ;;
    infinity)    echo "never" ;;
    *) echo "compute_expires: invalid ttl: $ttl" >&2; return 4 ;;
  esac
}

# Main profile-write function
write_profile() {
  local slug="$1"
  local ttl="$2"
  local body="$3"
  local supersedes="${4:-}"  # comma-separated IDs, may be empty

  validate_slug "$slug" || return $?

  case "$ttl" in
    one_day|one_week|one_month|one_quarter|one_year|infinity) ;;
    *) echo "write_profile: invalid ttl: $ttl" >&2; return 4 ;;
  esac

  local id_suffix
  id_suffix=$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom 2>/dev/null | head -c 4 || true)
  local id="prof_${id_suffix}"
  local created
  created=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local expires
  expires=$(compute_expires "$ttl" "$created") || return $?

  local workspace="${WORKSPACE:-$PWD}"
  local dir="$workspace/.reflexio/profiles"
  mkdir -p "$dir"
  local path="$dir/${slug}-${id_suffix}.md"
  local tmp="${path}.tmp.$$"

  {
    echo "---"
    echo "type: profile"
    echo "id: $id"
    echo "created: $created"
    echo "ttl: $ttl"
    echo "expires: $expires"
    if [[ -n "$supersedes" ]]; then
      # Convert comma-separated list to YAML array
      local ids_yaml="[$(echo "$supersedes" | sed 's/,/, /g')]"
      echo "supersedes: $ids_yaml"
    fi
    echo "---"
    echo
    echo "$body"
  } > "$tmp"

  mv "$tmp" "$path"
  echo "$path"
}

main() {
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
    profile)
      shift
      # Parse: <slug> <ttl> [--body <str>|--body-file <path>|stdin] [--supersedes <ids>]
      local slug="${1:-}" ttl="${2:-}"
      [[ -z "$slug" || -z "$ttl" ]] && { usage; exit 2; }
      # Reject if ttl looks like a flag (user forgot ttl)
      [[ "$ttl" == --* ]] && { usage; exit 2; }
      shift 2
      local body="" body_source="" supersedes=""
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --body) body="$2"; body_source="arg"; shift 2 ;;
          --body-file) body="$(cat "$2")"; body_source="file"; shift 2 ;;
          --supersedes) supersedes="$2"; shift 2 ;;
          *) echo "unknown flag: $1" >&2; exit 2 ;;
        esac
      done
      if [[ -z "$body_source" ]]; then
        body="$(cat)"   # read from stdin
      fi
      write_profile "$slug" "$ttl" "$body" "$supersedes"
      ;;
    playbook)
      echo "not implemented" >&2
      exit 1
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
