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

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

echo "not implemented" >&2
exit 1
