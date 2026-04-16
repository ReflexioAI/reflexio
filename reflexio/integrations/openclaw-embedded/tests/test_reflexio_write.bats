#!/usr/bin/env bats

SCRIPT="${BATS_TEST_DIRNAME}/../scripts/reflexio-write.sh"

setup() {
  export WORKSPACE="$(mktemp -d)"
  mkdir -p "$WORKSPACE/.reflexio/profiles" "$WORKSPACE/.reflexio/playbooks"
}

teardown() {
  rm -rf "$WORKSPACE"
}

@test "prints usage when no arguments given" {
  run "$SCRIPT"
  [ "$status" -ne 0 ]
  [[ "$output" == *"Usage:"* ]]
}
