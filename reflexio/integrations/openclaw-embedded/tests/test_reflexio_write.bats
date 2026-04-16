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

@test "mkid subcommand prints prof_ prefix + 4 chars from [a-z0-9]" {
  run "$SCRIPT" mkid profile
  [ "$status" -eq 0 ]
  [[ "$output" =~ ^prof_[a-z0-9]{4}$ ]]
}

@test "mkid subcommand prints pbk_ prefix for playbook" {
  run "$SCRIPT" mkid playbook
  [ "$status" -eq 0 ]
  [[ "$output" =~ ^pbk_[a-z0-9]{4}$ ]]
}

@test "mkid produces different ids across calls (sanity randomness check)" {
  id1="$("$SCRIPT" mkid profile)"
  id2="$("$SCRIPT" mkid profile)"
  [ "$id1" != "$id2" ]
}
