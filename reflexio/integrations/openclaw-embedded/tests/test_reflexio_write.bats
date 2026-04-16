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

@test "validate-slug accepts diet-vegetarian" {
  run "$SCRIPT" validate-slug "diet-vegetarian"
  [ "$status" -eq 0 ]
}

@test "validate-slug accepts abc" {
  run "$SCRIPT" validate-slug "abc"
  [ "$status" -eq 0 ]
}

@test "validate-slug rejects Empty" {
  run "$SCRIPT" validate-slug ""
  [ "$status" -ne 0 ]
}

@test "validate-slug rejects uppercase" {
  run "$SCRIPT" validate-slug "Diet-Vegetarian"
  [ "$status" -ne 0 ]
}

@test "validate-slug rejects starting with hyphen" {
  run "$SCRIPT" validate-slug "-diet"
  [ "$status" -ne 0 ]
}

@test "validate-slug rejects longer than 48 chars" {
  # 49 chars
  run "$SCRIPT" validate-slug "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  [ "$status" -ne 0 ]
}

@test "validate-slug rejects slashes" {
  run "$SCRIPT" validate-slug "foo/bar"
  [ "$status" -ne 0 ]
}
