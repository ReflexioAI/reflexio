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

@test "profile write creates file in .reflexio/profiles with kebab+nanoid name" {
  cd "$WORKSPACE"
  run "$SCRIPT" profile diet-vegetarian one_year --body "User is vegetarian."
  [ "$status" -eq 0 ]
  [[ "$output" =~ \.reflexio/profiles/diet-vegetarian-[a-z0-9]{4}\.md$ ]]
  # File exists
  [ -f "$output" ]
}

@test "profile write emits frontmatter with type, id, created, ttl, expires" {
  cd "$WORKSPACE"
  path="$("$SCRIPT" profile diet-vegetarian one_year --body "User is vegetarian.")"
  run cat "$path"
  [[ "$output" == *"type: profile"* ]]
  [[ "$output" == *"id: prof_"* ]]
  [[ "$output" == *"created: "* ]]
  [[ "$output" == *"ttl: one_year"* ]]
  [[ "$output" == *"expires: "* ]]
}

@test "profile write body appears after frontmatter" {
  cd "$WORKSPACE"
  path="$("$SCRIPT" profile diet-vegetarian one_year --body "User is vegetarian — no meat.")"
  run cat "$path"
  [[ "$output" == *"User is vegetarian — no meat."* ]]
}

@test "profile write with ttl=infinity sets expires to never" {
  cd "$WORKSPACE"
  path="$("$SCRIPT" profile name-alice infinity --body "User's name is Alice.")"
  run cat "$path"
  [[ "$output" == *"expires: never"* ]]
}

@test "profile write reads body from --body-file" {
  cd "$WORKSPACE"
  echo "User has two cats." > body.txt
  path="$("$SCRIPT" profile pets-cats one_year --body-file body.txt)"
  run cat "$path"
  [[ "$output" == *"User has two cats."* ]]
}

@test "profile write reads body from stdin when no --body flag" {
  cd "$WORKSPACE"
  path="$(echo "User has a dog." | "$SCRIPT" profile pets-dog one_year)"
  run cat "$path"
  [[ "$output" == *"User has a dog."* ]]
}

@test "profile write rejects missing ttl" {
  cd "$WORKSPACE"
  run "$SCRIPT" profile diet-vegetarian --body "x"
  [ "$status" -ne 0 ]
}

@test "profile write rejects invalid ttl value" {
  cd "$WORKSPACE"
  run "$SCRIPT" profile diet-vegetarian one_millennium --body "x"
  [ "$status" -ne 0 ]
}

@test "playbook write creates file in .reflexio/playbooks" {
  cd "$WORKSPACE"
  body="$(cat <<EOF
## When
Composing a git commit.

## What
No AI-attribution trailers.

## Why
User corrected this; fix stuck.
EOF
)"
  run "$SCRIPT" playbook commit-no-ai-attribution --body "$body"
  [ "$status" -eq 0 ]
  [[ "$output" =~ \.reflexio/playbooks/commit-no-ai-attribution-[a-z0-9]{4}\.md$ ]]
  [ -f "$output" ]
}

@test "playbook write frontmatter has type, id, created (but not ttl)" {
  cd "$WORKSPACE"
  path="$("$SCRIPT" playbook test-pb --body "content")"
  run cat "$path"
  [[ "$output" == *"type: playbook"* ]]
  [[ "$output" == *"id: pbk_"* ]]
  [[ "$output" == *"created: "* ]]
  [[ "$output" != *"ttl:"* ]]
  [[ "$output" != *"expires:"* ]]
}

@test "playbook ignores ttl argument if passed" {
  # We don't want playbook to accept ttl. Playbook should reject extra positional args.
  cd "$WORKSPACE"
  run "$SCRIPT" playbook test-pb one_year --body "content"
  [ "$status" -ne 0 ]
}

@test "profile with single --supersedes emits YAML array" {
  cd "$WORKSPACE"
  path="$("$SCRIPT" profile diet-vegan one_year --body "User is vegan." --supersedes "prof_abc1")"
  run cat "$path"
  [[ "$output" == *"supersedes: [prof_abc1]"* ]]
}

@test "profile with multi --supersedes emits comma-joined YAML array" {
  cd "$WORKSPACE"
  path="$("$SCRIPT" profile diet-vegan one_year --body "User is vegan." --supersedes "prof_abc1,prof_def2")"
  run cat "$path"
  [[ "$output" == *"supersedes: [prof_abc1, prof_def2]"* ]]
}

@test "profile without supersedes omits the field" {
  cd "$WORKSPACE"
  path="$("$SCRIPT" profile diet-vegan one_year --body "x")"
  run cat "$path"
  [[ "$output" != *"supersedes:"* ]]
}

@test "playbook with --supersedes emits YAML array" {
  cd "$WORKSPACE"
  path="$("$SCRIPT" playbook test-pb --body "x" --supersedes "pbk_xyz9")"
  run cat "$path"
  [[ "$output" == *"supersedes: [pbk_xyz9]"* ]]
}

@test "profile write leaves no .tmp files behind on success" {
  cd "$WORKSPACE"
  "$SCRIPT" profile diet one_year --body "x" > /dev/null
  run find .reflexio -name "*.tmp*"
  [ -z "$output" ]
}

@test "playbook write leaves no .tmp files behind on success" {
  cd "$WORKSPACE"
  "$SCRIPT" playbook foo --body "x" > /dev/null
  run find .reflexio -name "*.tmp*"
  [ -z "$output" ]
}

@test "profile write: final file is either complete or absent (never partial)" {
  # We can't easily inject a mid-write failure, but we can verify the .tmp+rename
  # pattern by inspecting the script itself — smoke test.
  grep -q 'mv "\$tmp" "\$path"' "$SCRIPT"
}
