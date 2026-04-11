# Multi-User Scenario Evaluation Guide

This directory contains hand-authored multi-user conversation scenarios designed to test and evaluate the Reflexio extraction pipeline. Each scenario produces **profiles**, **user playbooks**, and **agent playbooks** when processed.

## Data Overview

### E2E Tier (small, frequent testing)


| File                         | Users                        | Turns | Tools |
| ---------------------------- | ---------------------------- | ----- | ----- |
| `e2e/customer_support.yaml`  | Priya Sharma, Marcus Johnson | 34    | No    |
| `e2e/developer_support.yaml` | Elena Rodriguez, Raj Patel   | 36    | No    |


### Eval Tier (deep evaluation with tool interactions)


| File                                  | Users                               | Turns | Tools                                                                                                   |
| ------------------------------------- | ----------------------------------- | ----- | ------------------------------------------------------------------------------------------------------- |
| `eval/ecommerce_refund.yaml`          | Priya, Marcus                       | 41    | `check_order_status`, `track_shipment`, `initiate_return`, `apply_discount`                             |
| `eval/healthcare_scheduling.yaml`     | David Kim, Aisha Williams, Tom Chen | 46    | `check_doctor_availability`, `book_appointment`, `check_insurance_coverage`, `verify_accessibility`     |
| `eval/financial_advisory.yaml`        | Sophie Laurent, James O'Brien       | 34    | `lookup_tax_treaty`, `calculate_qdro_estimate`, `compare_retirement_plans`, `check_contribution_limits` |
| `eval/saas_onboarding.yaml`           | Elena, Raj                          | 36    | `search_docs`, `check_compatibility`, `generate_api_key`, `create_webhook`                              |
| `eval/analytics_schema_debugging.yaml`| Alex                                | 8     | `Read`, `describe_table`, `run_sql`                                                                     |


---

## Quick Start

### 1. Start the server

```bash
reflexio services start --storage sqlite
```

### 2. Configure extractors

Set up profile and playbook extraction with aggressive batch settings for fast evaluation.

From the repo root:

```bash
reflexio config set --file tests/test_data/scenarios/eval_config.json
```

See `[eval_config.json](eval_config.json)` for the full configuration.

### 3. Export YAML scenarios to publishable JSON

Run `export_json.py` to convert the YAML scenarios into JSON files the CLI can publish:

```bash
# Export all e2e scenarios
python tests/test_data/scenarios/export_json.py --tier e2e

# Export all eval scenarios
python tests/test_data/scenarios/export_json.py --tier eval

# Export everything
python tests/test_data/scenarios/export_json.py --tier all

# Export a single scenario
python tests/test_data/scenarios/export_json.py --file tests/test_data/scenarios/e2e/customer_support.yaml

# Custom output directory
python tests/test_data/scenarios/export_json.py --tier e2e --output /tmp/scenarios
```

This creates one JSON file per user (e.g., `customer-support_priya.json`, `customer-support_marcus.json`).

### 4. Publish interactions

Publish each exported JSON file. The `user_id` is embedded in each JSON file, so `--user-id` on the CLI is optional (the file's value takes precedence):

```bash
# Publish all e2e files
for f in tests/test_data/scenarios/output/e2e/*.json; do
  reflexio publish --file "$f" --wait
done

# Or publish a single user
reflexio publish --file tests/test_data/scenarios/output/e2e/customer-support_priya.json --wait
```

---

## Verify Extraction Results

### Check profiles

```bash
# List profiles for a specific user
reflexio user-profiles list --user-id multiuser-customer-support-priya

# Search profiles semantically
reflexio user-profiles search "fashion buyer" --user-id multiuser-customer-support-priya
```

**Expected profiles for Priya**: name (Priya Sharma), location (London), occupation (fashion buyer at Harrods), clothing size (UK 8), petite range preference, upcoming wedding, email communication preference.

**Expected profiles for Marcus**: name (Marcus Johnson), location (Austin TX), membership tier (gold), occupation (freelance designer), work arrangement (remote/WFH), notification preference (SMS over email).

### Check user playbooks

```bash
# List all user playbooks
reflexio user-playbooks list --user-id multiuser-customer-support-priya

# Search for specific correction patterns
reflexio user-playbooks search "timeline" --user-id multiuser-customer-support-priya
reflexio user-playbooks search "tracking tool" --user-id multiuser-customer-support-priya
```

**Expected user playbooks for Priya**:

- Give exact dates, not ranges like "7-10 days"
- Present options as numbered lists
- Skip basic sizing advice (fashion professional)

**Expected user playbooks for Marcus**:

- Keep responses short, no filler
- Don't repeat user's information back
- Send updates via SMS, not email
- Use tracking tools proactively

**`analytics_schema_debugging.yaml`**: a synthetic analytics-debugging session
where the agent makes wrong column-name guesses against a fictional event
warehouse, recovers via a `SELECT * LIMIT 1` workaround, builds a cohort
comparison with asymmetric filters, and then gets challenged by the user.
This scenario exercises profile extraction (domain/schema facts about the
user's data) and playbook extraction (behavioral policies — trust-check,
symmetric cohort filtering, schema-discovery workflow, oversized-file
handling) in a single run. Run it to verify that the two extractors produce
disjoint sets: profiles should carry the table/column/unit facts, playbooks
should carry the agent SOPs, and neither should duplicate content from the
other. Publish with `reflexio publish --file ... --force-extraction --wait`.

### Check agent playbooks (after aggregation)

```bash
# Run aggregation to cluster similar user playbooks with agent 
reflexio agent-playbooks aggregate --agent-version support-bot-v2.1 --wait

# List aggregated agent playbooks
reflexio agent-playbooks list

# Search for generalizable rules
reflexio agent-playbooks search "check constraints before acting"
```

**Expected agent playbooks** (generalizable rules derived from multiple users):

1. "Gather all relevant constraints before proposing solutions"
2. "When a relevant tool is available, use it proactively on the first attempt"
3. "Assess user expertise level and adjust response depth/format accordingly"
4. "Explain actions and their implications before executing tools"

---

## Playbook Design Philosophy

The scenarios are designed with a layered playbook structure:

### User playbooks (per-user)

Each user has:

- **Common correction theme** shared with other users (e.g., "agent should check constraints before acting")
- **User-specific instructions** unique to that person (output format, tone, style, detail level)

### Agent playbooks (generalizable)

Derived from clustering similar user playbooks across users:

- Strip user-specific preferences (format, tone)
- Keep the behavioral rule that applies to ALL users
- Result: actionable SOPs the agent can follow for any new user

### Cross-scenario clustering targets


| Common Theme                    | Users Who Express It                                             |
| ------------------------------- | ---------------------------------------------------------------- |
| Check constraints before acting | Priya (urgency), David (scheduling), Sophie (dual-tax)           |
| Use available tools proactively | Marcus (tracking), Aisha (insurance), Elena (compatibility)      |
| Calibrate to user expertise     | Elena (senior→code), Raj (junior→basics), Aisha (clinical terms) |
| Explain before taking action    | Raj (panics at tools), Tom (age check), James (empathy)          |


---

## YAML Schema Reference

```yaml
id: scenario-name                    # Unique scenario identifier
description: "..."                    # Human-readable description

participants:
  user_key:
    display_name: "Full Name"
    role: User                        # or Assistant
  agent:
    role: Assistant

tools:                                # Optional (eval tier only)
  - name: tool_name
    description: "What this tool does"

conversations:
  user_key:
    source: "source-tag"
    agent_version: "v1.0"
    session_id: "unique-session-id"
    turns:
      - speaker: user_key             # References a participant key
        content: "Message text"
      - speaker: agent
        content: "Agent response"
        tool_calls:                   # Optional, agent turns only
          - name: tool_name
            input: { key: value }
            output: { key: value }
```

---

## Running E2E Tests

The open-source e2e tests use the customer_support scenario automatically via the `sample_interaction_requests` pytest fixture:

```bash
# Run all e2e tests
uv run pytest tests/e2e_tests/ -v -o 'addopts=' --timeout=300

# Run a specific test
uv run pytest tests/e2e_tests/test_interaction_workflows.py -v -o 'addopts='
```

