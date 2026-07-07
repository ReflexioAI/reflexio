# Search golden set

Each YAML file is one search eval case, loaded by `tests/eval/conftest.py`
(`_load("search")`) and driven by `tests/eval/search/runner.py`.

## Case schema

```yaml
id: <str>                      # required, unique; also the pytest param id
category: preference | supersession | temporal_window | temporal_current | recall
query: <str>                   # the search query sent to unified search
conversation_history: []       # optional [{role, content}] turns
request_overrides:             # optional UnifiedSearchRequest field overrides
  user_id: u1                  # default u1
  top_k: 5
  search_mode: fts             # provider default is fts (deterministic keyless runs)
  entity_types: [profiles]

# Seeded entities. All timestamps are RELATIVE (age_days), resolved against
# the real clock at seed time, so cases never rot as calendar time passes.
seeded_profiles:
  - key: p_new                 # case-local key; doubles as the profile_id
    user_id: u1
    content: <str>
    age_days: 2                # last_modified_timestamp = now - age_days
    ttl: infinity              # ProfileTimeToLive value; expiration derived
seeded_user_playbooks:
  - key: b_new
    user_id: u1
    trigger: <str>
    content: <str>
    rationale: <str>           # optional
    age_days: 5                # created_at = now - age_days
seeded_agent_playbooks:
  - key: ap_new
    trigger: <str>
    content: <str>
    age_days: 5
    playbook_status: approved  # default approved (searchable by default)

# Gold labels. Candidates are case-local keys; the runner resolves them to
# real storage ids via the seeding map.
expected_top_candidates: [p_new]   # judged/measured within the key's own arm
must_NOT_rank_first: [p_old]       # ranking any of these first fails the case
expected_answer: <str>             # what a correct top hit conveys
expected_time_window:              # optional; asserted against the planner
  start_days_ago: 7                # once the deep tier exists (Phase 1+)
  end_days_ago: 0
notes_for_judge: <str>
```

Seeded entities are all **live** (no `superseded_by`/archived markers):
tombstoned rows are excluded from search anyway, so the realistic hard case —
extraction missed a supersession and retrieval must prefer the fresh fact —
requires both the stale and fresh items to be live.

## Categories

- `recall` — direct lexical/semantic match; a sanity floor every backend
  should pass.
- `preference` — disambiguation / bridging (e.g. "DB" vs a dataframe lib).
- `supersession` — stale fact vs fresh fact, both live; fresh must win.
- `temporal_current` — "current/latest X" phrasing; recency must dominate.
- `temporal_window` — query names a time window ("this week"); only
  in-window items are correct.
