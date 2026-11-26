
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    last_modified_timestamp INTEGER NOT NULL,
    generated_from_request_id TEXT NOT NULL DEFAULT '',
    profile_time_to_live TEXT NOT NULL DEFAULT 'infinity',
    expiration_timestamp INTEGER NOT NULL DEFAULT 4102444800,
    custom_features TEXT,
    embedding TEXT,
    source TEXT DEFAULT '',
    status TEXT,
    extractor_names TEXT,
    expanded_terms TEXT,
    tags TEXT,
    source_interaction_ids TEXT,
    source_span TEXT,
    notes TEXT,
    reader_angle TEXT,
    merged_into TEXT,
    superseded_by TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    retired_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_status ON profiles(status);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'User',
    token_count INTEGER,
    user_action TEXT NOT NULL DEFAULT 'none',
    user_action_description TEXT NOT NULL DEFAULT '',
    interacted_image_url TEXT NOT NULL DEFAULT '',
    image_encoding TEXT NOT NULL DEFAULT '',
    shadow_content TEXT NOT NULL DEFAULT '',
    expert_content TEXT NOT NULL DEFAULT '',
    tools_used TEXT,
    citations TEXT,
    retrieved_learnings TEXT,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_request_id ON interactions(request_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_user_created_at_desc
    ON interactions(user_id, created_at DESC, interaction_id DESC);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    agent_version TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL CHECK (trim(session_id) != ''),
    evaluation_only INTEGER NOT NULL DEFAULT 0,
    retrieval_experiment_id TEXT,
    retrieval_experiment_arm TEXT,
    CHECK (
        (retrieval_experiment_id IS NULL AND retrieval_experiment_arm IS NULL)
        OR
        (retrieval_experiment_id IS NOT NULL AND retrieval_experiment_arm IN ('treatment', 'holdout'))
    )
);
CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests(session_id);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_session_created_at_asc
    ON requests(session_id, created_at ASC, request_id ASC);
CREATE INDEX IF NOT EXISTS idx_requests_retrieval_experiment
    ON requests(retrieval_experiment_id, user_id, session_id, created_at, request_id);

CREATE TABLE IF NOT EXISTS session_outcomes (
    outcome_id TEXT NOT NULL UNIQUE,
    outcome_revision INTEGER NOT NULL CHECK (outcome_revision >= 1),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure', 'unknown')),
    occurred_at INTEGER NOT NULL,
    source TEXT NOT NULL,
    label TEXT,
    value REAL,
    metadata TEXT,
    outcome_contract_digest TEXT NOT NULL,
    finalized_trajectory_digest TEXT NOT NULL,
    governance_subject_ref TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_occurred_at ON session_outcomes(occurred_at);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_session_id ON session_outcomes(session_id);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_source_outcome ON session_outcomes(source, outcome);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_label ON session_outcomes(label);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_subject_ref ON session_outcomes(governance_subject_ref);

CREATE TABLE IF NOT EXISTS user_playbooks (
    user_playbook_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    playbook_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    request_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    trigger TEXT,
    rationale TEXT,
    blocking_issue TEXT,
    source_interaction_ids TEXT,
    status TEXT,
    source TEXT,
    embedding TEXT,
    expanded_terms TEXT,
    tags TEXT,
    source_span TEXT,
    notes TEXT,
    reader_angle TEXT,
    merged_into INTEGER,
    superseded_by INTEGER,
    retired_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_user_playbooks_playbook_name ON user_playbooks(playbook_name);
CREATE INDEX IF NOT EXISTS idx_user_playbooks_agent_version ON user_playbooks(agent_version);
CREATE INDEX IF NOT EXISTS idx_user_playbooks_status ON user_playbooks(status);
CREATE INDEX IF NOT EXISTS idx_user_playbooks_created_at ON user_playbooks(created_at);

CREATE TABLE IF NOT EXISTS agent_playbooks (
    agent_playbook_id INTEGER PRIMARY KEY AUTOINCREMENT,
    playbook_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    trigger TEXT,
    rationale TEXT,
    blocking_issue TEXT,
    playbook_status TEXT NOT NULL DEFAULT 'pending',
    playbook_metadata TEXT NOT NULL DEFAULT '',
    embedding TEXT,
    expanded_terms TEXT,
    tags TEXT,
    status TEXT,
    merged_into INTEGER,
    superseded_by INTEGER,
    retired_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_playbook_name ON agent_playbooks(playbook_name);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_agent_version ON agent_playbooks(agent_version);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_status ON agent_playbooks(status);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_created_at ON agent_playbooks(created_at);

CREATE TABLE IF NOT EXISTS agent_success_evaluation_result (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    evaluation_name TEXT,
    is_success INTEGER NOT NULL DEFAULT 0,
    failure_type TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    regular_vs_shadow TEXT,
    number_of_correction_per_session INTEGER NOT NULL DEFAULT 0,
    user_turns_to_resolution INTEGER,
    is_escalated INTEGER NOT NULL DEFAULT 0,
    tags TEXT,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_agent_version ON agent_success_evaluation_result(agent_version);
CREATE INDEX IF NOT EXISTS idx_eval_created_at ON agent_success_evaluation_result(created_at);
CREATE INDEX IF NOT EXISTS idx_eval_created_at_desc
    ON agent_success_evaluation_result(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_agent_version_created_at_desc
    ON agent_success_evaluation_result(agent_version, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_identity_created_at_desc
    ON agent_success_evaluation_result(user_id, session_id, evaluation_name, agent_version, created_at DESC);

CREATE TABLE IF NOT EXISTS retrieved_learning_evaluation (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    interaction_id INTEGER,
    interaction_created_at INTEGER,
    kind TEXT NOT NULL,
    learning_id TEXT NOT NULL,
    is_relevant INTEGER,
    relevance_reason TEXT NOT NULL DEFAULT '',
    impact TEXT,
    impact_reason TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    governance_subject_ref TEXT,
    UNIQUE (user_id, session_id, interaction_id, kind, learning_id)
);
CREATE INDEX IF NOT EXISTS idx_rle_created_at_result_id
    ON retrieved_learning_evaluation(created_at DESC, result_id DESC);
CREATE INDEX IF NOT EXISTS idx_rle_interaction_created_at
    ON retrieved_learning_evaluation(
        interaction_created_at DESC, interaction_id DESC, result_id DESC
    );
CREATE INDEX IF NOT EXISTS idx_rle_session_id
    ON retrieved_learning_evaluation(session_id);
CREATE INDEX IF NOT EXISTS idx_rle_subject_ref
    ON retrieved_learning_evaluation(governance_subject_ref);

CREATE TABLE IF NOT EXISTS agent_playbook_source_user_playbooks (
    agent_playbook_id INTEGER NOT NULL,
    user_playbook_id INTEGER NOT NULL,
    source_interaction_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (agent_playbook_id, user_playbook_id)
);
CREATE INDEX IF NOT EXISTS idx_apsup_agent ON agent_playbook_source_user_playbooks(agent_playbook_id);
CREATE INDEX IF NOT EXISTS idx_apsup_user ON agent_playbook_source_user_playbooks(user_playbook_id);

CREATE TABLE IF NOT EXISTS playbook_optimization_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    optimizer_kind TEXT NOT NULL DEFAULT 'optimizer_legacy_unknown'
        CHECK (optimizer_kind IN (
            'gepa',
            'offline_tuner_open_world',
            'offline_tuner_legacy',
            'optimizer_legacy_unknown'
        )),
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    best_candidate_id INTEGER,
    successor_target_id INTEGER,
    decision_reason TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    discovery_key TEXT,
    attempt_key TEXT,
    lease_owner TEXT,
    lease_fence INTEGER NOT NULL DEFAULT 0 CHECK (lease_fence >= 0),
    lease_expires_at INTEGER,
    stage TEXT CHECK (stage IS NULL OR stage IN (
        'evidence_frozen',
        'discovery_analyzed',
        'candidate_generated',
        'held_out_analyzed',
        'publishing',
        'applied',
        'abstained',
        'failed'
    )),
    terminal_outcome TEXT CHECK (terminal_outcome IS NULL OR terminal_outcome IN (
        'applied',
        'insufficient_negative_evidence',
        'insufficient_positive_evidence',
        'insufficient_coverage',
        'deployment_unsupported',
        'candidate_regressed',
        'candidate_did_not_improve',
        'incumbent_changed',
        'generation_failed',
        'publication_failed',
        'governance_erased',
        'no_grounded_hypothesis',
        'analyst_unqualified',
        'heldout_evidence_failed',
        'stale_incumbent',
        'governance_invalidated',
        'infrastructure_failure'
    )),
    expected_population_manifest_digest TEXT,
    generation_selection_manifest_digest TEXT,
    replay_manifest_digest TEXT,
    candidate_content_digest TEXT,
    search_projection_digest TEXT,
    publication_scope_digest TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poj_target ON playbook_optimization_jobs(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_poj_status ON playbook_optimization_jobs(status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_discovery
    ON playbook_optimization_jobs(optimizer_kind, discovery_key)
    WHERE status IN ('pending', 'running') AND discovery_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_attempt
    ON playbook_optimization_jobs(optimizer_kind, attempt_key)
    WHERE status IN ('pending', 'running') AND attempt_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_target
    ON playbook_optimization_jobs(optimizer_kind, target_kind, target_id)
    WHERE status IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS case_first_exposure (
    equivalence_group_id TEXT PRIMARY KEY,
    first_role TEXT NOT NULL CHECK (first_role IN (
        'generation', 'improvement', 'preservation', 'safety'
    )),
    owner_discovery_key TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS case_exposure_uses (
    job_id INTEGER NOT NULL,
    equivalence_group_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN (
        'generation', 'improvement', 'preservation', 'safety'
    )),
    ownership TEXT NOT NULL CHECK (ownership IN (
        'self_generation', 'self_improvement', 'self_preservation',
        'self_safety', 'foreign'
    )),
    content_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (job_id, equivalence_group_id, role),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_case_exposure_uses_group
    ON case_exposure_uses(equivalence_group_id);

CREATE TABLE IF NOT EXISTS playbook_optimization_artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN (
        'expected_population_manifest',
        'generation_selection',
        'candidate',
        'candidate_search_projection',
        'open_world_evidence_bundle',
        'open_world_discovery_memo',
        'open_world_candidate',
        'open_world_attempt_decision'
    )),
    content_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (job_id, artifact_kind),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_poa_job
    ON playbook_optimization_artifacts(job_id);

CREATE TABLE IF NOT EXISTS offline_tuner_open_world_qualifications (
    component_identity_digest TEXT NOT NULL,
    suite_digest TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    class_counts_json TEXT NOT NULL,
    observation_digests_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (component_identity_digest, suite_digest)
);
CREATE TRIGGER IF NOT EXISTS offline_tuner_open_world_qualifications_no_update
BEFORE UPDATE ON offline_tuner_open_world_qualifications
BEGIN
    SELECT RAISE(ABORT, 'open-world qualification records are immutable');
END;
CREATE TRIGGER IF NOT EXISTS offline_tuner_open_world_qualifications_no_delete
BEFORE DELETE ON offline_tuner_open_world_qualifications
BEGIN
    SELECT RAISE(ABORT, 'open-world qualification records are immutable');
END;

CREATE TABLE IF NOT EXISTS playbook_optimization_candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    parent_candidate_ids TEXT NOT NULL DEFAULT '[]',
    aggregate_score REAL,
    is_winner INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poc_job ON playbook_optimization_candidates(job_id);

CREATE TABLE IF NOT EXISTS playbook_optimization_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    scenario_user_playbook_id INTEGER,
    source_interaction_ids TEXT NOT NULL DEFAULT '[]',
    score REAL NOT NULL DEFAULT 0.0,
    verdict TEXT NOT NULL DEFAULT 'tie',
    likert INTEGER NOT NULL DEFAULT 0,
    rationale TEXT NOT NULL DEFAULT '',
    asi_json TEXT NOT NULL DEFAULT '{}',
    incumbent_rollout_json TEXT NOT NULL DEFAULT '[]',
    candidate_rollout_json TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poe_job ON playbook_optimization_evaluations(job_id);
CREATE INDEX IF NOT EXISTS idx_poe_candidate ON playbook_optimization_evaluations(candidate_id);

CREATE TABLE IF NOT EXISTS playbook_optimization_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poev_job ON playbook_optimization_events(job_id);

CREATE TABLE IF NOT EXISTS user_playbook_publication_claims (
    optimizer_kind TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    owner TEXT NOT NULL,
    publication_fence INTEGER NOT NULL CHECK (publication_fence >= 1),
    worker_fence INTEGER NOT NULL CHECK (worker_fence >= 1),
    consumed INTEGER NOT NULL DEFAULT 0 CHECK (consumed IN (0, 1)),
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (optimizer_kind, job_id),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_playbook_publication_staging (
    optimizer_kind TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    attempt_key TEXT NOT NULL,
    claim_owner TEXT NOT NULL,
    publication_fence INTEGER NOT NULL,
    worker_fence INTEGER NOT NULL,
    incumbent_user_playbook_id INTEGER NOT NULL,
    incumbent_content_digest TEXT NOT NULL,
    incumbent_trigger TEXT,
    incumbent_semantic_digest TEXT NOT NULL,
    revised_content TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    projection_json TEXT NOT NULL,
    projection_digest TEXT NOT NULL,
    proof_json TEXT NOT NULL,
    proof_digest TEXT NOT NULL,
    subject_epochs_json TEXT NOT NULL,
    request_id TEXT NOT NULL,
    staging_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (optimizer_kind, job_id),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_playbook_publication_results (
    optimizer_kind TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('applied', 'incumbent_changed')),
    successor_user_playbook_id INTEGER,
    staging_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (optimizer_kind, job_id),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _operation_state (
    service_name TEXT PRIMARY KEY,
    operation_state TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS _agent_runs (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    extractor_kind TEXT NOT NULL,
    user_id TEXT,
    request_id TEXT NOT NULL,
    agent_version TEXT,
    source TEXT,
    source_interaction_ids TEXT NOT NULL DEFAULT '[]',
    window_start_interaction_id INTEGER,
    window_end_interaction_id INTEGER,
    extractor_config_hash TEXT,
    status TEXT NOT NULL,
    generation_request_snapshot TEXT NOT NULL DEFAULT '{}',
    service_config_snapshot TEXT,
    agent_context_snapshot TEXT,
    committed_output TEXT,
    pending_tool_call_ids TEXT NOT NULL DEFAULT '[]',
    max_steps_remaining INTEGER,
    resume_attempts INTEGER NOT NULL DEFAULT 0,
    finalization_attempts INTEGER NOT NULL DEFAULT 0,
    next_resume_at TEXT,
    claimed_by TEXT,
    claimed_at TEXT,
    agent_completed_at TEXT,
    finalized_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expires_at TEXT,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_ready ON _agent_runs(status, next_resume_at, updated_at);
CREATE INDEX IF NOT EXISTS idx_agent_runs_binding ON _agent_runs(org_id, extractor_kind, user_id);

CREATE TABLE IF NOT EXISTS _agent_run_finalization_receipts (
    run_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('profile', 'user_playbook')),
    learning_ids TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (run_id) REFERENCES _agent_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS _pending_tool_calls (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    user_id TEXT,
    scope TEXT NOT NULL DEFAULT '{}',
    scope_hash TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    dedup_key TEXT NOT NULL,
    status TEXT NOT NULL,
    question_text TEXT NOT NULL,
    answer_format TEXT,
    args TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    result TEXT,
    embedding TEXT,
    superseded_by TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    resolved_at TEXT,
    expires_at TEXT NOT NULL,
    cache_until TEXT NOT NULL,
    valid_until TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_tool_calls_active ON _pending_tool_calls(org_id, scope_hash, tool_name, dedup_key, status, cache_until);
CREATE INDEX IF NOT EXISTS idx_pending_tool_calls_prior ON _pending_tool_calls(org_id, scope_hash, tool_name, status, valid_until);

CREATE TABLE IF NOT EXISTS _run_tool_dependencies (
    run_id TEXT NOT NULL,
    pending_tool_call_id TEXT NOT NULL,
    dependency_kind TEXT NOT NULL DEFAULT 'followup',
    resolved_at TEXT,
    consumed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (run_id, pending_tool_call_id),
    FOREIGN KEY (run_id) REFERENCES _agent_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (pending_tool_call_id) REFERENCES _pending_tool_calls(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_run_tool_dependencies_pending ON _run_tool_dependencies(pending_tool_call_id, resolved_at, consumed_at);
CREATE INDEX IF NOT EXISTS idx_run_tool_dependencies_ready ON _run_tool_dependencies(run_id, resolved_at, consumed_at);

-- FTS5 virtual tables
CREATE VIRTUAL TABLE IF NOT EXISTS interactions_fts USING fts5(
    content, user_action_description,
    tokenize="porter unicode61"
);

CREATE VIRTUAL TABLE IF NOT EXISTS profiles_fts USING fts5(
    profile_id, content,
    tokenize="porter unicode61"
);

CREATE VIRTUAL TABLE IF NOT EXISTS user_playbooks_fts USING fts5(
    search_text,
    tokenize="porter unicode61"
);

CREATE VIRTUAL TABLE IF NOT EXISTS agent_playbooks_fts USING fts5(
    search_text,
    tokenize="porter unicode61"
);

-- Unicode lexical candidate indexes. Main-table triggers keep these normalized
-- n-gram sidecars current so non-ASCII searches never scan every document.
CREATE VIRTUAL TABLE IF NOT EXISTS interactions_unicode_fts USING fts5(
    search_ngrams,
    tokenize="unicode61 remove_diacritics 0"
);
CREATE VIRTUAL TABLE IF NOT EXISTS profiles_unicode_fts USING fts5(
    search_ngrams,
    tokenize="unicode61 remove_diacritics 0"
);
CREATE VIRTUAL TABLE IF NOT EXISTS user_playbooks_unicode_fts USING fts5(
    search_ngrams,
    tokenize="unicode61 remove_diacritics 0"
);
CREATE VIRTUAL TABLE IF NOT EXISTS agent_playbooks_unicode_fts USING fts5(
    search_ngrams,
    tokenize="unicode61 remove_diacritics 0"
);

CREATE TRIGGER IF NOT EXISTS interactions_unicode_fts_ai
AFTER INSERT ON interactions BEGIN
    INSERT INTO interactions_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.user_action_description, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS interactions_unicode_fts_au
AFTER UPDATE OF content, user_action_description ON interactions BEGIN
    DELETE FROM interactions_unicode_fts WHERE rowid = old.rowid;
    INSERT INTO interactions_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.user_action_description, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS interactions_unicode_fts_ad
AFTER DELETE ON interactions BEGIN
    DELETE FROM interactions_unicode_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS profiles_unicode_fts_ai
AFTER INSERT ON profiles BEGIN
    INSERT INTO profiles_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.expanded_terms, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS profiles_unicode_fts_au
AFTER UPDATE OF content, expanded_terms ON profiles BEGIN
    DELETE FROM profiles_unicode_fts WHERE rowid = old.rowid;
    INSERT INTO profiles_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.expanded_terms, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS profiles_unicode_fts_ad
AFTER DELETE ON profiles BEGIN
    DELETE FROM profiles_unicode_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS user_playbooks_unicode_fts_ai
AFTER INSERT ON user_playbooks BEGIN
    INSERT INTO user_playbooks_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.trigger, '') || ' ' ||
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.rationale, '') || ' ' ||
            COALESCE(new.source, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS user_playbooks_unicode_fts_au
AFTER UPDATE OF trigger, content, rationale, source ON user_playbooks BEGIN
    DELETE FROM user_playbooks_unicode_fts WHERE rowid = old.rowid;
    INSERT INTO user_playbooks_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.trigger, '') || ' ' ||
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.rationale, '') || ' ' ||
            COALESCE(new.source, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS user_playbooks_unicode_fts_ad
AFTER DELETE ON user_playbooks BEGIN
    DELETE FROM user_playbooks_unicode_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS agent_playbooks_unicode_fts_ai
AFTER INSERT ON agent_playbooks BEGIN
    INSERT INTO agent_playbooks_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.trigger, '') || ' ' ||
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.rationale, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS agent_playbooks_unicode_fts_au
AFTER UPDATE OF trigger, content, rationale ON agent_playbooks BEGIN
    DELETE FROM agent_playbooks_unicode_fts WHERE rowid = old.rowid;
    INSERT INTO agent_playbooks_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.trigger, '') || ' ' ||
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.rationale, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS agent_playbooks_unicode_fts_ad
AFTER DELETE ON agent_playbooks BEGIN
    DELETE FROM agent_playbooks_unicode_fts WHERE rowid = old.rowid;
END;

-- ============================================================================
-- Braintrust connector (Plan C-backend)
-- ============================================================================

CREATE TABLE IF NOT EXISTS braintrust_connection (
    org_id TEXT PRIMARY KEY,
    api_key_enc TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    workspace_name TEXT NOT NULL DEFAULT '',
    project_ids TEXT NOT NULL DEFAULT '[]',
    last_sync_ts INTEGER,
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS imported_score (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    session_id TEXT,
    scorer_name TEXT NOT NULL,
    value REAL NOT NULL,
    ts INTEGER NOT NULL,
    UNIQUE (org_id, source, source_run_id, scorer_name)
);
CREATE INDEX IF NOT EXISTS idx_imported_score_session
    ON imported_score (org_id, session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_imported_score_ts ON imported_score (org_id, ts);

-- ============================================================================
-- Per-turn shadow comparison verdicts (F1)
-- ============================================================================

CREATE TABLE IF NOT EXISTS shadow_comparison_verdicts (
    verdict_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id          TEXT    NOT NULL,
    session_id              TEXT    NOT NULL,
    agent_version           TEXT    NOT NULL,
    reflexio_is_request_1   INTEGER NOT NULL,
    better_request          TEXT    NOT NULL CHECK (better_request IN ('1','2','tie')),
    is_significantly_better INTEGER NOT NULL,
    comparison_reason       TEXT,
    judge_prompt_version    TEXT    NOT NULL,
    created_at              TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_session
    ON shadow_comparison_verdicts (session_id, agent_version);
CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_created_at
    ON shadow_comparison_verdicts (created_at);
CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_prompt_v
    ON shadow_comparison_verdicts (judge_prompt_version);
CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_prompt_created_at_desc
    ON shadow_comparison_verdicts (judge_prompt_version, created_at DESC);

-- ============================================================================
-- Append-only, content-free lineage event log
-- ============================================================================

CREATE TABLE IF NOT EXISTS lineage_event (
    event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id           TEXT NOT NULL,
    entity_type      TEXT NOT NULL,
    entity_id        TEXT NOT NULL,
    op               TEXT NOT NULL,
    prov_relation    TEXT NOT NULL DEFAULT '',
    source_ids       TEXT NOT NULL DEFAULT '[]',
    actor            TEXT NOT NULL DEFAULT '',
    request_id       TEXT NOT NULL DEFAULT '',
    reason           TEXT NOT NULL DEFAULT '',
    created_at       INTEGER NOT NULL,
    from_status      TEXT,
    to_status        TEXT,
    status_namespace TEXT,
    model_name       TEXT,
    provider         TEXT,
    UNIQUE (org_id, entity_type, entity_id, op, request_id)
);
CREATE INDEX IF NOT EXISTS idx_lineage_entity ON lineage_event (entity_type, entity_id);

-- ============================================================================
-- Durable learning pipeline — cross-org job queue
-- ============================================================================

CREATE TABLE IF NOT EXISTS learning_jobs (
    job_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    job_type TEXT NOT NULL DEFAULT 'learning',
    latest_request_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    claimed_by TEXT,
    claim_token TEXT,
    claim_expires_at TEXT,
    covers_through TEXT,
    force_extraction INTEGER NOT NULL DEFAULT 0,
    skip_aggregation INTEGER NOT NULL DEFAULT 0,
    project_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS learning_jobs_coalesce
    ON learning_jobs (org_id, user_id, job_type) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS learning_jobs_poll
    ON learning_jobs (created_at) WHERE status IN ('pending','failed','claimed');

