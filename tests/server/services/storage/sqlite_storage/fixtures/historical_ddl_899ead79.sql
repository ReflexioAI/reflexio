
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
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
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
    user_action TEXT NOT NULL DEFAULT 'none',
    user_action_description TEXT NOT NULL DEFAULT '',
    interacted_image_url TEXT NOT NULL DEFAULT '',
    shadow_content TEXT NOT NULL DEFAULT '',
    expert_content TEXT NOT NULL DEFAULT '',
    tools_used TEXT,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_request_id ON interactions(request_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    agent_version TEXT NOT NULL DEFAULT '',
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests(session_id);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);

CREATE TABLE IF NOT EXISTS user_playbooks (
    user_playbook_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    playbook_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    request_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    structured_data TEXT,
    source_interaction_ids TEXT,
    status TEXT,
    source TEXT,
    embedding TEXT,
    expanded_terms TEXT
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
    structured_data TEXT,
    playbook_status TEXT NOT NULL DEFAULT 'pending',
    playbook_metadata TEXT NOT NULL DEFAULT '',
    embedding TEXT,
    expanded_terms TEXT,
    status TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_playbook_name ON agent_playbooks(playbook_name);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_agent_version ON agent_playbooks(agent_version);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_status ON agent_playbooks(status);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_created_at ON agent_playbooks(created_at);

CREATE TABLE IF NOT EXISTS agent_success_evaluation_result (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_agent_version ON agent_success_evaluation_result(agent_version);
CREATE INDEX IF NOT EXISTS idx_eval_created_at ON agent_success_evaluation_result(created_at);

CREATE TABLE IF NOT EXISTS profile_change_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    added_profiles TEXT NOT NULL DEFAULT '[]',
    removed_profiles TEXT NOT NULL DEFAULT '[]',
    mentioned_profiles TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_pcl_user_id ON profile_change_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_pcl_created_at ON profile_change_logs(created_at);

CREATE TABLE IF NOT EXISTS playbook_aggregation_change_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    playbook_name TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    run_mode TEXT NOT NULL,
    added_playbooks TEXT NOT NULL DEFAULT '[]',
    removed_playbooks TEXT NOT NULL DEFAULT '[]',
    updated_playbooks TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_pacl_playbook_name ON playbook_aggregation_change_logs(playbook_name);
CREATE INDEX IF NOT EXISTS idx_pacl_agent_version ON playbook_aggregation_change_logs(agent_version);

CREATE TABLE IF NOT EXISTS _operation_state (
    service_name TEXT PRIMARY KEY,
    operation_state TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

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

