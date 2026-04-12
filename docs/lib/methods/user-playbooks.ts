import { MethodDef } from "../types";

const STATUS_ENUM = ["current", "archived", "pending", "archive_in_progress"];

export const userPlaybookMethods: MethodDef[] = [
  {
    id: "get-user-playbooks",
    pythonName: "get_user_playbooks",
    displayName: "Get User Playbooks",
    group: "user-playbooks",
    description:
      "Get user playbooks (per-user, per-request) with optional filtering.",
    httpMethod: "POST",
    endpoint: "/api/get_user_playbooks",
    requestStyle: "json_body",
    params: [
      {
        name: "limit",
        type: "number",
        required: false,
        default: 100,
        description: "Maximum number of results to return",
      },
      {
        name: "user_id",
        type: "string",
        required: false,
        description: "Filter by user ID",
      },
      {
        name: "playbook_name",
        type: "string",
        required: false,
        description: "Filter by playbook name/category",
      },
      {
        name: "agent_version",
        type: "string",
        required: false,
        description: "Filter by agent version",
      },
      {
        name: "status_filter",
        type: "json",
        required: false,
        description: `Status filter as JSON array. Values: ${STATUS_ENUM.join(", ")}, null (for current)`,
      },
    ],
  },
  {
    id: "search-user-playbooks",
    pythonName: "search_user_playbooks",
    displayName: "Search User Playbooks",
    group: "user-playbooks",
    description:
      "Search for user playbooks with semantic/text search and filtering.",
    httpMethod: "POST",
    endpoint: "/api/search_user_playbooks",
    requestStyle: "json_body",
    params: [
      {
        name: "query",
        type: "string",
        required: false,
        description: "Query for semantic/text search",
      },
      {
        name: "user_id",
        type: "string",
        required: false,
        description: "Filter by user (via request_id linkage to requests table)",
      },
      {
        name: "agent_version",
        type: "string",
        required: false,
        description: "Filter by agent version",
      },
      {
        name: "playbook_name",
        type: "string",
        required: false,
        description: "Filter by playbook name",
      },
      {
        name: "start_time",
        type: "datetime",
        required: false,
        description: "Start time for created_at filter (ISO 8601)",
      },
      {
        name: "end_time",
        type: "datetime",
        required: false,
        description: "End time for created_at filter (ISO 8601)",
      },
      {
        name: "status_filter",
        type: "json",
        required: false,
        description: `Status filter as JSON array. Values: ${STATUS_ENUM.join(", ")}, null (for current)`,
      },
      {
        name: "top_k",
        type: "number",
        required: false,
        default: 10,
        description: "Maximum number of results to return",
      },
      {
        name: "threshold",
        type: "number",
        required: false,
        default: 0.4,
        description: "Similarity threshold for vector search (0.0 to 1.0)",
      },
      {
        name: "enable_reformulation",
        type: "boolean",
        required: false,
        default: false,
        description: "Enable LLM query reformulation",
      },
      {
        name: "search_mode",
        type: "enum",
        required: false,
        default: "hybrid",
        description:
          "Search mode: vector (embedding similarity), fts (full-text search), or hybrid (combined with RRF)",
        enumValues: ["vector", "fts", "hybrid"],
      },
    ],
  },
  {
    id: "add-user-playbook",
    pythonName: "add_user_playbook",
    displayName: "Add User Playbook",
    group: "user-playbooks",
    description:
      "Add user playbooks directly to storage, bypassing the extraction pipeline.",
    httpMethod: "POST",
    endpoint: "/api/add_user_playbook",
    requestStyle: "json_body",
    params: [
      {
        name: "user_playbooks",
        type: "json",
        required: true,
        description:
          'List of user playbook objects. Each must have at least one of content / structured_data fields populated, e.g. [{"agent_version": "v1", "request_id": "req-1", "playbook_name": "greeting", "content": "..."}]',
      },
    ],
  },
  {
    id: "update-user-playbook",
    pythonName: "update_user_playbook",
    displayName: "Update User Playbook",
    group: "user-playbooks",
    description:
      "Update editable fields of a user playbook. structured_data is replaced wholesale.",
    httpMethod: "PUT",
    endpoint: "/api/update_user_playbook",
    requestStyle: "json_body",
    params: [
      {
        name: "user_playbook_id",
        type: "number",
        required: true,
        description: "The user playbook ID to update",
      },
      {
        name: "playbook_name",
        type: "string",
        required: false,
        description: "New playbook category name",
      },
      {
        name: "content",
        type: "string",
        required: false,
        description: "New content text",
      },
      {
        name: "structured_data",
        type: "json",
        required: false,
        description:
          'Replacement structured data block (full replacement), e.g. {"trigger": "...", "instruction": "...", "pitfall": "...", "rationale": "..."}',
      },
    ],
  },
  {
    id: "delete-user-playbook",
    pythonName: "delete_user_playbook",
    displayName: "Delete User Playbook",
    group: "user-playbooks",
    description: "Delete a user playbook by ID.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_user_playbook",
    requestStyle: "json_body",
    params: [
      {
        name: "user_playbook_id",
        type: "number",
        required: true,
        description: "The user playbook ID to delete",
      },
    ],
  },
  {
    id: "delete-user-playbooks-by-ids",
    pythonName: "delete_user_playbooks_by_ids",
    displayName: "Delete User Playbooks By IDs",
    group: "user-playbooks",
    description: "Delete multiple user playbooks by their IDs in one call.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_user_playbooks_by_ids",
    requestStyle: "json_body",
    params: [
      {
        name: "user_playbook_ids",
        type: "json",
        required: true,
        description: "List of user playbook IDs to delete, e.g. [1, 2, 3]",
      },
    ],
  },
];
