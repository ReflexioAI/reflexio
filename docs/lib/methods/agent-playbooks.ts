import { MethodDef } from "../types";

const STATUS_ENUM = ["current", "archived", "pending", "archive_in_progress"];
const PLAYBOOK_STATUS_ENUM = ["pending", "approved", "rejected"];

export const agentPlaybookMethods: MethodDef[] = [
  {
    id: "get-agent-playbooks",
    pythonName: "get_agent_playbooks",
    displayName: "Get Agent Playbooks",
    group: "agent-playbooks",
    description:
      "Get aggregated agent playbooks with optional filtering. Defaults to APPROVED playbooks.",
    httpMethod: "POST",
    endpoint: "/api/get_agent_playbooks",
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
      {
        name: "playbook_status_filter",
        type: "enum",
        required: false,
        description: "Filter by approval status",
        enumValues: PLAYBOOK_STATUS_ENUM,
      },
    ],
  },
  {
    id: "search-agent-playbooks",
    pythonName: "search_agent_playbooks",
    displayName: "Search Agent Playbooks",
    group: "agent-playbooks",
    description:
      "Search for aggregated agent playbooks with semantic/text search and filtering.",
    httpMethod: "POST",
    endpoint: "/api/search_agent_playbooks",
    requestStyle: "json_body",
    params: [
      {
        name: "query",
        type: "string",
        required: false,
        description: "Query for semantic/text search",
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
        name: "playbook_status_filter",
        type: "enum",
        required: false,
        description: "Filter by approval status",
        enumValues: PLAYBOOK_STATUS_ENUM,
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
    id: "add-agent-playbooks",
    pythonName: "add_agent_playbooks",
    displayName: "Add Agent Playbooks",
    group: "agent-playbooks",
    description:
      "Add aggregated agent playbooks directly to storage, bypassing the aggregation pipeline.",
    httpMethod: "POST",
    endpoint: "/api/add_agent_playbook",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_playbooks",
        type: "json",
        required: true,
        description:
          'List of agent playbook objects, e.g. [{"agent_version": "v1", "playbook_name": "greeting", "content": "...", "playbook_status": "pending", "playbook_metadata": "..."}]',
      },
    ],
  },
  {
    id: "update-agent-playbook",
    pythonName: "update_agent_playbook",
    displayName: "Update Agent Playbook",
    group: "agent-playbooks",
    description:
      "Update editable fields of an agent playbook. structured_data is replaced wholesale. For status-only changes, prefer update_agent_playbook_status.",
    httpMethod: "PUT",
    endpoint: "/api/update_agent_playbook",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_playbook_id",
        type: "number",
        required: true,
        description: "The agent playbook ID to update",
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
      {
        name: "playbook_status",
        type: "enum",
        required: false,
        description: "New approval status",
        enumValues: PLAYBOOK_STATUS_ENUM,
      },
    ],
  },
  {
    id: "update-agent-playbook-status",
    pythonName: "update_agent_playbook_status",
    displayName: "Update Agent Playbook Status",
    group: "agent-playbooks",
    description:
      "Dedicated endpoint to change only the approval status of an agent playbook (approve / pending / reject).",
    httpMethod: "PUT",
    endpoint: "/api/update_agent_playbook_status",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_playbook_id",
        type: "number",
        required: true,
        description: "The agent playbook ID",
      },
      {
        name: "playbook_status",
        type: "enum",
        required: true,
        description: "New approval status",
        enumValues: PLAYBOOK_STATUS_ENUM,
      },
    ],
  },
  {
    id: "delete-agent-playbook",
    pythonName: "delete_agent_playbook",
    displayName: "Delete Agent Playbook",
    group: "agent-playbooks",
    description: "Delete an agent playbook by ID.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_agent_playbook",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_playbook_id",
        type: "number",
        required: true,
        description: "The agent playbook ID to delete",
      },
    ],
  },
  {
    id: "delete-agent-playbooks-by-ids",
    pythonName: "delete_agent_playbooks_by_ids",
    displayName: "Delete Agent Playbooks By IDs",
    group: "agent-playbooks",
    description: "Delete multiple agent playbooks by their IDs in one call.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_agent_playbooks_by_ids",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_playbook_ids",
        type: "json",
        required: true,
        description: "List of agent playbook IDs to delete, e.g. [1, 2, 3]",
      },
    ],
  },
];
