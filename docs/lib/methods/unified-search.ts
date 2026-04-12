import { MethodDef } from "../types";

export const unifiedSearchMethods: MethodDef[] = [
  {
    id: "search",
    pythonName: "search",
    displayName: "Search",
    group: "unified-search",
    description:
      "Search across all entity types (profiles, agent playbooks, user playbooks) in parallel.",
    httpMethod: "POST",
    endpoint: "/api/search",
    requestStyle: "json_body",
    params: [
      {
        name: "query",
        type: "string",
        required: true,
        description: "Search query text",
      },
      {
        name: "top_k",
        type: "number",
        required: false,
        default: 5,
        description: "Maximum results per entity type",
      },
      {
        name: "threshold",
        type: "number",
        required: false,
        default: 0.3,
        description: "Similarity threshold for vector search (0.0 to 1.0)",
      },
      {
        name: "agent_version",
        type: "string",
        required: false,
        description:
          "Filter by agent version (agent_playbooks, user_playbooks)",
      },
      {
        name: "playbook_name",
        type: "string",
        required: false,
        description:
          "Filter by playbook name (agent_playbooks, user_playbooks)",
      },
      {
        name: "user_id",
        type: "string",
        required: false,
        description: "Filter by user ID (profiles, user_playbooks)",
      },
      {
        name: "enable_reformulation",
        type: "boolean",
        required: false,
        default: false,
        description: "Enable LLM query reformulation",
      },
      {
        name: "conversation_history",
        type: "json",
        required: false,
        description:
          'List of conversation turns for context, e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]',
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
];
