import { MethodDef } from "../types";

export const interactionMethods: MethodDef[] = [
  {
    id: "get-all-interactions",
    pythonName: "get_all_interactions",
    displayName: "Get All Interactions",
    group: "interactions",
    description: "Get all user interactions across all users.",
    httpMethod: "GET",
    endpoint: "/api/get_all_interactions",
    requestStyle: "query_params",
    params: [
      {
        name: "limit",
        type: "number",
        required: false,
        default: 100,
        description: "Maximum number of interactions to return",
      },
    ],
  },
  {
    id: "get-interactions",
    pythonName: "get_interactions",
    displayName: "Get Interactions",
    group: "interactions",
    description:
      "Get user interactions filtered by user ID, time range, and result count.",
    httpMethod: "POST",
    endpoint: "/api/get_interactions",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: true,
        description: "The user ID to get interactions for",
      },
      {
        name: "start_time",
        type: "datetime",
        required: false,
        description: "Filter by start time (ISO 8601)",
      },
      {
        name: "end_time",
        type: "datetime",
        required: false,
        description: "Filter by end time (ISO 8601)",
      },
      {
        name: "top_k",
        type: "number",
        required: false,
        default: 30,
        description: "Maximum number of results to return",
      },
    ],
  },
  {
    id: "search-interactions",
    pythonName: "search_interactions",
    displayName: "Search Interactions",
    group: "interactions",
    description:
      "Search for user interactions with semantic/text search and filtering.",
    httpMethod: "POST",
    endpoint: "/api/search_interactions",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: true,
        description: "The user ID to search for",
      },
      {
        name: "request_id",
        type: "string",
        required: false,
        description: "Filter by specific request ID",
      },
      {
        name: "query",
        type: "string",
        required: false,
        description: "Search query string",
      },
      {
        name: "start_time",
        type: "datetime",
        required: false,
        description: "Filter by start time (ISO 8601)",
      },
      {
        name: "end_time",
        type: "datetime",
        required: false,
        description: "Filter by end time (ISO 8601)",
      },
      {
        name: "top_k",
        type: "number",
        required: false,
        description: "Maximum number of results to return",
      },
      {
        name: "most_recent_k",
        type: "number",
        required: false,
        description: "Return most recent k interactions",
      },
      {
        name: "search_mode",
        type: "enum",
        required: false,
        description:
          "Search mode: vector (embedding similarity), fts (full-text search), or hybrid (combined with RRF)",
        enumValues: ["vector", "fts", "hybrid"],
      },
    ],
  },
  {
    id: "publish-interaction",
    pythonName: "publish_interaction",
    displayName: "Publish Interaction",
    group: "interactions",
    description: "Publish user interactions to be processed by the system.",
    httpMethod: "POST",
    endpoint: "/api/publish_interaction",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: true,
        description: "The user ID",
      },
      {
        name: "interaction_data_list",
        type: "json",
        required: true,
        description:
          'List of interaction data objects, e.g. [{"role": "User", "content": "Hello"}]',
      },
      {
        name: "source",
        type: "string",
        required: false,
        default: "",
        description: "The source of the interaction",
      },
      {
        name: "agent_version",
        type: "string",
        required: false,
        default: "",
        description: "The agent version",
      },
      {
        name: "session_id",
        type: "string",
        required: false,
        description: "Session ID for grouping requests together",
      },
    ],
  },
  {
    id: "delete-interaction",
    pythonName: "delete_interaction",
    displayName: "Delete Interaction",
    group: "interactions",
    description: "Delete a specific user interaction by ID.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_interaction",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: true,
        description: "The user ID",
      },
      {
        name: "interaction_id",
        type: "number",
        required: true,
        description: "The interaction ID to delete",
      },
    ],
  },
];
