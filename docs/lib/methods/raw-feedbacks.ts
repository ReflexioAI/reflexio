import { MethodDef } from "../types";

const STATUS_ENUM = ["current", "archived", "pending", "archive_in_progress"];

export const rawFeedbackMethods: MethodDef[] = [
  {
    id: "get-raw-feedbacks",
    pythonName: "get_raw_feedbacks",
    displayName: "Get Raw Feedbacks",
    group: "raw-feedbacks",
    description: "Get raw feedbacks with optional filtering.",
    httpMethod: "POST",
    endpoint: "/api/get_raw_feedbacks",
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
        name: "feedback_name",
        type: "string",
        required: false,
        description: "Filter by feedback name",
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
    id: "search-raw-feedbacks",
    pythonName: "search_raw_feedbacks",
    displayName: "Search Raw Feedbacks",
    group: "raw-feedbacks",
    description:
      "Search for raw feedbacks with semantic/text search and filtering.",
    httpMethod: "POST",
    endpoint: "/api/search_raw_feedbacks",
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
        description: "Filter by user",
      },
      {
        name: "agent_version",
        type: "string",
        required: false,
        description: "Filter by agent version",
      },
      {
        name: "feedback_name",
        type: "string",
        required: false,
        description: "Filter by feedback name",
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
        default: 0.5,
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
        description:
          "Search mode: vector (embedding similarity), fts (full-text search), or hybrid (combined with RRF)",
        enumValues: ["vector", "fts", "hybrid"],
      },
    ],
  },
  {
    id: "add-raw-feedback",
    pythonName: "add_raw_feedback",
    displayName: "Add Raw Feedback",
    group: "raw-feedbacks",
    description: "Add raw feedback directly to storage.",
    httpMethod: "POST",
    endpoint: "/api/add_raw_feedback",
    requestStyle: "json_body",
    params: [
      {
        name: "raw_feedbacks",
        type: "json",
        required: true,
        description:
          'List of raw feedback objects, e.g. [{"agent_version": "v1", "request_id": "req-1", "feedback_content": "..."}]',
      },
    ],
  },
  {
    id: "delete-raw-feedback",
    pythonName: "delete_raw_feedback",
    displayName: "Delete Raw Feedback",
    group: "raw-feedbacks",
    description: "Delete a raw feedback by ID.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_raw_feedback",
    requestStyle: "json_body",
    params: [
      {
        name: "raw_feedback_id",
        type: "number",
        required: true,
        description: "The raw feedback ID to delete",
      },
    ],
  },
];
