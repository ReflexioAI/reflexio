import { MethodDef } from "../types";

const STATUS_ENUM = ["current", "archived", "pending", "archive_in_progress"];
const FEEDBACK_STATUS_ENUM = ["pending", "approved", "rejected"];

export const feedbackMethods: MethodDef[] = [
  {
    id: "get-feedbacks",
    pythonName: "get_feedbacks",
    displayName: "Get Feedbacks",
    group: "feedbacks",
    description: "Get aggregated feedbacks with optional filtering.",
    httpMethod: "POST",
    endpoint: "/api/get_feedbacks",
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
      {
        name: "feedback_status_filter",
        type: "enum",
        required: false,
        description: "Filter by feedback status",
        enumValues: FEEDBACK_STATUS_ENUM,
      },
    ],
  },
  {
    id: "search-feedbacks",
    pythonName: "search_feedbacks",
    displayName: "Search Feedbacks",
    group: "feedbacks",
    description:
      "Search for aggregated feedbacks with semantic/text search and filtering.",
    httpMethod: "POST",
    endpoint: "/api/search_feedbacks",
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
        name: "feedback_status_filter",
        type: "enum",
        required: false,
        description: "Filter by feedback status",
        enumValues: FEEDBACK_STATUS_ENUM,
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
    id: "add-feedbacks",
    pythonName: "add_feedbacks",
    displayName: "Add Feedbacks",
    group: "feedbacks",
    description: "Add aggregated feedback directly to storage.",
    httpMethod: "POST",
    endpoint: "/api/add_feedbacks",
    requestStyle: "json_body",
    params: [
      {
        name: "feedbacks",
        type: "json",
        required: true,
        description:
          'List of feedback objects, e.g. [{"agent_version": "v1", "feedback_content": "...", "feedback_status": "pending"}]',
      },
    ],
  },
  {
    id: "delete-feedback",
    pythonName: "delete_feedback",
    displayName: "Delete Feedback",
    group: "feedbacks",
    description: "Delete an aggregated feedback by ID.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_feedback",
    requestStyle: "json_body",
    params: [
      {
        name: "feedback_id",
        type: "number",
        required: true,
        description: "The feedback ID to delete",
      },
    ],
  },
];
