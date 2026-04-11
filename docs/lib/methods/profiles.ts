import { MethodDef } from "../types";

const STATUS_ENUM = ["current", "archived", "pending", "archive_in_progress"];

export const profileMethods: MethodDef[] = [
  {
    id: "get-all-profiles",
    pythonName: "get_all_profiles",
    displayName: "Get All Profiles",
    group: "profiles",
    description: "Get all user profiles across all users.",
    httpMethod: "GET",
    endpoint: "/api/get_all_profiles",
    requestStyle: "query_params",
    params: [
      {
        name: "limit",
        type: "number",
        required: false,
        default: 100,
        description: "Maximum number of profiles to return",
      },
    ],
  },
  {
    id: "get-profiles",
    pythonName: "get_profiles",
    displayName: "Get Profiles",
    group: "profiles",
    description:
      "Get user profiles filtered by user ID, time range, status, and result count.",
    httpMethod: "POST",
    endpoint: "/api/get_profiles",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: true,
        description: "The user ID to get profiles for",
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
      {
        name: "status_filter",
        type: "json",
        required: false,
        description: `Status filter as JSON array, e.g. [null, "archived"]. Values: ${STATUS_ENUM.join(", ")}, null (for current)`,
      },
    ],
  },
  {
    id: "get-profile-change-log",
    pythonName: "get_profile_change_log",
    displayName: "Get Profile Change Log",
    group: "profiles",
    description:
      "Get the profile change log showing added, removed, and mentioned profiles.",
    httpMethod: "GET",
    endpoint: "/api/profile_change_log",
    requestStyle: "no_body",
    params: [],
  },
  {
    id: "search-profiles",
    pythonName: "search_profiles",
    displayName: "Search Profiles",
    group: "profiles",
    description:
      "Search for user profiles with semantic/text search and filtering.",
    httpMethod: "POST",
    endpoint: "/api/search_profiles",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: true,
        description: "The user ID to search for",
      },
      {
        name: "generated_from_request_id",
        type: "string",
        required: false,
        description: "Filter by request ID that generated the profile",
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
        default: 10,
        description: "Maximum number of results to return",
      },
      {
        name: "source",
        type: "string",
        required: false,
        description: "Filter by source",
      },
      {
        name: "custom_feature",
        type: "string",
        required: false,
        description: "Filter by custom feature",
      },
      {
        name: "extractor_name",
        type: "string",
        required: false,
        description: "Filter by extractor name",
      },
      {
        name: "threshold",
        type: "number",
        required: false,
        default: 0.5,
        description: "Similarity threshold (0.0 to 1.0)",
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
    id: "delete-profile",
    pythonName: "delete_profile",
    displayName: "Delete Profile",
    group: "profiles",
    description:
      "Delete user profiles by user ID, profile ID, or search query.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_profile",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: true,
        description: "The user ID",
      },
      {
        name: "profile_id",
        type: "string",
        required: false,
        default: "",
        description: "Specific profile ID to delete",
      },
      {
        name: "search_query",
        type: "string",
        required: false,
        default: "",
        description: "Query to match profiles for deletion",
      },
    ],
  },
];
