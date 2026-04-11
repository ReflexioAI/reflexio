import { MethodDef } from "../types";

export const requestSessionMethods: MethodDef[] = [
  {
    id: "get-requests",
    pythonName: "get_requests",
    displayName: "Get Requests",
    group: "requests-sessions",
    description:
      "Get requests with their associated interactions, grouped by session.",
    httpMethod: "POST",
    endpoint: "/api/get_requests",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: false,
        description: "Filter by user ID",
      },
      {
        name: "request_id",
        type: "string",
        required: false,
        description: "Filter by request ID",
      },
      {
        name: "session_id",
        type: "string",
        required: false,
        description: "Filter by session ID",
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
        name: "offset",
        type: "number",
        required: false,
        default: 0,
        description: "Number of results to skip",
      },
    ],
  },
  {
    id: "delete-request",
    pythonName: "delete_request",
    displayName: "Delete Request",
    group: "requests-sessions",
    description: "Delete a request and all its associated interactions.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_request",
    requestStyle: "json_body",
    params: [
      {
        name: "request_id",
        type: "string",
        required: true,
        description: "The request ID to delete",
      },
    ],
  },
  {
    id: "delete-session",
    pythonName: "delete_session",
    displayName: "Delete Session",
    group: "requests-sessions",
    description: "Delete all requests and interactions in a session.",
    httpMethod: "DELETE",
    endpoint: "/api/delete_session",
    requestStyle: "json_body",
    params: [
      {
        name: "session_id",
        type: "string",
        required: true,
        description: "The session ID to delete",
      },
    ],
  },
];
