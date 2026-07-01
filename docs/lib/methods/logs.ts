import { MethodDef } from "../types";

export const logMethods: MethodDef[] = [
  {
    id: "get-logs",
    pythonName: "get_logs",
    pythonCodeStyle: "requests",
    displayName: "Get Logs",
    group: "logs",
    description:
      "Retrieve recent OSS structured warning, error, and critical log events.",
    httpMethod: "GET",
    endpoint: "/api/logs",
    requestStyle: "query_params",
    params: [
      {
        name: "levels",
        type: "string",
        required: false,
        description:
          "Comma-separated levels to include. Supported values: warning, error, critical. Defaults to error,critical.",
      },
      {
        name: "since",
        type: "string",
        required: false,
        description:
          "Relative window such as 24h or 7d, or an ISO 8601 timestamp.",
      },
      {
        name: "q",
        type: "string",
        required: false,
        description:
          "Literal substring search across message, traceback, and logger name. Maximum 256 characters.",
      },
      {
        name: "limit",
        type: "number",
        required: false,
        description: "Maximum number of events to return, from 1 to 1000.",
      },
    ],
  },
];
