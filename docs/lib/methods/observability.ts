import { MethodDef } from "../types";

const SIGNAL_ENUM = ["stale", "duplicate", "high_cost_low_cite", "supersedeable"];

export const observabilityMethods: MethodDef[] = [
  {
    id: "get-injection-stats",
    pythonName: "get_injection_stats",
    displayName: "Get Injection Stats",
    group: "observability",
    description:
      "Get per-entity learning injection counts and token cost over a look-back window.",
    httpMethod: "POST",
    endpoint: "/api/get_injection_stats",
    requestStyle: "json_body",
    params: [
      {
        name: "days_back",
        type: "number",
        required: false,
        default: 30,
        description: "Look-back window in days",
      },
    ],
  },
  {
    id: "get-memory-review",
    pythonName: "get_memory_review",
    displayName: "Get Memory Review",
    group: "observability",
    description:
      "Get user-scoped memory review candidates. Set include_all_users=true only for explicit org-wide review.",
    httpMethod: "POST",
    endpoint: "/api/get_memory_review",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: false,
        description: "User whose playbooks should be reviewed",
      },
      {
        name: "include_all_users",
        type: "boolean",
        required: false,
        default: false,
        description:
          "Explicit opt-in for org-wide review. Required when user_id is omitted.",
      },
      {
        name: "days_back",
        type: "number",
        required: false,
        default: 60,
        description: "Look-back window in days",
      },
      {
        name: "signal_filter",
        type: "json",
        required: false,
        description: `Optional JSON array of signals to include. Values: ${SIGNAL_ENUM.join(", ")}`,
      },
    ],
  },
];
