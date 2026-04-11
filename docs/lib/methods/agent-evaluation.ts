import { MethodDef } from "../types";

export const agentEvaluationMethods: MethodDef[] = [
  {
    id: "get-agent-success-evaluation-results",
    pythonName: "get_agent_success_evaluation_results",
    displayName: "Get Agent Success Evaluation Results",
    group: "agent-evaluation",
    description: "Get agent success evaluation results with optional filtering.",
    httpMethod: "POST",
    endpoint: "/api/get_agent_success_evaluation_results",
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
        name: "agent_version",
        type: "string",
        required: false,
        description: "Filter by agent version",
      },
    ],
  },
];
