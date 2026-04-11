import { MethodDef } from "../types";

export const generationMethods: MethodDef[] = [
  {
    id: "rerun-profile-generation",
    pythonName: "rerun_profile_generation",
    displayName: "Rerun Profile Generation",
    group: "generation",
    description:
      "Rerun profile generation for users. Uses ALL interactions and outputs PENDING status profiles.",
    httpMethod: "POST",
    endpoint: "/api/rerun_profile_generation",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: false,
        description:
          "Specific user ID to rerun for. If omitted, runs for all users.",
      },
      {
        name: "start_time",
        type: "datetime",
        required: false,
        description: "Filter interactions by start time (ISO 8601)",
      },
      {
        name: "end_time",
        type: "datetime",
        required: false,
        description: "Filter interactions by end time (ISO 8601)",
      },
      {
        name: "source",
        type: "string",
        required: false,
        description: "Filter interactions by source",
      },
      {
        name: "extractor_names",
        type: "json",
        required: false,
        description:
          'List of extractor names to run, e.g. ["extractor1", "extractor2"]. If omitted, runs all.',
      },
    ],
  },
  {
    id: "manual-profile-generation",
    pythonName: "manual_profile_generation",
    displayName: "Manual Profile Generation",
    group: "generation",
    description:
      "Manually trigger profile generation with window-sized interactions. Outputs CURRENT status profiles.",
    httpMethod: "POST",
    endpoint: "/api/manual_profile_generation",
    requestStyle: "json_body",
    params: [
      {
        name: "user_id",
        type: "string",
        required: false,
        description: "Specific user ID to generate for. If omitted, all users.",
      },
      {
        name: "source",
        type: "string",
        required: false,
        description: "Filter interactions by source",
      },
      {
        name: "extractor_names",
        type: "json",
        required: false,
        description:
          'List of extractor names to run, e.g. ["extractor1", "extractor2"]',
      },
    ],
  },
  {
    id: "rerun-feedback-generation",
    pythonName: "rerun_feedback_generation",
    displayName: "Rerun Feedback Generation",
    group: "generation",
    description:
      "Rerun feedback generation for an agent version. Uses ALL interactions and outputs PENDING status feedbacks.",
    httpMethod: "POST",
    endpoint: "/api/rerun_feedback_generation",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_version",
        type: "string",
        required: true,
        description: "The agent version to evaluate",
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
        name: "feedback_name",
        type: "string",
        required: false,
        description: "Specific feedback type to generate",
      },
      {
        name: "source",
        type: "string",
        required: false,
        description: "Filter interactions by source",
      },
    ],
  },
  {
    id: "manual-feedback-generation",
    pythonName: "manual_feedback_generation",
    displayName: "Manual Feedback Generation",
    group: "generation",
    description:
      "Manually trigger feedback generation with window-sized interactions. Outputs CURRENT status feedbacks.",
    httpMethod: "POST",
    endpoint: "/api/manual_feedback_generation",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_version",
        type: "string",
        required: true,
        description: "The agent version to evaluate",
      },
      {
        name: "source",
        type: "string",
        required: false,
        description: "Filter interactions by source",
      },
      {
        name: "feedback_name",
        type: "string",
        required: false,
        description: "Specific feedback type to generate",
      },
    ],
  },
  {
    id: "run-feedback-aggregation",
    pythonName: "run_feedback_aggregation",
    displayName: "Run Feedback Aggregation",
    group: "generation",
    description:
      "Run feedback aggregation to cluster similar raw feedbacks into aggregated feedbacks.",
    httpMethod: "POST",
    endpoint: "/api/run_feedback_aggregation",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_version",
        type: "string",
        required: true,
        description: "The agent version",
      },
      {
        name: "feedback_name",
        type: "string",
        required: true,
        description: "The feedback type to aggregate",
      },
    ],
  },
];
