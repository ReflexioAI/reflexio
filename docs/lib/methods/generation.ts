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
    id: "rerun-playbook-generation",
    pythonName: "rerun_playbook_generation",
    displayName: "Rerun Playbook Generation",
    group: "generation",
    description:
      "Rerun user playbook generation for an agent version. Uses ALL interactions and outputs PENDING status user playbooks. Aggregation into agent playbooks is a separate step.",
    httpMethod: "POST",
    endpoint: "/api/rerun_playbook_generation",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_version",
        type: "string",
        required: false,
        description: "The agent version to generate playbooks for",
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
        name: "playbook_name",
        type: "string",
        required: false,
        description: "Specific playbook name/category to regenerate",
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
    id: "manual-playbook-generation",
    pythonName: "manual_playbook_generation",
    displayName: "Manual Playbook Generation",
    group: "generation",
    description:
      "Manually trigger user playbook generation with window-sized interactions. Outputs CURRENT status user playbooks.",
    httpMethod: "POST",
    endpoint: "/api/manual_playbook_generation",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_version",
        type: "string",
        required: false,
        description: "The agent version to generate playbooks for",
      },
      {
        name: "source",
        type: "string",
        required: false,
        description: "Filter interactions by source",
      },
      {
        name: "playbook_name",
        type: "string",
        required: false,
        description: "Specific playbook name/category to generate",
      },
    ],
  },
  {
    id: "run-playbook-aggregation",
    pythonName: "run_playbook_aggregation",
    displayName: "Run Playbook Aggregation",
    group: "generation",
    description:
      "Run playbook aggregation to cluster user playbooks into aggregated agent playbooks for the given name.",
    httpMethod: "POST",
    endpoint: "/api/run_playbook_aggregation",
    requestStyle: "json_body",
    params: [
      {
        name: "agent_version",
        type: "string",
        required: false,
        description: "The agent version",
      },
      {
        name: "playbook_name",
        type: "string",
        required: true,
        description: "The playbook name/category to aggregate",
      },
    ],
  },
];
