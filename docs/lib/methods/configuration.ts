import { MethodDef } from "../types";

export const configurationMethods: MethodDef[] = [
  {
    id: "get-config",
    pythonName: "get_config",
    displayName: "Get Config",
    group: "configuration",
    description: "Get the current configuration for the organization.",
    httpMethod: "GET",
    endpoint: "/api/get_config",
    requestStyle: "no_body",
    params: [],
  },
  {
    id: "set-config",
    pythonName: "set_config",
    displayName: "Set Config",
    group: "configuration",
    description: "Set configuration for the organization.",
    httpMethod: "POST",
    endpoint: "/api/set_config",
    requestStyle: "json_body",
    bodyFromParam: "config",
    params: [
      {
        name: "config",
        type: "json",
        required: true,
        description:
          "The configuration object. Get current config first, then modify and send back.",
      },
    ],
  },
];
