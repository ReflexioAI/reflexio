export type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";

export type ParamType =
  | "string"
  | "number"
  | "boolean"
  | "datetime"
  | "string[]"
  | "enum"
  | "json";

export interface ParamDef {
  name: string;
  type: ParamType;
  required: boolean;
  default?: unknown;
  description: string;
  enumValues?: string[];
}

export interface MethodDef {
  id: string;
  pythonName: string;
  displayName: string;
  group: string;
  description: string;
  httpMethod: HttpMethod;
  endpoint: string;
  requestStyle: "json_body" | "query_params" | "no_body";
  params: ParamDef[];
  // When set, the body is the JSON value of the named param (sent "as-is"),
  // not an object of {param: value}. Used for endpoints like /api/set_config
  // where FastAPI expects the top-level model directly.
  bodyFromParam?: string;
}

export interface ResourceGroup {
  id: string;
  name: string;
  icon: string;
  methods: MethodDef[];
}
