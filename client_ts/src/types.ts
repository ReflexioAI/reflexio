// Friendly names for the generated server schemas the client exposes. Every
// type here is the server's own (src/generated/openapi.ts), so a field the
// server renames fails the type check instead of silently losing data.

import type { components } from "./generated/openapi.js";

type Schemas = components["schemas"];

/** One turn of a conversation: a user message, an agent reply, its tool calls. */
export type Interaction = Schemas["InteractionData"];
export type ToolUsed = Schemas["ToolUsed"];
export type Citation = Schemas["Citation"];
export type RetrievedLearning = Schemas["RetrievedLearning"];

export type PublishInteractionResponse = Schemas["PublishUserInteractionResponse"];
export type SearchRequest = Schemas["UnifiedSearchRequest"];
export type SearchResponse = Schemas["UnifiedSearchViewResponse"];
export type FinalizeSessionRequest = Schemas["FinalizeSessionRequest"];
export type FinalizeSessionResponse = Schemas["FinalizeSessionResponse"];
export type LearningStatusResponse = Schemas["LearningStatusResponse"];
export type WhoamiResponse = Schemas["WhoamiResponse"];

export type GradeOnDemandRequest = Schemas["GradeOnDemandRequest"];
export type GradeOnDemandResponse = Schemas["GradeOnDemandResponse"];
export type GetInteractionsRequest = Schemas["GetInteractionsRequest"];
export type GetInteractionsResponse = Schemas["GetInteractionsViewResponse"];
export type GetProfilesRequest = Schemas["GetUserProfilesRequest"];
export type GetProfilesResponse = Schemas["GetProfilesViewResponse"];
export type GetUserPlaybooksRequest = Schemas["GetUserPlaybooksRequest"];
export type GetUserPlaybooksResponse = Schemas["GetUserPlaybooksViewResponse"];
export type GetAgentPlaybooksRequest = Schemas["GetAgentPlaybooksRequest"];
export type GetAgentPlaybooksResponse = Schemas["GetAgentPlaybooksViewResponse"];

export type Profile = Schemas["ProfileView"];
export type UserPlaybook = Schemas["UserPlaybookView"];
export type AgentPlaybook = Schemas["AgentPlaybookView"];

/**
 * Arguments to `publishInteraction`. The same fields as the REST body, except
 * the turns go in `interactions` (as in the Python client) and
 * `wait_for_response` asks the server to finish extraction before answering.
 *
 * There is no `request_id`, as in the Python client: the server does not
 * deduplicate atomically, so a caller-chosen ID would make a replay after a
 * timeout look safe when it can duplicate the batch.
 */
export type PublishInteractionParams = Omit<
  Schemas["PublishUserInteractionRequest"],
  "interaction_data_list" | "request_id"
> & {
  interactions: Interaction[];
  wait_for_response?: boolean;
};

/** Per-call overrides. */
export interface RequestOptions {
  /** Overrides the client's `timeoutMs` for this call. */
  timeoutMs?: number;
  /** Cancels the call when aborted. */
  signal?: AbortSignal;
}
