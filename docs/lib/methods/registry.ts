import { MethodDef, ResourceGroup } from "../types";
import { NAV_GROUPS } from "../constants";
import { interactionMethods } from "./interactions";
import { profileMethods } from "./profiles";
import { requestSessionMethods } from "./requests-sessions";
import { rawFeedbackMethods } from "./raw-feedbacks";
import { feedbackMethods } from "./feedbacks";
import { agentEvaluationMethods } from "./agent-evaluation";
import { unifiedSearchMethods } from "./unified-search";
import { generationMethods } from "./generation";
import { configurationMethods } from "./configuration";

const methodsByGroup: Record<string, MethodDef[]> = {
  interactions: interactionMethods,
  profiles: profileMethods,
  "requests-sessions": requestSessionMethods,
  "raw-feedbacks": rawFeedbackMethods,
  feedbacks: feedbackMethods,
  "agent-evaluation": agentEvaluationMethods,
  "unified-search": unifiedSearchMethods,
  generation: generationMethods,
  configuration: configurationMethods,
};

export const allMethods: MethodDef[] = Object.values(methodsByGroup).flat();

export const resourceGroups: ResourceGroup[] = NAV_GROUPS.map((group) => ({
  id: group.id,
  name: group.name,
  icon: group.icon,
  methods: methodsByGroup[group.id] ?? [],
}));

export function getMethodById(id: string): MethodDef | undefined {
  return allMethods.find((m) => m.id === id);
}

export function getMethodsByGroup(groupId: string): MethodDef[] {
  return methodsByGroup[groupId] ?? [];
}

export function getGroupById(groupId: string): ResourceGroup | undefined {
  return resourceGroups.find((g) => g.id === groupId);
}
