import { NAV_GROUPS } from "../constants";
import type { MethodDef, ResourceGroup } from "../types";
import { agentEvaluationMethods } from "./agent-evaluation";
import { agentPlaybookMethods } from "./agent-playbooks";
import { configurationMethods } from "./configuration";
import { generationMethods } from "./generation";
import { interactionMethods } from "./interactions";
import { profileMethods } from "./profiles";
import { requestSessionMethods } from "./requests-sessions";
import { unifiedSearchMethods } from "./unified-search";
import { userPlaybookMethods } from "./user-playbooks";

const methodsByGroup: Record<string, MethodDef[]> = {
	interactions: interactionMethods,
	profiles: profileMethods,
	"requests-sessions": requestSessionMethods,
	"user-playbooks": userPlaybookMethods,
	"agent-playbooks": agentPlaybookMethods,
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
