import { notFound } from "next/navigation";
import { getMethodById, getMethodsByGroup, resourceGroups } from "@/lib/methods/registry";
import { MethodPage } from "@/components/method/method-page";

interface PageProps {
  params: Promise<{ group: string; method: string }>;
}

export default async function MethodRoute({ params }: PageProps) {
  const { group, method: methodId } = await params;

  const methodDef = getMethodById(methodId);
  if (!methodDef || methodDef.group !== group) {
    notFound();
  }

  return <MethodPage method={methodDef} />;
}

export async function generateStaticParams() {
  const routes: { group: string; method: string }[] = [];
  for (const group of resourceGroups) {
    for (const method of group.methods) {
      routes.push({ group: group.id, method: method.id });
    }
  }
  return routes;
}
