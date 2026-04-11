"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  MessageSquare,
  Users,
  FolderOpen,
  MessageCircle,
  ThumbsUp,
  Sparkles,
  BarChart,
  Search,
  Cpu,
  Settings,
  ChevronRight,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { resourceGroups } from "@/lib/methods/registry";
import { MethodBadge } from "@/components/method/method-badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";

const iconMap: Record<string, LucideIcon> = {
  MessageSquare,
  Users,
  FolderOpen,
  MessageCircle,
  ThumbsUp,
  Sparkles,
  BarChart,
  Search,
  Cpu,
  Settings,
};

export function Sidebar() {
  const pathname = usePathname();

  return (
    <ScrollArea className="h-full">
      <div className="px-3 py-4">
        <Link href="/" className="flex items-center gap-2 px-2 mb-6">
          <span className="font-semibold text-sm">Reflexio API</span>
        </Link>

        <div className="space-y-1">
          {resourceGroups.map((group) => (
            <SidebarGroup
              key={group.id}
              group={group}
              pathname={pathname}
            />
          ))}
        </div>
      </div>
    </ScrollArea>
  );
}

function SidebarGroup({
  group,
  pathname,
}: {
  group: (typeof resourceGroups)[0];
  pathname: string;
}) {
  const isActive = pathname.startsWith(`/${group.id}`);
  const [open, setOpen] = useState(isActive);
  const Icon = iconMap[group.icon] ?? MessageSquare;

  return (
    <Collapsible open={open} onOpenChange={(newOpen) => setOpen(newOpen)}>
      <CollapsibleTrigger className="flex items-center gap-2 w-full px-2 py-1.5 text-sm rounded-md hover:bg-accent transition-colors">
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 text-muted-foreground transition-transform",
            open && "rotate-90"
          )}
        />
        <Icon className="h-4 w-4 text-muted-foreground" />
        <span className={cn("font-medium", isActive && "text-foreground")}>
          {group.name}
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="ml-4 pl-3 border-l border-border space-y-0.5 py-1">
          {group.methods.map((method) => {
            const href = `/${group.id}/${method.id}`;
            const active = pathname === href;
            return (
              <Link
                key={method.id}
                href={href}
                className={cn(
                  "flex items-center gap-2 px-2 py-1.5 text-sm rounded-md transition-colors",
                  active
                    ? "bg-accent text-accent-foreground font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                )}
              >
                <MethodBadge method={method.httpMethod} size="xs" />
                <span className="truncate">{method.displayName}</span>
              </Link>
            );
          })}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
