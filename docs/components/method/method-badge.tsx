import { HttpMethod } from "@/lib/types";
import { cn } from "@/lib/utils";

const badgeStyles: Record<HttpMethod, string> = {
  GET: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
  POST: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
  PUT: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
  DELETE: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
};

export function MethodBadge({
  method,
  size = "sm",
}: {
  method: HttpMethod;
  size?: "sm" | "xs";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center font-mono font-semibold rounded",
        size === "sm" ? "px-2 py-0.5 text-xs" : "px-1.5 py-0.5 text-[10px]",
        badgeStyles[method]
      )}
    >
      {method}
    </span>
  );
}
