"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowDownUp, Logs, RefreshCw, Search, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSettings } from "@/hooks/use-settings";
import { cn } from "@/lib/utils";

const LOG_LEVELS = ["warning", "error", "critical"] as const;
const TIME_WINDOWS = ["1h", "24h", "all"] as const;
const DEFAULT_LEVELS = ["error", "critical"] as const;
const DEFAULT_WINDOW = "24h" as const;
const DEFAULT_SORT = "newest" as const;
const RESULT_LIMIT = 200;

type LogLevel = (typeof LOG_LEVELS)[number];
type TimeWindow = (typeof TIME_WINDOWS)[number];
type SortOrder = "newest" | "oldest";

interface LogItem {
  timestamp: string;
  level: LogLevel;
  logger_name: string;
  message: string;
  exception_text: string | null;
}

interface LogsResponse {
  items: LogItem[];
  limit: number;
}

interface RequestErrorState {
  status: number | null;
  message: string;
}

const levelStyles: Record<LogLevel, { badge: string; filter: string; label: string }> = {
  warning: {
    badge:
      "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-300",
    filter:
      "border-amber-300 bg-amber-50 text-amber-700 hover:bg-amber-100 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-300 dark:hover:bg-amber-950/50",
    label: "Warning",
  },
  error: {
    badge:
      "border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/30 dark:text-rose-300",
    filter:
      "border-rose-300 bg-rose-50 text-rose-700 hover:bg-rose-100 dark:border-rose-900/60 dark:bg-rose-950/30 dark:text-rose-300 dark:hover:bg-rose-950/50",
    label: "Error",
  },
  critical: {
    badge:
      "border-red-700 bg-red-50 text-red-800 dark:border-red-500/70 dark:bg-red-950/50 dark:text-red-200",
    filter:
      "border-red-700 bg-red-50 text-red-800 hover:bg-red-100 dark:border-red-500/70 dark:bg-red-950/50 dark:text-red-200 dark:hover:bg-red-950/70 font-semibold",
    label: "Critical",
  },
};

function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

async function parseResponse(response: Response) {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function getErrorMessage(payload: unknown, fallback: string) {
  if (typeof payload === "string" && payload.trim()) {
    return payload;
  }

  if (!payload || typeof payload !== "object") {
    return fallback;
  }

  if ("detail" in payload && typeof payload.detail === "string") {
    const detail = payload.detail.trim();
    if (detail) {
      return detail;
    }
  }

  if ("message" in payload && typeof payload.message === "string") {
    const message = payload.message.trim();
    if (message) {
      return message;
    }
  }

  return fallback;
}

function getWindowLabel(window: TimeWindow) {
  if (window === "all") {
    return "All";
  }
  return window;
}

function getSortLabel(order: SortOrder) {
  return order === "newest" ? "Newest first" : "Oldest first";
}

export function LogsDashboard() {
  const { apiEndpoint } = useSettings();
  const [selectedLevels, setSelectedLevels] = useState<LogLevel[]>([
    ...DEFAULT_LEVELS,
  ]);
  const [timeWindow, setTimeWindow] = useState<TimeWindow>(DEFAULT_WINDOW);
  const [sortOrder, setSortOrder] = useState<SortOrder>(DEFAULT_SORT);
  const [searchValue, setSearchValue] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [items, setItems] = useState<LogItem[]>([]);
  const [responseLimit, setResponseLimit] = useState(RESULT_LIMIT);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<RequestErrorState | null>(null);
  const [refreshNonce, setRefreshNonce] = useState(0);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      setDebouncedSearch(searchValue.trim());
    }, 250);

    return () => window.clearTimeout(timeout);
  }, [searchValue]);

  const levelsParam = useMemo(
    () => LOG_LEVELS.filter((level) => selectedLevels.includes(level)).join(","),
    [selectedLevels]
  );

  const requestUrl = useMemo(() => {
    const baseUrl = apiEndpoint.replace(/\/$/, "");
    const params = new URLSearchParams();

    if (levelsParam) {
      params.set("levels", levelsParam);
    }
    if (timeWindow !== "all") {
      params.set("since", timeWindow);
    }
    if (debouncedSearch) {
      params.set("q", debouncedSearch);
    }
    params.set("limit", String(RESULT_LIMIT));

    return `${baseUrl}/api/logs?${params.toString()}`;
  }, [apiEndpoint, debouncedSearch, levelsParam, timeWindow]);

  const displayItems = useMemo(() => {
    const sorted = [...items].sort((left, right) => {
      const leftTime = new Date(left.timestamp).getTime();
      const rightTime = new Date(right.timestamp).getTime();
      return sortOrder === "newest"
        ? rightTime - leftTime
        : leftTime - rightTime;
    });
    return sorted;
  }, [items, sortOrder]);

  const loadLogs = useCallback(
    async (signal: AbortSignal) => {
      setLoading(true);
      setError(null);
      setItems([]);
      setResponseLimit(RESULT_LIMIT);

      try {
        const response = await fetch(requestUrl, { signal });
        const payload = await parseResponse(response);

        if (!response.ok) {
          setError({
            status: response.status,
            message: getErrorMessage(payload, "Failed to load logs."),
          });
          return;
        }

        const data = payload as Partial<LogsResponse> | null;
        setItems(Array.isArray(data?.items) ? data.items : []);
        setResponseLimit(
          typeof data?.limit === "number" ? data.limit : RESULT_LIMIT
        );
      } catch (requestError) {
        if (
          requestError instanceof DOMException &&
          requestError.name === "AbortError"
        ) {
          return;
        }

        setError({
          status: null,
          message:
            requestError instanceof Error
              ? requestError.message
              : "Failed to load logs.",
        });
      } finally {
        if (!signal.aborted) {
          setLoading(false);
        }
      }
    },
    [requestUrl]
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadLogs(controller.signal);

    return () => controller.abort();
  }, [loadLogs, refreshNonce]);

  const toggleLevel = useCallback((level: LogLevel) => {
    setSelectedLevels((current) => {
      if (current.includes(level)) {
        return current.length > 1
          ? current.filter((candidate) => candidate !== level)
          : current;
      }

      return [...current, level];
    });
  }, []);

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 border-b border-border px-6 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className="inline-flex size-8 items-center justify-center rounded-lg border border-border bg-muted/50">
                <Logs className="size-4 text-muted-foreground" />
              </span>
              <div>
                <h1 className="text-lg font-semibold">Logs</h1>
                <code className="rounded bg-muted px-2 py-0.5 font-mono text-xs text-muted-foreground">
                  GET /api/logs
                </code>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              {items.length} rows
              {responseLimit ? ` / ${responseLimit}` : ""}
            </p>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setRefreshNonce((value) => value + 1)}
            disabled={loading}
          >
            <RefreshCw
              className={cn("size-4", loading && "animate-spin")}
            />
            Refresh
          </Button>
        </div>
      </div>

      <div className="flex-1 min-h-0 p-4">
        <div className="flex h-full min-h-0 flex-col rounded-lg border border-border bg-background">
          <div className="shrink-0 border-b border-border px-4 py-3">
            <div className="flex flex-wrap items-center gap-2">
              {LOG_LEVELS.map((level) => {
                const active = selectedLevels.includes(level);
                return (
                  <Button
                    key={level}
                    type="button"
                    variant="outline"
                    size="xs"
                    aria-pressed={active}
                    onClick={() => toggleLevel(level)}
                    className={cn(
                      "capitalize",
                      active && levelStyles[level].filter
                    )}
                  >
                    {levelStyles[level].label}
                  </Button>
                );
              })}

              <Select
                value={timeWindow}
                onValueChange={(value) => setTimeWindow(value as TimeWindow)}
              >
                <SelectTrigger
                  className="h-7 min-w-20 text-xs"
                  aria-label="Time window"
                >
                  <SelectValue placeholder="Window" />
                </SelectTrigger>
                <SelectContent>
                  {TIME_WINDOWS.map((window) => (
                    <SelectItem key={window} value={window} className="text-xs">
                      {getWindowLabel(window)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Button
                type="button"
                variant="outline"
                size="xs"
                aria-label={`Sort logs: ${getSortLabel(sortOrder)}`}
                onClick={() =>
                  setSortOrder((current) =>
                    current === "newest" ? "oldest" : "newest"
                  )
                }
              >
                <ArrowDownUp className="size-3.5" />
                {getSortLabel(sortOrder)}
              </Button>

              <div className="relative min-w-52 flex-1">
                <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={searchValue}
                  onChange={(event) => setSearchValue(event.target.value)}
                  aria-label="Search logs"
                  placeholder="Search logs"
                  className="h-7 pl-8 text-xs"
                />
              </div>
            </div>

            {error && (
              <div className="mt-3 flex items-start gap-2 rounded-md border border-destructive/20 bg-destructive/5 px-3 py-2 text-sm text-destructive">
                <TriangleAlert className="mt-0.5 size-4 shrink-0" />
                <div className="min-w-0">
                  <div className="font-medium">
                    API failure{error.status ? ` (${error.status})` : ""}
                  </div>
                  <p className="break-words">{error.message}</p>
                </div>
              </div>
            )}
          </div>

          <div className="min-h-0 flex-1 overflow-auto">
            <table className="w-full min-w-[720px] table-fixed text-sm">
              <thead className="sticky top-0 z-10 bg-background">
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="w-52 px-4 py-2 font-medium">Timestamp</th>
                  <th className="w-28 px-4 py-2 font-medium">Level</th>
                  <th className="px-4 py-2 font-medium">Message</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td
                      colSpan={3}
                      className="px-4 py-10 text-center text-sm text-muted-foreground"
                    >
                      <div className="inline-flex items-center gap-2">
                        <RefreshCw className="size-4 animate-spin" />
                        Loading logs...
                      </div>
                    </td>
                  </tr>
                ) : items.length === 0 ? (
                  <tr>
                    <td
                      colSpan={3}
                      className="px-4 py-10 text-center text-sm text-muted-foreground"
                    >
                      {error ? "Unable to load log entries." : "No log entries found."}
                    </td>
                  </tr>
                ) : (
                  displayItems.map((item, index) => {
                    return (
                      <tr
                        key={`${item.timestamp}:${index}`}
                        className="border-b border-border align-top last:border-b-0"
                      >
                        <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                          <span title={item.timestamp}>
                            {formatTimestamp(item.timestamp)}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className={cn(
                              "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium",
                              levelStyles[item.level].badge
                            )}
                          >
                            {levelStyles[item.level].label}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <p className="break-words text-sm leading-5">
                            {item.message}
                          </p>
                          {item.exception_text && (
                            <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted px-3 py-2 text-xs">
                              {item.exception_text}
                            </pre>
                          )}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
