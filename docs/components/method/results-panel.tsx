"use client";

import { Copy, Check, Loader2, Table, Braces, ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ExecutionResult } from "@/lib/execution/api-executor";
import { JsonView } from "./json-view";

interface ResultsPanelProps {
  result: ExecutionResult | null;
  loading: boolean;
  error: string | null;
}

// Keys to skip in the top-level summary (shown in table instead)
const ARRAY_KEYS_TO_TABLE = new Set([
  "interactions",
  "user_profiles",
  "feedbacks",
  "raw_feedbacks",
  "sessions",
  "agent_success_evaluation_results",
  "profile_change_logs",
  "profiles",
  "change_logs",
]);

// Columns to hide in tables (too long / not useful in overview)
const HIDDEN_COLUMNS = new Set([
  "embedding",
  "image_encoding",
  "interacted_image_url",
]);

// Long text columns get truncated
const MAX_CELL_LENGTH = 120;

function findTableData(data: unknown): { key: string; rows: Record<string, unknown>[] } | null {
  if (!data || typeof data !== "object" || Array.isArray(data)) return null;

  const obj = data as Record<string, unknown>;
  for (const key of Object.keys(obj)) {
    if (!ARRAY_KEYS_TO_TABLE.has(key)) continue;
    const val = obj[key];
    if (Array.isArray(val) && val.length > 0 && typeof val[0] === "object" && val[0] !== null) {
      return { key, rows: val as Record<string, unknown>[] };
    }
  }
  return null;
}

function getColumns(rows: Record<string, unknown>[]): string[] {
  const colSet = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!HIDDEN_COLUMNS.has(key)) colSet.add(key);
    }
  }
  return Array.from(colSet);
}

function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") {
    if (value.length > MAX_CELL_LENGTH) return value.slice(0, MAX_CELL_LENGTH) + "...";
    return value;
  }
  const str = JSON.stringify(value);
  if (str.length > MAX_CELL_LENGTH) return str.slice(0, MAX_CELL_LENGTH) + "...";
  return str;
}

function formatTimestamp(value: unknown): string {
  if (typeof value !== "number") return formatCellValue(value);
  // Unix timestamps (seconds) are > 1e9 and < 1e11
  if (value > 1e9 && value < 1e11) {
    return new Date(value * 1000).toLocaleString();
  }
  return String(value);
}

const TIMESTAMP_COLUMNS = new Set([
  "created_at",
  "last_modified_timestamp",
  "expiration_timestamp",
  "started_at",
  "completed_at",
  "updated_at",
]);

export function ResultsPanel({ result, loading, error }: ResultsPanelProps) {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState<"table" | "json">("table");

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full min-h-[200px]">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Executing...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="rounded-md bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-3">
          <p className="text-sm text-red-700 dark:text-red-400 font-medium">
            Error
          </p>
          <p className="text-xs text-red-600 dark:text-red-300 mt-1">
            {error}
          </p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full min-h-[200px]">
        <div className="text-center">
          <p className="text-sm text-muted-foreground">No results yet</p>
          <p className="text-xs text-muted-foreground/70 mt-1">
            Click Run or press Cmd+Enter to execute
          </p>
        </div>
      </div>
    );
  }

  const jsonStr = JSON.stringify(result.data, null, 2);
  const tableData = findTableData(result.data);
  const hasTable = tableData !== null && tableData.rows.length > 0;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(jsonStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/30">
        <div className="flex items-center gap-3">
          <StatusBadge status={result.status} />
          <span className="text-xs text-muted-foreground">
            {result.duration.toFixed(0)}ms
          </span>
          {hasTable && (
            <span className="text-xs text-muted-foreground">
              {tableData.rows.length} {tableData.key.replace(/_/g, " ")}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {hasTable && (
            <div className="flex items-center border border-border rounded-md overflow-hidden mr-1">
              <button
                onClick={() => setViewMode("table")}
                className={cn(
                  "px-2 py-1 text-xs flex items-center gap-1 transition-colors",
                  viewMode === "table"
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <Table className="h-3 w-3" />
                Table
              </button>
              <button
                onClick={() => setViewMode("json")}
                className={cn(
                  "px-2 py-1 text-xs flex items-center gap-1 transition-colors",
                  viewMode === "json"
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <Braces className="h-3 w-3" />
                JSON
              </button>
            </div>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleCopy}
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-green-500" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
          </Button>
        </div>
      </div>

      {/* Content */}
      {hasTable && viewMode === "table" ? (
        <div className="flex-1 overflow-auto">
          <MetaSummary data={result.data as Record<string, unknown>} tableKey={tableData.key} />
          <ResultTable rows={tableData.rows} />
        </div>
      ) : (
        <JsonView json={jsonStr} />
      )}
    </div>
  );
}

function MetaSummary({ data, tableKey }: { data: Record<string, unknown>; tableKey: string }) {
  const metaEntries = Object.entries(data).filter(
    ([key]) => key !== tableKey && !ARRAY_KEYS_TO_TABLE.has(key)
  );
  if (metaEntries.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 px-4 py-2 border-b border-border bg-muted/10">
      {metaEntries.map(([key, value]) => (
        <span key={key} className="text-xs">
          <span className="text-muted-foreground">{key}:</span>{" "}
          <span className="font-medium">
            {typeof value === "boolean"
              ? value
                ? "true"
                : "false"
              : String(value ?? "-")}
          </span>
        </span>
      ))}
    </div>
  );
}

function ResultTable({ rows }: { rows: Record<string, unknown>[] }) {
  const columns = getColumns(rows);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  return (
    <table className="w-full text-xs">
      <thead className="sticky top-0 bg-muted/50 backdrop-blur z-10">
        <tr>
          <th className="w-8 px-2 py-2" />
          {columns.map((col) => (
            <th
              key={col}
              className="px-3 py-2 text-left font-medium text-muted-foreground whitespace-nowrap border-b border-border"
            >
              {col}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <TableRow
            key={i}
            row={row}
            columns={columns}
            index={i}
            expanded={expandedRow === i}
            onToggle={() => setExpandedRow(expandedRow === i ? null : i)}
          />
        ))}
      </tbody>
    </table>
  );
}

function TableRow({
  row,
  columns,
  index,
  expanded,
  onToggle,
}: {
  row: Record<string, unknown>;
  columns: string[];
  index: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        className={cn(
          "border-b border-border hover:bg-muted/30 cursor-pointer transition-colors",
          index % 2 === 0 ? "bg-transparent" : "bg-muted/10",
          expanded && "bg-accent/30"
        )}
        onClick={onToggle}
      >
        <td className="px-2 py-1.5 text-muted-foreground">
          {expanded ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
        </td>
        {columns.map((col) => (
          <td
            key={col}
            className="px-3 py-1.5 max-w-[300px] truncate font-mono"
          >
            {TIMESTAMP_COLUMNS.has(col)
              ? formatTimestamp(row[col])
              : formatCellValue(row[col])}
          </td>
        ))}
      </tr>
      {expanded && (
        <tr>
          <td colSpan={columns.length + 1} className="p-0">
            <div className="bg-muted/20 border-b border-border max-h-[400px] overflow-auto">
              <JsonView json={JSON.stringify(row, null, 2)} />
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function StatusBadge({ status }: { status: number }) {
  const isSuccess = status >= 200 && status < 300;
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-semibold",
        isSuccess
          ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400"
          : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
      )}
    >
      {status}
    </span>
  );
}
