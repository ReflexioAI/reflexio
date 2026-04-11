"use client";

import { ParamDef } from "@/lib/types";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ParamFormProps {
  params: ParamDef[];
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
}

export function ParamForm({ params, values, onChange }: ParamFormProps) {
  if (params.length === 0) {
    return (
      <p className="text-xs text-muted-foreground italic px-1">
        No parameters required
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {params.map((param) => (
        <ParamField
          key={param.name}
          param={param}
          value={values[param.name]}
          onChange={(value) => onChange(param.name, value)}
        />
      ))}
    </div>
  );
}

function ParamField({
  param,
  value,
  onChange,
}: {
  param: ParamDef;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const id = `param-${param.name}`;

  if (param.type === "boolean") {
    return (
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor={id} className="text-xs">
            {param.name}
            {param.required && <span className="text-red-500 ml-0.5">*</span>}
          </Label>
          <p className="text-[10px] text-muted-foreground">
            {param.description}
          </p>
        </div>
        <Switch
          id={id}
          checked={value === true || value === "true"}
          onCheckedChange={(checked) => onChange(checked)}
        />
      </div>
    );
  }

  if (param.type === "enum" && param.enumValues) {
    return (
      <div className="space-y-1">
        <Label htmlFor={id} className="text-xs">
          {param.name}
          {param.required && <span className="text-red-500 ml-0.5">*</span>}
        </Label>
        <Select
          value={(value as string) ?? null}
          onValueChange={(v) => onChange(v)}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue placeholder="Select..." />
          </SelectTrigger>
          <SelectContent>
            {param.enumValues.map((ev) => (
              <SelectItem key={ev} value={ev} className="text-xs">
                {ev}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-[10px] text-muted-foreground">
          {param.description}
        </p>
      </div>
    );
  }

  if (param.type === "json") {
    return (
      <div className="space-y-1">
        <Label htmlFor={id} className="text-xs">
          {param.name}
          {param.required && <span className="text-red-500 ml-0.5">*</span>}
        </Label>
        <textarea
          id={id}
          value={(value as string) ?? ""}
          onChange={(e) => onChange(e.target.value)}
          placeholder={param.description}
          rows={3}
          className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-xs ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 font-mono resize-y"
        />
        <p className="text-[10px] text-muted-foreground">
          {param.description}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <Label htmlFor={id} className="text-xs">
        {param.name}
        {param.required && <span className="text-red-500 ml-0.5">*</span>}
      </Label>
      <Input
        id={id}
        type={param.type === "number" ? "number" : "text"}
        value={(value as string) ?? ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder={
          param.type === "datetime" ? "2024-01-01T00:00:00Z" : param.description
        }
        className="h-8 text-xs font-mono"
      />
      <p className="text-[10px] text-muted-foreground">{param.description}</p>
    </div>
  );
}
