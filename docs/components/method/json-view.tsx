"use client";

import { useMemo } from "react";

/**
 * Lightweight JSON syntax highlighter.
 * Tokenizes a pre-formatted JSON string and wraps each token in a colored span.
 */

type TokenType = "key" | "string" | "number" | "boolean" | "null" | "punctuation";

interface Token {
  type: TokenType;
  value: string;
}

function tokenize(json: string): (Token | string)[] {
  const tokens: (Token | string)[] = [];
  // Regex matches: strings (keys & values), numbers, booleans, null, and punctuation
  const re =
    /("(?:[^"\\]|\\.)*")\s*(:)|("(?:[^"\\]|\\.)*")|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b|(true|false)\b|(null)\b|([{}[\]:,])/g;

  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = re.exec(json)) !== null) {
    // Preserve whitespace between tokens
    if (match.index > lastIndex) {
      tokens.push(json.slice(lastIndex, match.index));
    }

    if (match[1] !== undefined) {
      // Key (string followed by colon)
      tokens.push({ type: "key", value: match[1] });
      tokens.push({ type: "punctuation", value: ":" });
    } else if (match[3] !== undefined) {
      tokens.push({ type: "string", value: match[3] });
    } else if (match[4] !== undefined) {
      tokens.push({ type: "number", value: match[4] });
    } else if (match[5] !== undefined) {
      tokens.push({ type: "boolean", value: match[5] });
    } else if (match[6] !== undefined) {
      tokens.push({ type: "null", value: match[6] });
    } else if (match[7] !== undefined) {
      tokens.push({ type: "punctuation", value: match[7] });
    }

    lastIndex = re.lastIndex;
  }

  if (lastIndex < json.length) {
    tokens.push(json.slice(lastIndex));
  }

  return tokens;
}

const colorMap: Record<TokenType, string> = {
  key: "text-blue-700 dark:text-blue-400",
  string: "text-emerald-700 dark:text-emerald-400",
  number: "text-amber-700 dark:text-amber-400",
  boolean: "text-purple-700 dark:text-purple-400",
  null: "text-red-600 dark:text-red-400",
  punctuation: "text-muted-foreground",
};

export function JsonView({ json }: { json: string }) {
  const highlighted = useMemo(() => {
    const tokens = tokenize(json);
    return tokens.map((token, i) => {
      if (typeof token === "string") return token;
      return (
        <span key={i} className={colorMap[token.type]}>
          {token.value}
        </span>
      );
    });
  }, [json]);

  return (
    <pre className="flex-1 overflow-auto p-4 text-xs font-mono leading-relaxed whitespace-pre-wrap break-all">
      {highlighted}
    </pre>
  );
}
