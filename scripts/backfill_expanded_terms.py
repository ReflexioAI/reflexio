#!/usr/bin/env python3
"""Backfill expanded_terms for existing documents.

Iterates over rows in raw_feedbacks, feedbacks, and/or profiles where
expanded_terms IS NULL, calls DocumentExpander.expand() on the content,
and writes the result back into the expanded_terms column and FTS index.

Usage:
    uv run python scripts/backfill_expanded_terms.py --db-path ./data/reflexio.db
    uv run python scripts/backfill_expanded_terms.py --db-path ./data/reflexio.db --table raw_feedbacks
    uv run python scripts/backfill_expanded_terms.py --db-path ./data/reflexio.db --dry-run
    uv run python scripts/backfill_expanded_terms.py --db-path ./data/reflexio.db --limit 50
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # scripts/
_PROJECT_ROOT = _THIS_DIR.parent  # repo root

sys.path.insert(0, str(_PROJECT_ROOT))

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.pre_retrieval import DocumentExpander

# Tables that have an expanded_terms column and the column used as content source.
_TABLE_CONFIG: dict[str, dict[str, str]] = {
    "raw_feedbacks": {
        "id_col": "raw_feedback_id",
        "content_col": "feedback_content",
        "fts_table": "raw_feedbacks_fts",
    },
    "feedbacks": {
        "id_col": "feedback_id",
        "content_col": "feedback_content",
        "fts_table": "feedbacks_fts",
    },
    "profiles": {
        "id_col": "profile_id",
        "content_col": "content",
        "fts_table": "profiles_fts",
    },
}


def _build_fts_search_text(
    row: sqlite3.Row, content_col: str, expanded_terms: str
) -> str:
    """Build the FTS search_text string matching the storage layer convention.

    Args:
        row: The database row (dict-like).
        content_col: Name of the content column.
        expanded_terms: The newly computed expanded terms string.

    Returns:
        Combined search text for FTS indexing.
    """
    # For raw_feedbacks and feedbacks, FTS search_text = trigger || content || expanded_terms
    structured_data_raw = (
        row["structured_data"] if "structured_data" in row.keys() else None  # noqa: SIM118
    )
    trigger = ""
    if structured_data_raw:
        try:
            sd = json.loads(structured_data_raw)
            trigger = sd.get("trigger", "")
        except (json.JSONDecodeError, TypeError):  # fmt: skip
            pass

    content = row[content_col] or ""
    parts = [p for p in [trigger, content, expanded_terms] if p]
    return " ".join(parts)


def _backfill_table(
    conn: sqlite3.Connection,
    expander: DocumentExpander,
    table: str,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> int:
    """Backfill expanded_terms for a single table.

    Args:
        conn: SQLite connection.
        expander: Configured DocumentExpander instance.
        table: Table name (must be a key in _TABLE_CONFIG).
        dry_run: If True, compute expansions but do not write.
        limit: Maximum number of rows to process (None = all).

    Returns:
        Number of rows processed.
    """
    cfg = _TABLE_CONFIG[table]
    id_col = cfg["id_col"]
    content_col = cfg["content_col"]
    fts_table = cfg["fts_table"]

    cols = f"{id_col}, {content_col}, structured_data"
    sql = f"SELECT {cols} FROM {table} WHERE expanded_terms IS NULL"  # noqa: S608  table from _TABLE_CONFIG
    if limit:
        sql += " LIMIT ?"

    rows = conn.execute(sql, (limit,) if limit else ()).fetchall()
    total = len(rows)
    if not total:
        print(f"  {table}: no rows need backfilling")
        return 0

    processed = 0
    for row in rows:
        row_id = row[id_col]
        content = row[content_col] or ""
        result = expander.expand(content)

        if dry_run:
            print(f"  [DRY-RUN] {table} {id_col}={row_id}: {result.expanded_text!r}")
        else:
            conn.execute(
                f"UPDATE {table} SET expanded_terms = ? WHERE {id_col} = ?",  # noqa: S608
                (result.expanded_text or None, row_id),
            )

            # Update FTS index
            if table == "profiles":
                # Profiles FTS uses profile_id TEXT key, not rowid
                conn.execute("DELETE FROM profiles_fts WHERE profile_id = ?", (row_id,))
                fts_content = content
                if result.expanded_text:
                    fts_content = f"{content} {result.expanded_text}"
                conn.execute(
                    "INSERT INTO profiles_fts(profile_id, content) VALUES (?, ?)",
                    (row_id, fts_content),
                )
            else:
                # raw_feedbacks and feedbacks use integer rowid
                search_text = _build_fts_search_text(
                    row, content_col, result.expanded_text
                )
                conn.execute(f"DELETE FROM {fts_table} WHERE rowid = ?", (row_id,))  # noqa: S608
                conn.execute(
                    f"INSERT INTO {fts_table}(rowid, search_text) VALUES (?, ?)",  # noqa: S608
                    (row_id, search_text),
                )

            # Batch commits every 50 rows for performance (avoids per-row fsync)
            if processed % 50 == 0:
                conn.commit()

        processed += 1
        if processed % 10 == 0 or processed == total:
            prefix = "[DRY-RUN] " if dry_run else ""
            print(f"  {prefix}Expanded {processed}/{total} {table}")

    if not dry_run:
        conn.commit()  # Final commit for remaining rows
    return processed


def main() -> None:
    """Entry point for the backfill CLI."""
    parser = argparse.ArgumentParser(
        description="Backfill expanded_terms for existing documents in SQLite."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--table",
        choices=list(_TABLE_CONFIG.keys()),
        default=None,
        help="Process only this table (default: all tables).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview expansions without writing to the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process per table.",
    )
    args = parser.parse_args()

    db_path: Path = args.db_path
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    llm_client = LiteLLMClient()
    prompt_manager = PromptManager()
    expander = DocumentExpander(llm_client=llm_client, prompt_manager=prompt_manager)

    tables = [args.table] if args.table else list(_TABLE_CONFIG.keys())
    total_processed = 0

    print(f"Backfilling expanded_terms in: {db_path}")
    if args.dry_run:
        print("  Mode: DRY-RUN (no writes)")
    if args.limit:
        print(f"  Limit: {args.limit} rows per table")

    for table in tables:
        total_processed += _backfill_table(
            conn, expander, table, dry_run=args.dry_run, limit=args.limit
        )

    conn.close()
    print(f"\nDone. Processed {total_processed} rows total.")


if __name__ == "__main__":
    main()
