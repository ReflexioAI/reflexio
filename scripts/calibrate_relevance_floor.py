#!/usr/bin/env python3
"""Score labeled (query, relevant, junk) pairs to calibrate the relevance floor.

Run (from the reflexio repo root): uv run python scripts/calibrate_relevance_floor.py

Replace SAMPLES with real (query, relevant-doc, junk-doc) triples from the target
corpus, then read off a floor below the relevant cluster and above the junk cluster.
This is a one-off analysis, not a test.
"""

from __future__ import annotations

from reflexio.server.llm.rerank import score_pairs

# (query, relevant_doc, clearly_irrelevant_doc) — replace with real corpus samples.
SAMPLES: list[tuple[str, str, str]] = [
    (
        "What are the user's dietary preferences?",
        "The user is vegetarian and avoids dairy.",
        "The user's preferred cloud deployment region is us-east-1.",
    ),
]


def main() -> None:
    rel_scores: list[float] = []
    junk_scores: list[float] = []
    for query, rel, junk in SAMPLES:
        r, j = score_pairs(query, [rel, junk])
        rel_scores.append(r)
        junk_scores.append(j)
        print(f"q={query!r}\n  relevant={r:+.2f}\n  junk={j:+.2f}")
    if rel_scores and junk_scores:
        print("\n--- summary ---")
        print(f"relevant: min={min(rel_scores):+.2f} max={max(rel_scores):+.2f}")
        print(f"junk    : min={min(junk_scores):+.2f} max={max(junk_scores):+.2f}")
        print(f"suggested floor: just below min(relevant) = {min(rel_scores):+.2f}")


if __name__ == "__main__":
    main()
