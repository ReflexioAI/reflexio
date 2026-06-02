"""Reflection decision-eval harness (AI-judged).

Scaffolding to measure whether the reflection step makes the right
decision for a cited item, and to catch regressions when the
``memory_reflection`` prompt changes.

This package contains *bounded scaffolding plus a tiny illustrative
fixture* — it is NOT a curated eval dataset. Real cases are curated
later; the fixture (``fixtures/illustrative_cases.json``) exists only to
exercise the harness end-to-end.
"""
