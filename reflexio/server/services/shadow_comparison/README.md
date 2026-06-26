# shadow_comparison

Compact LLM-as-judge capability for Reflexio-vs-shadow response comparison.

- `judge.py` owns prompt rendering and the LLM call for one interaction.
- `outcome.py` owns pure position randomization and Reflexio-relative win/loss/tie derivation.

This package intentionally does not use `components/`: the current two-file split already separates LLM orchestration from pure outcome logic.
