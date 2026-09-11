# /reflexio/server/services/shadow_comparison
Description: LLM judging of regular and shadow responses with Reflexio-relative outcomes.

## Main Entry Points


- `dispatcher.py` schedules assistant turns carrying shadow content from publish.
- `worker.py` owns bounded background execution and verdict persistence.
- `judge.py` owns prompt rendering and the LLM call for one interaction.
- `outcome.py` owns pure position randomization and Reflexio-relative win/loss/tie derivation.

## Purpose


Produce independent comparison verdicts for evaluation reporting.

## Architecture Pattern


`judge.py` renders the prompt and invokes the model; `outcome.py` randomizes answer positions and converts the result back to a Reflexio-relative win/loss/tie.

## Requirements / Problems to Avoid


This package intentionally does not use `components/`: dispatch, worker execution, judging, and pure outcome logic already have focused files.
