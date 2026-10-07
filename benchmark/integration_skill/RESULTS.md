# Initial comparison: 2026-10-07

Both skill versions passed all six trials. This experiment found no correctness improvement on these two application-repair fixtures. It verifies that the updated instructions are usable; it does not establish that they improve integration success, cloud persistence, or tuning effectiveness.

| Fixture | Original passes | Updated passes | Original mean seconds | Updated mean seconds |
| --- | --- | --- | --- | --- |
| Python SDK | 3/3 | 3/3 | 130.93 | 145.97 |
| Direct HTTP | 3/3 | 3/3 | 144.67 | 157.93 |

| Metric across six trials | Original | Updated |
| --- | --- | --- |
| Mean agent seconds | 137.80 | 151.95 |
| Median agent seconds | 128.41 | 153.69 |
| Input tokens, including cached tokens | 1,991,682 | 2,120,494 |
| Cached input tokens | 1,806,080 | 1,867,648 |
| Uncached input tokens | 185,602 | 252,846 |
| Output tokens | 36,332 | 40,151 |
| Missing usage records | 0 | 0 |
| Timeouts / infrastructure errors | 0 / 0 | 0 / 0 |

The updated skill took longer and consumed more tokens in this sample. These are observed counts, not dollar costs or statistically established differences. The agents reported 7–16 passing local tests; manual review confirmed those counts against captured pytest output. Every report explicitly limited verification to offline work. No unsupported live persistence or completed tuning claim was found. Protected inputs were unchanged in every trial.

## Protocol and provenance

- Original skill: `7ce76a843eef201f53029a027cba21a914a3ed29`.
- Updated skill: `91956d6c59865746a1591062dfa0ff2202285b90`, [PR #594](https://github.com/ReflexioAI/reflexio/pull/594).
- Model: `gpt-6.1-sol`, medium reasoning; Codex CLI `0.160.1`.
- Two fixtures × two versions × three repetitions, sequential alternating order, fresh application/environment per trial, 900-second limit, no retries.
- Python `3.12.14`, lightweight `reflexio-client==0.2.16` wheel built from the original commit. Installed SDK source SHA-256: `421736dcd1385418772fa1f5e0f7288d6ef7dd0232a6b94c28d6deae5d11b79f`.
- Agent commands and independent grading used filesystem isolation and disabled networking. No cloud requests, production changes, or customer data were involved.

Trials began with the runner represented by the initial benchmark commit (`3238c3ed`, now `0e6d6db5` after rebasing). Review identified grader false-pass cases, process cleanup, infrastructure classification, and provenance gaps while trials were running. The prompt, fixtures, dependencies, and skill variants stayed unchanged. After all trials finished, **all 12 saved applications were regraded with the same final grader**, without further agent calls. Final grader SHA-256: `2c53f62f81069e666c79342f85572b7996f00c6c6ff29e8cd988870656d2e8f9`.

The execution manifest and initial runner snapshot were preserved. Environment provenance captured after execution began is labeled separately; final grading provenance and initial grades are also retained. These results do not claim that the trials used the subsequently hardened runner from the start.

Full local evidence is under `~/.reflexio/validation/integration-skill-benchmark-2026-10-07/`: `results.json`, `report.md`, manifests, per-trial transcripts, final reports, diffs, and initial/final grades. Large transcripts and copied dependency environments are not committed. See [README.md](README.md) for reproduction and interpretation limits. Both fixtures deliberately start with missing attribution and provide local API support; broader discovery tasks and connected tracing remain unmeasured.
