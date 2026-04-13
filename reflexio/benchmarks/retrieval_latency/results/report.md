# Retrieval latency benchmark

- Trials per cell: 30
- Warmup per cell: 3
- Query set: 20 fixed queries (see scenarios.py)
- Cell format: `p50 / p95 (mean)` in ms

## agent_playbook


| backend | layer   | N=100              | N=1000             |
| ------- | ------- | ------------------ | ------------------ |
| sqlite  | http    | 18.8 / 39.1 (21.4) | 19.2 / 21.3 (19.7) |
| sqlite  | service | 15.2 / 16.9 (15.5) | 16.9 / 21.6 (17.6) |


## profile


| backend | layer   | N=100           | N=1000             |
| ------- | ------- | --------------- | ------------------ |
| sqlite  | http    | 4.1 / 5.6 (4.4) | 20.7 / 25.6 (21.2) |
| sqlite  | service | 1.8 / 2.2 (2.0) | 17.4 / 21.2 (17.9) |


## unified


| backend | layer   | N=100              | N=1000             |
| ------- | ------- | ------------------ | ------------------ |
| sqlite  | http    | 24.7 / 30.3 (25.4) | 58.8 / 71.8 (59.9) |
| sqlite  | service | 20.1 / 21.5 (20.1) | 57.5 / 72.7 (58.3) |


## user_playbook


| backend | layer   | N=100           | N=1000             |
| ------- | ------- | --------------- | ------------------ |
| sqlite  | http    | 3.9 / 4.8 (4.2) | 19.5 / 22.0 (19.8) |
| sqlite  | service | 1.6 / 2.0 (1.7) | 17.1 / 17.8 (17.1) |


