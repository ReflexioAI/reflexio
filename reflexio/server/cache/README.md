# Configuration cache ownership

`get_reflexio()` and `get_cached_request_context()` share construction,
invalidation fencing and organization/storage-directory isolation. Existing
borrowers retain their objects when an entry is evicted; eviction does not close
clients that another request may still use.

Cross-request reuse requires a revision captured before configuration loading
and a successful equal current revision. A failed hit probe triggers a fresh
load. If that load fails, its error propagates rather than returning old policy.
An adapter whose `get_version()` returns `None` still works, but each caller loads
current configuration without caching it. Implement revision probing to obtain
safe reuse. Failed construction-time probes likewise remain uncached.

Time is not proof that configuration remains current. The existing one-hour
instance turnover and `REFLEXIO_CACHE_MAX_SIZE` capacity are retained: instances
also capture model defaults and prompt assets outside the organization revision.
Removing turnover safely requires a version/reload owner for every captured
input, not merely changing TTLCache to LRUCache. This change adds no input and
claims no supported-knob reduction. `get_cache_stats()` keeps its size/capacity/
TTL diagnostic shape.

Regression coverage includes failed/unsupported probes, failed refresh and
recovery, real file-backed configuration through the HTTP route, concurrent
construction/invalidation, search deadlines and bounded capacity. No paid model
or remote database is needed for this cache contract.
