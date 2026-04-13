"""
Retrieval latency benchmark — main CLI entry point.

Measures end-to-end latency for profile search, user/agent playbook search,
and unified cross-entity search across storage backends and corpus sizes.
Two layers are supported:

- ``service`` — direct method call on :class:`Reflexio` (in-process, no
  HTTP). Measures reformulation + embedding lookup + storage retrieval.
- ``http`` — FastAPI ``TestClient`` POST to the corresponding ``/api/...``
  endpoint. Measures the same thing plus FastAPI routing, Pydantic
  serialization, and middleware overhead. No sockets — TCP/kernel cost is
  not included.

Query embeddings are pre-cached on disk (see :mod:`embed_cache`), so
embedder latency does not contaminate the measurement. Document embeddings
during seeding are deterministic fakes — document vector content doesn't
affect query-time latency, only row count does.

Usage::

    uv run python -m reflexio.benchmarks.retrieval_latency.bench \\
        --sizes 100,1000 --trials 30 --backend sqlite --layer service,http

See ``--help`` for the full flag list.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from reflexio.benchmarks.retrieval_latency.backends import (
    BACKENDS,
    BENCH_ORG_ID,
    BackendHandle,
)
from reflexio.benchmarks.retrieval_latency.embed_cache import (
    QueryEmbedCache,
    make_cached_query_embedder,
    patch_storage_get_embedding,
)
from reflexio.benchmarks.retrieval_latency.report import (
    RunResults,
    aggregate,
    load_baseline,
    render_markdown,
    write_results_json,
)
from reflexio.benchmarks.retrieval_latency.scenarios import QUERIES
from reflexio.benchmarks.retrieval_latency.seed import pick_bench_user_id, seed_corpus
from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
    SearchUserProfileRequest,
    UnifiedSearchRequest,
)

logger = logging.getLogger("reflexio.benchmarks.retrieval_latency")

Layer = Literal["service", "http"]
RetrievalType = Literal["profile", "user_playbook", "agent_playbook", "unified"]

DEFAULT_SIZES = (100, 1_000, 10_000)
DEFAULT_TRIALS = 50
DEFAULT_WARMUP = 5
DEFAULT_BACKENDS = ("sqlite", "supabase")
DEFAULT_LAYERS = ("service", "http")
DEFAULT_RETRIEVALS = ("profile", "user_playbook", "agent_playbook", "unified")

# Default output lives next to the bench module so reports are colocated with
# the script that produced them, not stranded at the repo root.
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_ROOT = _THIS_DIR / "results"


def _build_profile_request(query_idx: int) -> SearchUserProfileRequest:
    """
    Build a profile search request bound to the canned query set.

    Reformulation is disabled so the benchmark does not measure LLM cost.

    Args:
        query_idx (int): Index into the canned query set (wraps around).

    Returns:
        SearchUserProfileRequest: A ready-to-execute request.
    """
    return SearchUserProfileRequest(
        user_id=pick_bench_user_id(query_idx),
        query=QUERIES[query_idx % len(QUERIES)],
        top_k=10,
        enable_reformulation=False,
    )


def _build_user_playbook_request(query_idx: int) -> SearchUserPlaybookRequest:
    """
    Build a user playbook search request.

    Args:
        query_idx (int): Index into the canned query set (wraps around).

    Returns:
        SearchUserPlaybookRequest: A ready-to-execute request.
    """
    return SearchUserPlaybookRequest(
        query=QUERIES[query_idx % len(QUERIES)],
        user_id=pick_bench_user_id(query_idx),
        top_k=10,
        enable_reformulation=False,
    )


def _build_agent_playbook_request(query_idx: int) -> SearchAgentPlaybookRequest:
    """
    Build an agent playbook search request.

    Args:
        query_idx (int): Index into the canned query set (wraps around).

    Returns:
        SearchAgentPlaybookRequest: A ready-to-execute request.
    """
    return SearchAgentPlaybookRequest(
        query=QUERIES[query_idx % len(QUERIES)],
        top_k=10,
        enable_reformulation=False,
    )


def _build_unified_request(query_idx: int) -> UnifiedSearchRequest:
    """
    Build a unified search request.

    Args:
        query_idx (int): Index into the canned query set (wraps around).

    Returns:
        UnifiedSearchRequest: A ready-to-execute request.
    """
    return UnifiedSearchRequest(
        query=QUERIES[query_idx % len(QUERIES)],
        user_id=pick_bench_user_id(query_idx),
        top_k=10,
        enable_reformulation=False,
    )


def _service_call(
    reflexio: Reflexio,
    retrieval: RetrievalType,
    query_idx: int,
    org_id: str,
) -> None:
    """
    Execute one service-layer retrieval call in-process.

    Dispatches on ``retrieval`` to the matching ``Reflexio`` facade method.
    The response is discarded — the benchmark measures time, not accuracy.

    Args:
        reflexio (Reflexio): Live service-layer facade.
        retrieval (RetrievalType): Which retrieval path to exercise.
        query_idx (int): Query index to use for this call.
        org_id (str): Org ID for unified search's feature-flag checks.
    """
    match retrieval:
        case "profile":
            reflexio.search_profiles(_build_profile_request(query_idx))
        case "user_playbook":
            reflexio.search_user_playbooks(_build_user_playbook_request(query_idx))
        case "agent_playbook":
            reflexio.search_agent_playbooks(_build_agent_playbook_request(query_idx))
        case "unified":
            reflexio.unified_search(_build_unified_request(query_idx), org_id=org_id)


# Map retrieval type to (HTTP path, request builder) for the http layer.
_HTTP_ROUTES: dict[RetrievalType, tuple[str, Callable[[int], Any]]] = {
    "profile": ("/api/search_profiles", _build_profile_request),
    "user_playbook": ("/api/search_user_playbooks", _build_user_playbook_request),
    "agent_playbook": ("/api/search_agent_playbooks", _build_agent_playbook_request),
    "unified": ("/api/search", _build_unified_request),
}


def _http_call(client: Any, retrieval: RetrievalType, query_idx: int) -> None:
    """
    Execute one HTTP-layer retrieval call via FastAPI ``TestClient``.

    Args:
        client: A fastapi.testclient.TestClient bound to the benchmark app.
        retrieval (RetrievalType): Which retrieval path to exercise.
        query_idx (int): Query index to use for this call.

    Raises:
        RuntimeError: If the endpoint returns a non-200 response — that
            indicates a wiring problem, not a latency result, and should
            abort the run.
    """
    path, build = _HTTP_ROUTES[retrieval]
    payload = build(query_idx).model_dump(mode="json")
    resp = client.post(path, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP benchmark call failed: {retrieval} {path} -> "
            f"{resp.status_code} {resp.text[:200]}"
        )


def _make_test_client(handle: BackendHandle) -> Any:
    """
    Build a FastAPI ``TestClient`` wired to the benchmark Reflexio instance.

    The server-side ``get_reflexio`` cache is pre-populated so FastAPI
    handlers reuse our pre-seeded instance instead of creating their own
    (which would have no data and no custom storage config).

    Args:
        handle (BackendHandle): The live benchmark backend handle.

    Returns:
        TestClient: A fastapi.testclient.TestClient bound to a fresh app.
    """
    from fastapi.testclient import TestClient

    from reflexio.server.api import create_app
    from reflexio.server.cache import reflexio_cache

    with reflexio_cache._reflexio_cache_lock:
        reflexio_cache._reflexio_cache[(handle.org_id, None)] = handle.reflexio

    app = create_app(get_org_id=lambda: handle.org_id)
    return TestClient(app)


def _time_loop(
    call: Callable[[int], None],
    trials: int,
    warmup: int,
) -> list[float]:
    """
    Run ``warmup`` + ``trials`` timed invocations and return sample ms.

    Warmup invocations are discarded. Each sample uses
    :func:`time.perf_counter_ns` so the clock is monotonic and has
    sub-microsecond resolution on all supported platforms.

    Args:
        call (Callable[[int], None]): Fn that takes a query index and runs
            one benchmark invocation.
        trials (int): Number of timed samples to collect.
        warmup (int): Number of untimed warmup invocations.

    Returns:
        list[float]: ``trials`` latency samples in milliseconds.
    """
    for i in range(warmup):
        call(i)
    samples: list[float] = []
    for i in range(trials):
        t0 = time.perf_counter_ns()
        call(i)
        samples.append((time.perf_counter_ns() - t0) / 1_000_000.0)
    return samples


def _git_sha() -> str:
    """
    Return the current git HEAD short SHA, or ``"unknown"`` if unavailable.

    Used for run provenance in the results JSON. Never raises — the
    benchmark must complete even when the checkout isn't a git repo.

    Returns:
        str: Short SHA or ``"unknown"``.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except subprocess.SubprocessError, FileNotFoundError, OSError:
        return "unknown"


def _run_one_cell(
    handle: BackendHandle,
    client: Any | None,
    retrieval: RetrievalType,
    layer: Layer,
    n: int,
    trials: int,
    warmup: int,
) -> list[dict[str, Any]]:
    """
    Benchmark one ``(backend, retrieval, layer, n)`` cell and return rows.

    Args:
        handle (BackendHandle): Live backend handle.
        client: FastAPI TestClient (only required for ``layer='http'``).
        retrieval (RetrievalType): Which retrieval path to exercise.
        layer (Layer): ``service`` or ``http``.
        n (int): Corpus size already seeded into the backend.
        trials (int): Timed samples per cell.
        warmup (int): Untimed warmup invocations per cell.

    Returns:
        list[dict]: One row per trial, ready to append to raw results.
    """
    if layer == "service":

        def call(qi: int) -> None:
            _service_call(handle.reflexio, retrieval, qi, handle.org_id)
    else:
        if client is None:
            raise RuntimeError("http layer requires a TestClient")

        def call(qi: int) -> None:
            _http_call(client, retrieval, qi)

    samples = _time_loop(call, trials=trials, warmup=warmup)
    return [
        {
            "backend": handle.name,
            "retrieval_type": retrieval,
            "layer": layer,
            "n": n,
            "trial": i,
            "elapsed_ms": sample,
            "query_idx": i % len(QUERIES),
        }
        for i, sample in enumerate(samples)
    ]


def _run_for_backend(
    handle: BackendHandle,
    sizes: list[int],
    layers: list[Layer],
    retrievals: list[RetrievalType],
    trials: int,
    warmup: int,
    embed_cache: QueryEmbedCache,
) -> list[dict[str, Any]]:
    """
    Run the full (N × retrieval × layer) sweep on one backend.

    For each N: wipes the backend, re-seeds ``n`` rows of each entity type,
    then patches the storage with the cached query embedder and executes
    every (retrieval × layer) cell.

    A single wipe failure is tolerated (and logged) so a flaky delete doesn't
    abort a full run, but two consecutive failures raise — repeated failures
    silently contaminate larger-N cells with stale rows from smaller-N runs,
    which looks like a perf change in the report for no real reason.

    Args:
        handle (BackendHandle): Live backend handle.
        sizes (list[int]): Corpus sizes to sweep.
        layers (list[Layer]): Layers to benchmark.
        retrievals (list[RetrievalType]): Retrieval types to benchmark.
        trials (int): Timed samples per cell.
        warmup (int): Untimed warmup invocations per cell.
        embed_cache (QueryEmbedCache): Pre-populated query embedding cache.

    Returns:
        list[dict]: All raw rows for this backend.

    Raises:
        RuntimeError: If ``_wipe_backend`` fails twice in a row on the same
            backend, indicating delete is broken and every subsequent cell
            would be reading contaminated data.
    """
    rows: list[dict[str, Any]] = []
    client: Any | None = None
    if "http" in layers:
        client = _make_test_client(handle)
    consecutive_wipe_failures = 0
    for n in sizes:
        logger.info("Backend=%s, seeding N=%d", handle.name, n)
        if _wipe_backend(handle):
            consecutive_wipe_failures = 0
        else:
            consecutive_wipe_failures += 1
            if consecutive_wipe_failures >= 2:
                raise RuntimeError(
                    f"Backend {handle.name!r} wipe failed twice in a row; "
                    "aborting so larger-N cells don't get contaminated with "
                    "rows from previous runs."
                )
        seed_corpus(handle.reflexio, handle.storage, n)
        cached = make_cached_query_embedder(embed_cache)
        with patch_storage_get_embedding(handle.storage, cached):
            for retrieval in retrievals:
                for layer in layers:
                    logger.info(
                        "Measuring backend=%s retrieval=%s layer=%s N=%d",
                        handle.name,
                        retrieval,
                        layer,
                        n,
                    )
                    rows.extend(
                        _run_one_cell(
                            handle=handle,
                            client=client,
                            retrieval=retrieval,
                            layer=layer,
                            n=n,
                            trials=trials,
                            warmup=warmup,
                        )
                    )
    return rows


def _wipe_backend(handle: BackendHandle) -> bool:
    """
    Delete all seeded data from a backend so the next ``N`` starts fresh.

    Logs and swallows a single failure — the caller tracks consecutive
    failures and aborts on the second one, so one flaky delete doesn't kill
    a whole run but a broken delete path does.

    Args:
        handle (BackendHandle): Live backend handle.

    Returns:
        bool: ``True`` if every delete-all call succeeded, ``False`` if any
            raised. ``False`` signals the caller to bump its consecutive
            failure counter.
    """
    try:
        handle.reflexio.delete_all_profiles_bulk()
        handle.reflexio.delete_all_user_playbooks_bulk()
        handle.reflexio.delete_all_agent_playbooks_bulk()
    except Exception as err:  # noqa: BLE001
        logger.warning("Backend wipe failed on %s: %s", handle.name, err)
        return False
    return True


def _parse_csv_list[T](raw: str, valid: tuple[T, ...]) -> list[T]:
    """
    Parse a comma-separated CLI value against an allowlist.

    Args:
        raw (str): Raw CLI value, e.g. ``"sqlite,supabase"``.
        valid (tuple[T, ...]): Allowed values.

    Returns:
        list[T]: Parsed values in input order.

    Raises:
        ValueError: If any entry is not in ``valid``.
    """
    items = [x.strip() for x in raw.split(",") if x.strip()]
    for item in items:
        if item not in valid:
            raise ValueError(f"Invalid value {item!r}; choose from {valid}")
    return items  # type: ignore[return-value]


def _parse_sizes(raw: str) -> list[int]:
    """
    Parse a comma-separated list of corpus sizes from the CLI.

    Args:
        raw (str): E.g. ``"100,1000,10000"``.

    Returns:
        list[int]: Parsed integers, sorted ascending.

    Raises:
        ValueError: If any token is not a positive integer.
    """
    sizes = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if any(s <= 0 for s in sizes):
        raise ValueError("Sizes must be positive integers")
    return sorted(sizes)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for the benchmark.

    Args:
        argv (list[str] | None): Argument list, or ``None`` for sys.argv.

    Returns:
        argparse.Namespace: Parsed args.
    """
    parser = argparse.ArgumentParser(
        prog="reflexio-bench-retrieval",
        description="Retrieval latency benchmark (profile / playbook / unified).",
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=list(DEFAULT_SIZES),
        help="Comma-separated corpus sizes (default: 100,1000,10000).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Timed samples per cell (default: 50).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Untimed warmup invocations per cell (default: 5).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=",".join(DEFAULT_BACKENDS),
        help="Comma-separated backends (default: sqlite,supabase).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=",".join(DEFAULT_LAYERS),
        help="Comma-separated layers (default: service,http).",
    )
    parser.add_argument(
        "--retrieval",
        type=str,
        default=",".join(DEFAULT_RETRIEVALS),
        help="Comma-separated retrieval types (default: all four).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for results.json + report.md. "
        f"Default: {_DEFAULT_OUTPUT_ROOT}/<timestamp>/",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline results.json to diff the report against.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point: parse args, run the sweep, write outputs.

    Args:
        argv (list[str] | None): Optional override of sys.argv.

    Returns:
        int: Exit code — 0 on success, 1 on a skipped-every-backend run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    backends = _parse_csv_list(args.backend, DEFAULT_BACKENDS)
    layers = cast(list[Layer], _parse_csv_list(args.layer, DEFAULT_LAYERS))
    retrievals = cast(
        list[RetrievalType], _parse_csv_list(args.retrieval, DEFAULT_RETRIEVALS)
    )

    embed_cache = QueryEmbedCache()
    embed_cache.ensure(QUERIES)

    raw: list[dict[str, Any]] = []
    with ExitStack() as stack:
        for backend_name in backends:
            handle = stack.enter_context(BACKENDS[backend_name]())
            if handle is None:
                logger.warning("Skipping backend %s (unavailable)", backend_name)
                continue
            raw.extend(
                _run_for_backend(
                    handle=handle,
                    sizes=args.sizes,
                    layers=layers,
                    retrievals=retrievals,
                    trials=args.trials,
                    warmup=args.warmup,
                    embed_cache=embed_cache,
                )
            )

    if not raw:
        logger.error("No rows collected — every backend was skipped.")
        return 1

    stats = aggregate(raw)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    output_dir = args.output_dir or _DEFAULT_OUTPUT_ROOT / timestamp
    results = RunResults(
        config={
            "timestamp": timestamp,
            "git_sha": _git_sha(),
            "org_id": BENCH_ORG_ID,
            "sizes": args.sizes,
            "trials": args.trials,
            "warmup": args.warmup,
            "backends": backends,
            "layers": layers,
            "retrievals": retrievals,
        },
        raw=raw,
        stats=stats,
    )
    write_results_json(results, output_dir / "results.json")
    baseline = load_baseline(args.baseline) if args.baseline else {}
    (output_dir / "report.md").write_text(render_markdown(results, baseline))
    logger.info(
        "Wrote %s and %s", output_dir / "results.json", output_dir / "report.md"
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
