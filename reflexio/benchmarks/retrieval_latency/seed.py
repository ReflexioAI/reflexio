"""
Deterministic synthetic corpus for the retrieval latency benchmark.

Builds ``UserProfile``, ``UserPlaybook``, and ``AgentPlaybook`` objects from
a fixed vocabulary and template set, then inserts them through the real
service-layer add methods (``Reflexio.add_user_profile`` et al.) so that the
storage index, FTS table, and vector index are exercised through the same
code paths that production uses.

Profiles and user playbooks are spread across a small fixed user pool so
that a per-user filter (``user_id``) still hits a non-trivial number of
rows — without that, profile search would always return at most one row
regardless of N.
"""

from __future__ import annotations

import logging
import random
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime

from reflexio.benchmarks.retrieval_latency.embed_cache import (
    make_fake_document_embedder,
    patch_storage_get_embedding,
)
from reflexio.benchmarks.retrieval_latency.scenarios import (
    CONTENT_TEMPLATES,
    VOCABULARY,
)
from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.domain.entities import (
    AddAgentPlaybookRequest,
    AddUserPlaybookRequest,
    AddUserProfileRequest,
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)

logger = logging.getLogger(__name__)

# Small pool of user_ids — profiles and user-playbooks distribute across
# these so that per-user filters still touch a meaningful number of rows.
USER_POOL_SIZE = 10
# Small pool of agent versions / playbook names — gives playbook-name filters
# enough selectivity to look like real usage without being single-row lookups.
AGENT_VERSION_POOL_SIZE = 3
PLAYBOOK_NAME_POOL_SIZE = 5

# Per-entity RNG seeds. All derived from a single base so the entire corpus
# is reproducible from one number, but each entity type gets its own stream so
# they don't collide content with each other.
_BENCH_SEED_BASE = 42
_PROFILE_SEED = _BENCH_SEED_BASE
_USER_PLAYBOOK_SEED = _BENCH_SEED_BASE + 1
_AGENT_PLAYBOOK_SEED = _BENCH_SEED_BASE + 2

# Batch size used when sending profiles/playbooks through add_* — keeps
# request objects under a sensible size and amortizes storage transactions.
_SEED_BATCH_SIZE = 100


def _user_id(idx: int) -> str:
    """
    Deterministically map an object index to a user_id in the fixed pool.

    Args:
        idx (int): The sequential index of the object being generated.

    Returns:
        str: A user_id string of the form ``bench_user_{i}``.
    """
    return f"bench_user_{idx % USER_POOL_SIZE}"


def _agent_version(idx: int) -> str:
    """
    Deterministically map an object index to an agent_version.

    Args:
        idx (int): The sequential index of the object being generated.

    Returns:
        str: An agent_version string of the form ``bench_agent_v{i}``.
    """
    return f"bench_agent_v{idx % AGENT_VERSION_POOL_SIZE}"


def _playbook_name(idx: int) -> str:
    """
    Deterministically map an object index to a playbook_name.

    Args:
        idx (int): The sequential index of the object being generated.

    Returns:
        str: A playbook_name string of the form ``bench_playbook_{i}``.
    """
    return f"bench_playbook_{idx % PLAYBOOK_NAME_POOL_SIZE}"


def _content(rng: random.Random) -> str:
    """
    Build one line of synthetic content from the fixed vocabulary.

    Picks a template and fills its single slot with a vocabulary word. Also
    appends 3–8 extra vocabulary tokens to give BM25 some additional surface
    area to match against.

    Args:
        rng (random.Random): Seeded RNG controlling all choices.

    Returns:
        str: A synthetic content string (typically 10–20 tokens).
    """
    template = rng.choice(CONTENT_TEMPLATES)
    word = rng.choice(VOCABULARY)
    extra_count = rng.randint(3, 8)
    extras = " ".join(rng.choice(VOCABULARY) for _ in range(extra_count))
    return f"{template.format(word=word)} {extras}"


def build_profiles(n: int) -> list[UserProfile]:
    """
    Build ``n`` deterministic synthetic ``UserProfile`` objects.

    Args:
        n (int): Number of profiles to generate.

    Returns:
        list[UserProfile]: Profiles spread across the fixed user pool. Each
        has a fresh UUID profile_id, a synthetic content string, and a
        current timestamp (fixed across a single run via ``rng`` for
        determinism at the seeding step).
    """
    rng = random.Random(_PROFILE_SEED)
    now = int(datetime.now(UTC).timestamp())
    return [
        UserProfile(
            profile_id=f"bench_profile_{i}_{uuid.UUID(int=rng.getrandbits(128))}",
            user_id=_user_id(i),
            content=_content(rng),
            last_modified_timestamp=now,
            generated_from_request_id=f"bench_req_{i}",
        )
        for i in range(n)
    ]


def build_user_playbooks(n: int) -> list[UserPlaybook]:
    """
    Build ``n`` deterministic synthetic ``UserPlaybook`` objects.

    Args:
        n (int): Number of user playbooks to generate.

    Returns:
        list[UserPlaybook]: Playbooks spread across the fixed user and
        agent-version pools. ``user_playbook_id`` is left at the default
        ``0`` so storage assigns fresh IDs during insertion.
    """
    rng = random.Random(_USER_PLAYBOOK_SEED)
    return [
        UserPlaybook(
            user_id=_user_id(i),
            agent_version=_agent_version(i),
            playbook_name=_playbook_name(i),
            request_id=f"bench_req_{i}",
            content=_content(rng),
        )
        for i in range(n)
    ]


def build_agent_playbooks(n: int) -> list[AgentPlaybook]:
    """
    Build ``n`` deterministic synthetic ``AgentPlaybook`` objects.

    Args:
        n (int): Number of agent playbooks to generate.

    Returns:
        list[AgentPlaybook]: Playbooks spread across the fixed agent-version
        and playbook-name pools.
    """
    rng = random.Random(_AGENT_PLAYBOOK_SEED)
    return [
        AgentPlaybook(
            agent_version=_agent_version(i),
            playbook_name=_playbook_name(i),
            content=_content(rng),
        )
        for i in range(n)
    ]


def _batched[T](items: list[T], size: int) -> Iterable[list[T]]:
    """
    Yield contiguous batches of ``items`` with length up to ``size``.

    Args:
        items (list[T]): Input list to batch.
        size (int): Maximum batch length.

    Yields:
        list[T]: Each batch in order.
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def seed_corpus(reflexio: Reflexio, storage: object, n: int) -> None:
    """
    Insert a full synthetic corpus into the given Reflexio instance.

    Document embeddings are generated from a deterministic fake embedder
    for the duration of this call — document vector *content* doesn't affect
    query-time latency (only row count does), so the optimization trades
    vector meaningfulness for seeding speed. Query-time retrieval is
    expected to patch the storage with a real query embedding cache instead.

    Args:
        reflexio (Reflexio): Service-layer facade with storage wired up.
        storage: The underlying ``BaseStorage`` instance to patch.
        n (int): Number of each entity type to seed (profiles, user
            playbooks, agent playbooks — ``3 * n`` total rows).
    """
    fake = make_fake_document_embedder()
    with patch_storage_get_embedding(storage, fake):
        logger.info("Seeding %d profiles / user_playbooks / agent_playbooks", n)
        for batch in _batched(build_profiles(n), _SEED_BATCH_SIZE):
            reflexio.add_user_profile(AddUserProfileRequest(user_profiles=batch))
        for batch in _batched(build_user_playbooks(n), _SEED_BATCH_SIZE):
            reflexio.add_user_playbook(AddUserPlaybookRequest(user_playbooks=batch))
        for batch in _batched(build_agent_playbooks(n), _SEED_BATCH_SIZE):
            reflexio.add_agent_playbook(AddAgentPlaybookRequest(agent_playbooks=batch))
    logger.info("Seeding complete (n=%d)", n)


def pick_bench_user_id(query_idx: int) -> str:
    """
    Choose a user_id for profile/user-playbook search from the seeded pool.

    Rotates through the pool by query index so successive queries aren't
    always hitting the same user's rows.

    Args:
        query_idx (int): Index of the current query (0-based).

    Returns:
        str: A user_id that exists in the seeded corpus.
    """
    return f"bench_user_{query_idx % USER_POOL_SIZE}"
