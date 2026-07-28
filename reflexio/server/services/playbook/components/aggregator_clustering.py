from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from reflexio.models.api_schema.service_schemas import UserPlaybook

logger = logging.getLogger(__name__)

# Threshold for switching between clustering algorithms
# Below this, use Agglomerative (works better with small datasets)
# Above this, use HDBSCAN (scales better, handles noise)
CLUSTERING_ALGORITHM_THRESHOLD = 50


def compute_cluster_fingerprint(cluster_playbooks: list[UserPlaybook]) -> str:
    """
    Compute a fingerprint for a cluster based on its user_playbook_ids.
    The fingerprint is deterministic and order-independent.

    Args:
        cluster_playbooks: List of raw playbooks in this cluster

    Returns:
        str: SHA-256 hash (truncated to 16 hex chars) of sorted user_playbook_ids
    """
    sorted_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
    id_str = ",".join(str(id) for id in sorted_ids)
    return hashlib.sha256(id_str.encode()).hexdigest()[:16]


def determine_cluster_changes(
    clusters: dict[int, list[UserPlaybook]],
    prev_fingerprints: dict,
) -> tuple[dict[int, list[UserPlaybook]], list[int]]:
    """
    Compare current cluster fingerprints against stored fingerprints to determine changes.

    Args:
        clusters: Current clusters (cluster_id -> list of UserPlaybook)
        prev_fingerprints: Previous fingerprint state
            (fingerprint_hash -> {"agent_playbook_id": int, "user_playbook_ids": list})

    Returns:
        tuple of:
            - changed_clusters: Only clusters needing new LLM calls
            - playbook_ids_to_archive: Old playbook_ids from changed/disappeared clusters
    """
    # Compute fingerprints for current clusters
    current_fingerprints = {}
    for cluster_id, cluster_playbooks in clusters.items():
        fp = compute_cluster_fingerprint(cluster_playbooks)
        current_fingerprints[cluster_id] = fp

    current_fp_set = set(current_fingerprints.values())
    prev_fp_set = set(prev_fingerprints.keys())

    # Changed clusters: fingerprints that are new (not in previous state)
    changed_clusters = {}
    for cluster_id, fp in current_fingerprints.items():
        if fp not in prev_fp_set:
            changed_clusters[cluster_id] = clusters[cluster_id]

    # Playbook IDs to archive: from fingerprints that disappeared or changed
    playbook_ids_to_archive = []
    for fp, fp_data in prev_fingerprints.items():
        if fp not in current_fp_set:
            playbook_id = fp_data.get("agent_playbook_id")
            if playbook_id is not None:
                playbook_ids_to_archive.append(playbook_id)

    return changed_clusters, playbook_ids_to_archive


def cluster_by_trigger_mock(
    user_playbooks: list[UserPlaybook], min_cluster_size: int
) -> dict[int, list[UserPlaybook]]:
    """
    Simple mock clustering by exact trigger match.

    Args:
        user_playbooks: List of user playbooks with trigger field
        min_cluster_size: Minimum number of playbooks per cluster

    Returns:
        dict[int, list[UserPlaybook]]: Clusters grouped by trigger
    """
    # Group by trigger. Playbooks without a trigger have nothing to group on, so
    # they are skipped rather than collapsed into a single empty-string bucket
    # that would surface as a spurious cluster.
    condition_groups: dict[str, list[UserPlaybook]] = {}
    for fb in user_playbooks:
        condition = fb.trigger or ""
        if not condition:
            continue
        if condition not in condition_groups:
            condition_groups[condition] = []
        condition_groups[condition].append(fb)

    # Convert to cluster format, filtering by min_cluster_size
    clusters: dict[int, list[UserPlaybook]] = {}
    cluster_id = 0
    for playbooks_group in condition_groups.values():
        if len(playbooks_group) >= min_cluster_size:
            clusters[cluster_id] = playbooks_group
            cluster_id += 1

    logger.info(
        "Mock mode: created %d trigger clusters from %d playbooks",
        len(clusters),
        len(user_playbooks),
    )
    return clusters


def cluster_with_agglomerative(
    distance_matrix: np.ndarray,
    min_cluster_size: int,  # noqa: ARG001
    distance_threshold: float,
) -> np.ndarray:
    """
    Cluster using Agglomerative Clustering - best for small datasets.

    Args:
        distance_matrix: Precomputed cosine distance matrix
        min_cluster_size: Minimum cluster size (used for logging only,
                          filtering happens in get_clusters)
        distance_threshold: Maximum cosine distance to merge clusters (1 - similarity_threshold)

    Returns:
        np.ndarray: Cluster labels for each point
    """
    from sklearn.cluster import AgglomerativeClustering

    logger.info(
        "Using Agglomerative Clustering for %d playbooks (< %d threshold), distance_threshold=%.2f",
        len(distance_matrix),
        CLUSTERING_ALGORITHM_THRESHOLD,
        distance_threshold,
    )

    clusterer = AgglomerativeClustering(
        n_clusters=None,  # type: ignore[reportArgumentType]
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )

    return clusterer.fit_predict(distance_matrix)


def cluster_with_hdbscan(
    distance_matrix: np.ndarray,
    min_cluster_size: int,
    distance_threshold: float,
) -> np.ndarray:
    """
    Cluster using HDBSCAN - best for large datasets with potential noise.

    Args:
        distance_matrix: Precomputed cosine distance matrix
        min_cluster_size: Minimum number of points to form a cluster
        distance_threshold: Maximum cosine distance for cluster merging (1 - similarity_threshold)

    Returns:
        np.ndarray: Cluster labels for each point (-1 indicates noise)
    """
    import hdbscan

    logger.info(
        "Using HDBSCAN for %d playbooks (>= %d threshold), distance_threshold=%.2f",
        len(distance_matrix),
        CLUSTERING_ALGORITHM_THRESHOLD,
        distance_threshold,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="precomputed",
        cluster_selection_epsilon=0.0,
        cluster_selection_epsilon_max=distance_threshold,
    )

    return clusterer.fit_predict(distance_matrix)
