"""GEPA-specific proof and search projection construction for publication."""

from __future__ import annotations

import json
from decimal import Decimal
from hashlib import sha256
from typing import Any

from reflexio.models.api_schema.domain import (
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationJob,
    UserPlaybook,
)
from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.services.playbook.publication import (
    PUBLICATION_PROJECTION_JSON_METADATA_KEY,
    PUBLICATION_PROOF_JSON_METADATA_KEY,
    PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY,
    DecisionProofEnvelope,
    PublicationRequest,
    PublicationSearchProjection,
    canonical_json_bytes,
)

from .models import ScenarioWindow

GEPA_PROJECTOR_ID = "reflexio-gepa-user-playbook-search-projector"
GEPA_PROJECTOR_VERSION = "1"
GEPA_PROJECTOR_CODE_DIGEST = (
    "aaa757a569cca1222097b3d6175e02d62486fc85dc884ceed29bd9c64819564b"
)
_PROOF_SCHEMA_VERSION = "gepa-user-playbook-decision-v1"
_PROJECTION_SCHEMA_VERSION = "offline-tuner-candidate-search-projection-v1"
GEPA_PUBLICATION_AUTHORITY_METADATA_KEY = "gepa_publication_authority"


def _decimal(value: float) -> str:
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError("GEPA publication decimals must be finite")
    return "0" if decimal == 0 else format(decimal.normalize(), "f")


def gepa_adoption_authority_from_config(
    *,
    config: PlaybookOptimizerConfig,
    validation_windows: list[ScenarioWindow],
    adoption_enabled: bool | None = None,
) -> dict[str, Any]:
    return {
        "adoption_policy": {
            "auto_update_user_playbooks": (
                config.auto_update_user_playbooks
                if adoption_enabled is None
                else adoption_enabled
            ),
            "min_commit_likert": config.min_commit_likert,
            "min_commit_score": repr(float(config.min_commit_score)),
            "min_commit_windows": config.min_commit_windows,
        },
        "validation_manifest": {
            "windows": [
                {
                    "scenario_user_playbook_id": window.user_playbook_id,
                    "source_interaction_ids": list(window.source_interaction_ids),
                    "min_commit_likert": config.min_commit_likert,
                    "min_commit_score": repr(float(config.min_commit_score)),
                }
                for window in validation_windows
            ],
        },
    }


def build_gepa_search_projection(
    storage: Any,
    incumbent: UserPlaybook,
    revised_content: str,
) -> PublicationSearchProjection:
    successor = incumbent.model_copy(
        update={
            "user_playbook_id": 0,
            "content": revised_content,
            "status": None,
            "embedding": [],
            "expanded_terms": None,
        }
    )
    storage.precompute_user_playbook_embeddings([successor])
    embedding = tuple(_decimal(value) for value in successor.embedding)
    expanded_terms = (
        (successor.expanded_terms.strip(),)
        if successor.expanded_terms and successor.expanded_terms.strip()
        else ()
    )
    lexical_document = " ".join(
        value
        for value in (incumbent.trigger, revised_content, *expanded_terms)
        if value
    )
    content_digest = sha256(revised_content.encode()).hexdigest()
    payload = {
        "candidate_content_digest": content_digest,
        "embedding": list(embedding),
        "embedding_model_id": storage.embedding_model_name,
        "expanded_terms": list(expanded_terms),
        "lexical_document": lexical_document,
        "preserved_trigger": incumbent.trigger,
        "projector_code_digest": GEPA_PROJECTOR_CODE_DIGEST,
        "projector_id": GEPA_PROJECTOR_ID,
        "projector_version": GEPA_PROJECTOR_VERSION,
        "schema_version": _PROJECTION_SCHEMA_VERSION,
    }
    canonical_json = canonical_json_bytes(payload).decode()
    return PublicationSearchProjection(
        schema_version=_PROJECTION_SCHEMA_VERSION,
        canonical_json=canonical_json,
        digest=sha256(canonical_json.encode()).hexdigest(),
        projector_id=GEPA_PROJECTOR_ID,
        projector_version=GEPA_PROJECTOR_VERSION,
        projector_code_digest=GEPA_PROJECTOR_CODE_DIGEST,
        candidate_content_digest=content_digest,
        preserved_trigger=incumbent.trigger,
        embedding_model_id=storage.embedding_model_name,
        embedding=embedding,
        expanded_terms=expanded_terms,
        lexical_document=lexical_document,
    )


def _proof_payload(
    *,
    job: PlaybookOptimizationJob,
    winner: PlaybookOptimizationCandidate,
    evaluations: list[PlaybookOptimizationEvaluation],
    metadata: dict[str, Any],
    subject_epochs_json: str,
    projection_digest: str,
) -> dict[str, Any]:
    authority = metadata.get(GEPA_PUBLICATION_AUTHORITY_METADATA_KEY)
    if not isinstance(authority, dict):
        raise ValueError("GEPA publication authority snapshot is missing")
    all_evaluations = sorted(evaluations, key=lambda item: item.evaluation_id)
    return {
        "adoption_authority": authority,
        "candidate": {
            "aggregate_score": _decimal(winner.aggregate_score or 0.0),
            "candidate_id": winner.candidate_id,
            "candidate_index": winner.candidate_index,
            "content_digest": sha256(winner.content.encode()).hexdigest(),
            "metadata_json": winner.metadata_json,
            "parent_candidate_ids": list(winner.parent_candidate_ids),
        },
        "decision": "apply",
        "evaluations": [
            {
                "candidate_id": item.candidate_id,
                "candidate_rollout_json": item.candidate_rollout_json,
                "created_at": item.created_at,
                "evaluation_id": item.evaluation_id,
                "incumbent_rollout_json": item.incumbent_rollout_json,
                "likert": item.likert,
                "rationale": item.rationale,
                "scenario_user_playbook_id": item.scenario_user_playbook_id,
                "score": _decimal(item.score),
                "source_interaction_ids": item.source_interaction_ids,
                "target_id": item.target_id,
                "target_kind": item.target_kind,
                "asi_json": item.asi_json,
                "verdict": item.verdict,
            }
            for item in all_evaluations
        ],
        "job": {
            "attempt_key": job.attempt_key,
            "job_id": job.job_id,
            "target_id": job.target_id,
            "target_kind": job.target_kind,
        },
        "optimizer_kind": "gepa",
        "projection_digest": projection_digest,
        "schema_version": _PROOF_SCHEMA_VERSION,
        "subject_epochs": json.loads(subject_epochs_json),
        "winning_adoption": _gepa_adoption_result_from_snapshot(
            winner=winner,
            evaluations=evaluations,
            authority=authority,
        ),
    }


def build_gepa_decision_proof(
    *,
    job: PlaybookOptimizationJob,
    winner: PlaybookOptimizationCandidate,
    evaluations: list[PlaybookOptimizationEvaluation],
    metadata: dict[str, Any],
    subject_epochs_json: str,
    projection_digest: str,
) -> DecisionProofEnvelope:
    canonical_json = canonical_json_bytes(
        _proof_payload(
            job=job,
            winner=winner,
            evaluations=evaluations,
            metadata=metadata,
            subject_epochs_json=subject_epochs_json,
            projection_digest=projection_digest,
        )
    ).decode()
    return DecisionProofEnvelope(
        optimizer_kind="gepa",
        schema_version=_PROOF_SCHEMA_VERSION,
        canonical_json=canonical_json,
        digest=sha256(canonical_json.encode()).hexdigest(),
        decision="apply",
    )


def _gepa_adoption_result_from_snapshot(
    *,
    winner: PlaybookOptimizationCandidate,
    evaluations: list[PlaybookOptimizationEvaluation],
    authority: dict[str, Any],
) -> dict[str, Any]:
    return gepa_winner_adoption_result(
        winner_candidate_id=winner.candidate_id,
        aggregate_score=winner.aggregate_score,
        evaluations=evaluations,
        authority=authority,
    )


def gepa_winner_adoption_result(
    *,
    winner_candidate_id: int,
    aggregate_score: float | None,
    evaluations: list[PlaybookOptimizationEvaluation],
    authority: dict[str, Any],
) -> dict[str, Any]:
    policy = authority.get("adoption_policy")
    validation_manifest = authority.get("validation_manifest")
    if not isinstance(policy, dict) or not isinstance(validation_manifest, dict):
        raise ValueError("GEPA adoption authority snapshot is invalid")
    windows = validation_manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("GEPA validation manifest is missing")
    min_windows = int(policy["min_commit_windows"])
    window_thresholds = {
        _window_key(
            item.get("scenario_user_playbook_id"),
            item.get("source_interaction_ids"),
        ): {
            "min_commit_likert": int(item["min_commit_likert"]),
            "min_commit_score": str(item["min_commit_score"]),
        }
        for item in windows
    }
    passed_window_keys: set[tuple[int | None, tuple[int, ...]]] = set()
    winning_windows = []
    for evaluation in evaluations:
        if evaluation.candidate_id != winner_candidate_id:
            continue
        key = _window_key(
            evaluation.scenario_user_playbook_id,
            evaluation.source_interaction_ids,
        )
        thresholds = window_thresholds.get(key)
        if thresholds is None:
            continue
        min_score = float(thresholds["min_commit_score"])
        min_likert = int(thresholds["min_commit_likert"])
        passed = (
            evaluation.verdict == "candidate"
            and evaluation.score >= min_score
            and evaluation.likert >= min_likert
        )
        if passed:
            passed_window_keys.add(key)
        winning_windows.append(
            {
                "evaluation_id": evaluation.evaluation_id,
                "likert": evaluation.likert,
                "passed": passed,
                "score": _decimal(evaluation.score),
                "thresholds": thresholds,
                "window": {
                    "scenario_user_playbook_id": evaluation.scenario_user_playbook_id,
                    "source_interaction_ids": list(evaluation.source_interaction_ids),
                },
            }
        )
    policy_min_score = float(policy["min_commit_score"])
    passes = (
        bool(policy.get("auto_update_user_playbooks"))
        and aggregate_score is not None
        and aggregate_score >= policy_min_score
        and len(passed_window_keys) >= min_windows
    )
    return {
        "aggregate_score": _decimal(aggregate_score or 0.0),
        "min_commit_windows": min_windows,
        "passes": passes,
        "passed_window_count": len(passed_window_keys),
        "winning_windows": winning_windows,
    }


def _window_key(
    scenario_user_playbook_id: object,
    source_interaction_ids: object,
) -> tuple[int | None, tuple[int, ...]]:
    if scenario_user_playbook_id is not None and not isinstance(
        scenario_user_playbook_id, int
    ):
        raise ValueError("GEPA validation window playbook id is invalid")
    if not isinstance(source_interaction_ids, list):
        raise ValueError("GEPA validation window source ids are invalid")
    return scenario_user_playbook_id, tuple(
        int(item) for item in source_interaction_ids
    )


def parse_gepa_search_projection(canonical_json: str) -> PublicationSearchProjection:
    payload = json.loads(canonical_json)
    if not isinstance(payload, dict):
        raise ValueError("GEPA search projection payload is invalid")
    return PublicationSearchProjection(
        schema_version=payload["schema_version"],
        canonical_json=canonical_json,
        digest=sha256(canonical_json.encode()).hexdigest(),
        projector_id=payload["projector_id"],
        projector_version=payload["projector_version"],
        projector_code_digest=payload["projector_code_digest"],
        candidate_content_digest=payload["candidate_content_digest"],
        preserved_trigger=payload["preserved_trigger"],
        embedding_model_id=payload["embedding_model_id"],
        embedding=tuple(payload["embedding"]),
        expanded_terms=tuple(payload["expanded_terms"]),
        lexical_document=payload["lexical_document"],
    )


def parse_gepa_decision_proof(canonical_json: str) -> DecisionProofEnvelope:
    payload = json.loads(canonical_json)
    if not isinstance(payload, dict):
        raise ValueError("GEPA decision proof payload is invalid")
    return DecisionProofEnvelope(
        optimizer_kind=payload["optimizer_kind"],
        schema_version=payload["schema_version"],
        canonical_json=canonical_json,
        digest=sha256(canonical_json.encode()).hexdigest(),
        decision=payload["decision"],
    )


class GEPAUserPlaybookDecisionVerifier:
    """Rebuild GEPA adoption authority exclusively from durable records."""

    def __init__(
        self,
        storage: Any,
    ) -> None:
        self._storage = storage

    def verify(self, request: PublicationRequest) -> None:
        job = self._storage.get_playbook_optimization_job(request.job_id)
        if (
            job is None
            or (job.optimizer_kind, job.target_kind, job.target_id)
            != ("gepa", "user_playbook", request.incumbent_user_playbook_id)
            or job.attempt_key != request.attempt_key
            or job.stage != "publishing"
            or job.best_candidate_id is None
        ):
            raise ValueError("GEPA durable job is not publishable")
        winners = [
            item
            for item in self._storage.list_playbook_optimization_candidates(job.job_id)
            if item.is_winner
        ]
        if len(winners) != 1 or winners[0].candidate_id != job.best_candidate_id:
            raise ValueError("GEPA durable winner changed")
        winner = winners[0]
        evaluations = self._storage.list_playbook_optimization_evaluations(job.job_id)
        if any(item.verdict == "aborted" for item in evaluations):
            raise ValueError("GEPA evaluation aborted")
        metadata = json.loads(job.metadata_json)
        authority = metadata.get(GEPA_PUBLICATION_AUTHORITY_METADATA_KEY)
        if not isinstance(authority, dict):
            raise ValueError("GEPA durable authority snapshot is missing")
        adoption = _gepa_adoption_result_from_snapshot(
            winner=winner,
            evaluations=evaluations,
            authority=authority,
        )
        if not adoption["passes"]:
            raise ValueError("GEPA durable winner fails adoption rules")
        if any(
            item.target_kind != "user_playbook" or item.target_id != job.target_id
            for item in evaluations
            if item.candidate_id == winner.candidate_id
        ):
            raise ValueError("GEPA evaluation target changed")
        subject_epochs = self._storage.get_user_playbook_publication_subject_epochs(
            job.target_id
        )
        incumbent = self._storage.get_user_playbook_by_id(job.target_id)
        if (
            subject_epochs != request.subject_epochs_json
            or metadata.get(PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY)
            != json.loads(subject_epochs)
            or incumbent is None
            or incumbent.trigger != request.projection.preserved_trigger
            or winner.content != request.revised_content
            or job.candidate_content_digest
            != request.projection.candidate_content_digest
            or job.search_projection_digest != request.projection.digest
            or metadata.get(PUBLICATION_PROJECTION_JSON_METADATA_KEY)
            != request.projection.canonical_json
        ):
            raise ValueError("GEPA publication binding changed")
        stored_proof_json = metadata.get(PUBLICATION_PROOF_JSON_METADATA_KEY)
        if not isinstance(stored_proof_json, str):
            raise ValueError("GEPA durable decision proof is missing")
        if parse_gepa_decision_proof(stored_proof_json) != request.decision_proof:
            raise ValueError("GEPA durable decision proof changed")
        proof_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "publication_proof_digest",
                PUBLICATION_PROOF_JSON_METADATA_KEY,
                PUBLICATION_PROJECTION_JSON_METADATA_KEY,
                PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY,
            }
        }
        try:
            expected = build_gepa_decision_proof(
                job=job,
                winner=winner,
                evaluations=evaluations,
                metadata=proof_metadata,
                subject_epochs_json=subject_epochs,
                projection_digest=request.projection.digest,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("GEPA durable decision proof changed") from exc
        if (
            expected != request.decision_proof
            or metadata.get("publication_proof_digest") != expected.digest
        ):
            raise ValueError("GEPA durable decision proof changed")
