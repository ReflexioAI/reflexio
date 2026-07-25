"""GEPA-specific proof and search projection construction for publication."""

from __future__ import annotations

import json
from collections.abc import Callable
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

AdoptionCheck = Callable[
    [int, int, list[ScenarioWindow], float, PlaybookOptimizerConfig], bool
]


def _decimal(value: float) -> str:
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError("GEPA publication decimals must be finite")
    return "0" if decimal == 0 else format(decimal.normalize(), "f")


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
    config: PlaybookOptimizerConfig,
    metadata: dict[str, Any],
    subject_epochs_json: str,
    projection_digest: str,
) -> dict[str, Any]:
    winner_evaluations = sorted(
        (item for item in evaluations if item.candidate_id == winner.candidate_id),
        key=lambda item: item.evaluation_id,
    )
    return {
        "adoption_rules": {
            "auto_update_user_playbooks": config.auto_update_user_playbooks,
            "min_commit_likert": config.min_commit_likert,
            "min_commit_score": _decimal(config.min_commit_score),
            "min_commit_windows": config.min_commit_windows,
        },
        "candidate": {
            "aggregate_score": _decimal(winner.aggregate_score or 0.0),
            "candidate_id": winner.candidate_id,
            "content_digest": sha256(winner.content.encode()).hexdigest(),
        },
        "decision": "apply",
        "evaluations": [
            {
                "candidate_id": item.candidate_id,
                "evaluation_id": item.evaluation_id,
                "likert": item.likert,
                "scenario_user_playbook_id": item.scenario_user_playbook_id,
                "score": _decimal(item.score),
                "source_interaction_ids": item.source_interaction_ids,
                "target_id": item.target_id,
                "target_kind": item.target_kind,
                "verdict": item.verdict,
            }
            for item in winner_evaluations
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
        "validation_windows": metadata["validation_windows"],
    }


def build_gepa_decision_proof(
    *,
    job: PlaybookOptimizationJob,
    winner: PlaybookOptimizationCandidate,
    evaluations: list[PlaybookOptimizationEvaluation],
    config: PlaybookOptimizerConfig,
    metadata: dict[str, Any],
    subject_epochs_json: str,
    projection_digest: str,
) -> DecisionProofEnvelope:
    canonical_json = canonical_json_bytes(
        _proof_payload(
            job=job,
            winner=winner,
            evaluations=evaluations,
            config=config,
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


def _validation_windows(metadata: dict[str, Any]) -> list[ScenarioWindow]:
    records = metadata.get("validation_windows")
    if not isinstance(records, list) or not records:
        raise ValueError("GEPA durable validation windows are missing")
    try:
        return [
            ScenarioWindow(
                user_playbook_id=record["scenario_user_playbook_id"],
                source_interaction_ids=record["source_interaction_ids"],
            )
            for record in records
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("GEPA durable validation windows are invalid") from exc


class GEPAUserPlaybookDecisionVerifier:
    """Rebuild GEPA adoption authority exclusively from durable records."""

    def __init__(
        self,
        storage: Any,
        *,
        config_provider: Callable[[], PlaybookOptimizerConfig],
        adoption_check: AdoptionCheck,
    ) -> None:
        self._storage = storage
        self._config_provider = config_provider
        self._adoption_check = adoption_check

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
        config = self._config_provider()
        if (
            not config.auto_update_user_playbooks
            or winner.aggregate_score is None
            or not self._adoption_check(
                job.job_id,
                winner.candidate_id,
                _validation_windows(metadata),
                winner.aggregate_score,
                config,
            )
        ):
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
        ):
            raise ValueError("GEPA publication binding changed")
        proof_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            not in {"publication_proof_digest", PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY}
        }
        expected = build_gepa_decision_proof(
            job=job,
            winner=winner,
            evaluations=evaluations,
            config=config,
            metadata=proof_metadata,
            subject_epochs_json=subject_epochs,
            projection_digest=request.projection.digest,
        )
        if (
            expected != request.decision_proof
            or metadata.get("publication_proof_digest") != expected.digest
        ):
            raise ValueError("GEPA durable decision proof changed")
