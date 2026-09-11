"""Durable, project-scoped explicit aggregation operations."""

from typing import Literal

from pydantic import BaseModel, Field


class SubmitPlaybookAggregationRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=200, pattern=r".*\S.*")
    agent_version: str = Field(min_length=1, max_length=200, pattern=r".*\S.*")


class PlaybookAggregationResult(BaseModel):
    clusters_found: int = 0
    user_playbooks_processed: int = 0
    playbooks_generated: int = 0
    skipped: str | None = None


class PlaybookAggregationOperation(BaseModel):
    operation_id: str
    request_id: str
    agent_version: str
    status: Literal["queued", "running", "retrying", "succeeded", "failed"]
    created_at: int
    updated_at: int
    started_at: int | None = None
    completed_at: int | None = None
    attempts: int = 0
    next_attempt_at: int | None = None
    error: str | None = None
    result: PlaybookAggregationResult | None = None


class AggregationOperationConflictError(ValueError):
    def __init__(self, operation_id: str, message: str):
        super().__init__(message)
        self.operation_id = operation_id
