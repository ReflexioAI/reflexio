"""Revisioned inline stages under the existing actual native agent-run owner."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from typing import Any, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from reflexio.server.llm._litellm_types import (
    ExtractionInputIntegrityError,
    ModelProvenance,
)
from reflexio.server.llm.tools import ToolLoopTrace
from reflexio.server.services.storage.storage_base import AgentRunRecord, BaseStorage
from reflexio.server.services.storage.storage_base._procedural_execution import KEY

from .models import ContractError, Maintenance, Observation, Review
from .transport import IncompleteBatchError, StageClient
from .workflow import digest


class ContinuingClient(StageClient):
    """Completed stages and dispatch reservations survive later provider outages."""

    def bind(
        self, storage: BaseStorage, run: AgentRunRecord, contract: dict[str, Any]
    ) -> None:
        self.storage, self.run = storage, run
        contract = json.loads(json.dumps(contract))
        saved = run.generation_request_snapshot.get(KEY)
        if saved is not None and saved.get("contract") != contract:
            raise ExtractionInputIntegrityError(
                "Procedural execution input/profile changed"
            )
        self.checkpoint = (
            deepcopy(saved)
            if saved is not None
            else {
                "version": 1,
                "revision": -1,
                "contract": contract,
                "physical_calls": 0,
                "completed": {},
                "failed": {},
                "trace": self.trace.model_dump(mode="json"),
                "provenance": None,
            }
        )
        self.physical_calls = self.checkpoint["physical_calls"]
        self.trace = ToolLoopTrace.model_validate(self.checkpoint["trace"])
        if self.checkpoint["provenance"] is not None:
            self.latest_provenance = ModelProvenance(**self.checkpoint["provenance"])
        self.persisted = deepcopy(saved)
        self._save()

    def _save(self) -> None:
        value = deepcopy(self.checkpoint)
        value["revision"] += 1
        value["physical_calls"] = self.physical_calls
        value["trace"] = self.trace.model_dump(mode="json")
        value["provenance"] = (
            asdict(self.latest_provenance) if self.latest_provenance else None
        )
        self.checkpoint = self.storage.checkpoint_procedural_execution(
            self.run, expected=self.persisted, save=value
        )
        self.persisted = deepcopy(self.checkpoint)

    def _dispatch_guard(self, *args: Any) -> None:
        super()._dispatch_guard(*args)
        # The reservation commits before the native transport may dispatch.
        self._save()

    def ask[T](
        self,
        stage: str,
        prompt: str,
        body: dict[str, Any],
        models: dict[str, type[BaseModel]],
        assemble: Callable[[list[tuple[str, Any]]], T],
        validate: Callable[[T], Any],
        *,
        evidence_choices: list[str] | None = None,
    ) -> T:
        adapter = TypeAdapter(
            {"source": list[Observation], "maintain": Maintenance, "review": Review}[
                stage
            ]
        )
        key = digest(
            {
                "stage": stage,
                "prompt": prompt,
                "body": body,
                "models": {
                    name: model.model_json_schema() for name, model in models.items()
                },
                "evidence_choices": evidence_choices,
            }
        )
        failure = self.checkpoint["failed"].get(key)
        if failure is not None:
            error = (
                IncompleteBatchError
                if failure["kind"] == "IncompleteBatchError"
                else ContractError
            )
            raise error("Retained procedural method failure: " + failure["reason"])
        saved = self.checkpoint["completed"].get(key)
        if saved is not None:
            value = cast(T, adapter.validate_python(saved["output"]))
            validate(value)
            self.latest_provenance = ModelProvenance(**saved["provenance"])
            return value
        try:
            value = super().ask(
                stage,
                prompt,
                body,
                models,
                assemble,
                validate,
                evidence_choices=evidence_choices,
            )
        except (ContractError, ValidationError) as error:
            self.checkpoint["failed"][key] = {
                "kind": type(error).__name__,
                "reason": str(error)[:4000],
            }
            self._save()
            raise
        except Exception:
            self._save()
            raise
        if self.latest_provenance is None:
            raise ExtractionInputIntegrityError("Procedural stage lacks provenance")
        self.checkpoint["completed"][key] = {
            "output": adapter.dump_python(value, mode="json"),
            "provenance": asdict(self.latest_provenance),
        }
        self._save()
        return value
