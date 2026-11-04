"""Shared compression under the existing fenced job and provider managers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import Any

from reflexio.server.llm import StructuredOutputRepairError
from reflexio.server.llm.input_budget import extraction_provider_guard
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    ProviderRequestGuardError,
)
from reflexio.server.services.input_compression import CompressionNodeInput
from reflexio.server.services.input_compression_llm import ClientCompressionPlan

from ..evidence.shared_compaction_checkpoint import (
    PROTOCOL,
    Attempt,
    Part,
    Response,
    Root,
    Selection,
    SharedReceipt,
    decode_selection,
    digest,
    encode_selection,
    receipt,
)

# Applies only to newly admitted shared-protocol jobs. Legacy native requests
# retain the exact qualified behavior frozen before this integration.
shared_consumer_admission: ContextVar[bool] = ContextVar(
    "offline_shared_consumer_admission", default=False
)


def _content_digest(content: Any) -> str:
    return sha256(
        json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


class SharedCompactionJournal:
    """One job-local reservation ledger; cached choices never authorize calls."""

    def __init__(
        self,
        storage: Any,
        *,
        job_id: int,
        fence: int,
        identity: str,
        before_call: Callable[[], None],
        now: Callable[[], int],
        limit: int,
    ):
        self.storage, self.job_id, self.fence = storage, job_id, fence
        self.identity, self.authorize, self.now = identity, before_call, now
        self.limit = limit
        self.plan: ClientCompressionPlan | None = None
        self.node: CompressionNodeInput | None = None
        self.active_attempt: str | None = None
        self.consumed_attempt: str | None = None
        self.responses: dict[str, str] = {}
        self.last_response: Response | None = None
        self.returned: list[tuple[str | None, Any]] = []

    def _read(self, key: str) -> dict[str, Any] | None:
        self.authorize()
        return self.storage.session_compaction_checkpoint(
            job_id=self.job_id,
            identity=key,
            fence=self.fence,
            now=self.now(),
            protocol=PROTOCOL,
        )

    def _write(
        self, value: dict[str, Any], previous: dict[str, Any] | None = None
    ) -> None:
        self.authorize()
        self.storage.session_compaction_checkpoint(
            job_id=self.job_id,
            identity=value["identity"],
            fence=self.fence,
            now=self.now(),
            protocol=PROTOCOL,
            receipt=value,
            expected_digest=digest(previous) if previous is not None else None,
        )

    def bind(self, plan: ClientCompressionPlan) -> None:
        self.plan = plan
        expected = Root(
            source_digest=plan.compression.source_digest,
            plan_witness=plan.witness,
            request_contract_digest=plan.request_contract_digest,
            limit=self.limit,
            calls=0,
        )
        with self.storage.commit_scope():
            saved = self._read(self.identity)
            if saved is None:
                self._write(receipt(self.identity, expected, complete=False))
            else:
                known = SharedReceipt.model_validate(saved).payload
                if (
                    not isinstance(known, Root)
                    or known.model_copy(update={"calls": 0}) != expected
                ):
                    raise ProviderRequestGuardError(
                        "Shared compaction frozen plan changed"
                    )

    def inspect(self, node: CompressionNodeInput) -> None:
        self.authorize()
        self.node = node
        self.active_attempt = self.consumed_attempt = None
        self.last_response = None
        self.responses = {}
        self.returned = []

    def guard(self, params: dict[str, Any], *_args: Any) -> None:
        self.authorize()
        if self.plan is None or self.node is None:
            raise ProviderRequestGuardError(
                "Shared compaction dispatch lacks its frozen node"
            )
        # Actual built params include each repair/fallback, not merely a logical
        # node call. Store only a digest, never credentials or source content.
        request_digest = _content_digest(
            {k: v for k, v in params.items() if k != "api_key"}
        )
        with self.storage.commit_scope():
            old = self._read(self.identity)
            if old is None:
                raise ProviderRequestGuardError(
                    "Shared compaction reservation root missing"
                )
            root = SharedReceipt.model_validate(old)
            if (
                not isinstance(root.payload, Root)
                or root.complete
                or root.payload.calls >= root.payload.limit
            ):
                raise ProviderRequestGuardError(
                    "Shared compaction physical allowance exhausted"
                )
            ordinal = root.payload.calls + 1
            key = digest({"root": self.identity, "attempt": ordinal})
            attempt = Attempt(
                root=self.identity,
                ordinal=ordinal,
                node=self.node.key,
                request_digest=request_digest,
            )
            self._write(
                receipt(
                    self.identity,
                    root.payload.model_copy(update={"calls": ordinal}),
                    complete=False,
                ),
                old,
            )
            self._write(receipt(key, attempt, complete=False))
        self.active_attempt, self.consumed_attempt = key, None

    @contextmanager
    def provider_scope(self) -> Iterator[None]:
        inherited = extraction_provider_guard.get()

        def guard(*args: Any) -> None:
            self.guard(*args)
            if inherited is not None:
                inherited(*args)

        token = extraction_provider_guard.set(guard)
        try:
            yield
        finally:
            extraction_provider_guard.reset(token)

    def observe(
        self, attempt_key: str | None, response: Any, client: LiteLLMClient
    ) -> None:
        if attempt_key is None or self.node is None:
            raise ProviderRequestGuardError("Shared response lacks a reserved attempt")
        provenance = client._build_model_provenance(response)
        if not provenance.model_name:
            raise ProviderRequestGuardError(
                "Shared response has no observed model identity"
            )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        observed = Response(
            model=provenance.model_name,
            provider=provenance.provider,
            content_digest=_content_digest(choice.message.content),
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
        with self.storage.commit_scope():
            old = self._read(attempt_key)
            if old is None:
                raise ProviderRequestGuardError("Shared response reservation missing")
            value = SharedReceipt.model_validate(old).payload
            if not isinstance(value, Attempt) or value.node != self.node.key:
                raise ProviderRequestGuardError(
                    "Shared response belongs to another node"
                )
            self._write(
                receipt(
                    attempt_key,
                    value.model_copy(update={"response": observed}),
                    complete=True,
                ),
                old,
            )
        self.responses[observed.content_digest] = attempt_key
        self.last_response = observed

    def flush_responses(self, client: LiteLLMClient) -> None:
        # Native metering runs before this finally-path. A journal/lease error
        # cannot suppress provider usage or cause it to be billed twice.
        pending, self.returned = self.returned, []
        for key, response in pending:
            self.observe(key, response, client)

    def consume(self, content_digest: str) -> None:
        self.consumed_attempt = self.responses.get(content_digest)

    def _selection_key(self, key: str) -> str:
        return digest({"root": self.identity, "selection": key})

    def load(self, key: str) -> dict[str, Any] | None:
        self.authorize()
        if self.node is None or self.plan is None or key != self.node.key:
            raise ProviderRequestGuardError("Shared selection node changed")
        saved = self._read(self._selection_key(key))
        if saved is None:
            return None
        manifest = SharedReceipt.model_validate(saved).payload
        if not isinstance(manifest, Selection) or (
            manifest.root,
            manifest.node,
            manifest.input_refs_digest,
            manifest.request_contract_digest,
        ) != (
            self.identity,
            key,
            self.node.input_refs_digest,
            self.plan.request_contract_digest,
        ):
            raise ProviderRequestGuardError("Shared selection source contract changed")
        observed = self._read(manifest.attempt)
        if observed is None:
            raise ProviderRequestGuardError(
                "Shared selection lacks its observed attempt"
            )
        attempt = SharedReceipt.model_validate(observed).payload
        if (
            not isinstance(attempt, Attempt)
            or (attempt.root, attempt.node) != (self.identity, key)
            or attempt.response is None
            or attempt.response.content_digest != manifest.response_digest
        ):
            raise ProviderRequestGuardError("Shared selection observation differs")
        parts = []
        for index in range(manifest.parts):
            part = self._read(
                digest(
                    {
                        "root": self.identity,
                        "node": key,
                        "body": manifest.body_digest,
                        "part": index,
                    }
                )
            )
            if part is None:
                raise ProviderRequestGuardError(
                    "Shared selection is missing a committed part"
                )
            value = SharedReceipt.model_validate(part).payload
            if not isinstance(value, Part):
                raise ProviderRequestGuardError("Shared selection part kind changed")
            parts.append(value)
        return decode_selection(manifest, parts)

    def save(self, key: str, output: dict[str, Any]) -> None:
        self.authorize()
        if (
            self.node is None
            or self.plan is None
            or key != self.node.key
            or self.consumed_attempt is None
        ):
            raise ProviderRequestGuardError(
                "Shared selection lacks consumed response provenance"
            )
        body, encoded = encode_selection(output)
        body_digest = sha256(body).hexdigest()
        with self.storage.commit_scope():
            observed = self._read(self.consumed_attempt)
            if observed is None:
                raise ProviderRequestGuardError("Shared selection response missing")
            attempt = SharedReceipt.model_validate(observed).payload
            if (
                not isinstance(attempt, Attempt)
                or attempt.root != self.identity
                or attempt.node != key
                or attempt.response is None
            ):
                raise ProviderRequestGuardError(
                    "Shared selection attempt belongs to another node"
                )
            manifest = Selection(
                root=self.identity,
                node=key,
                input_refs_digest=self.node.input_refs_digest,
                request_contract_digest=self.plan.request_contract_digest,
                body_digest=body_digest,
                body_bytes=len(body),
                parts=len(encoded),
                attempt=self.consumed_attempt,
                response_digest=attempt.response.content_digest,
            )
            for index, data in enumerate(encoded):
                part_key = digest(
                    {
                        "root": self.identity,
                        "node": key,
                        "body": body_digest,
                        "part": index,
                    }
                )
                self._write(
                    receipt(
                        part_key,
                        Part(
                            root=self.identity,
                            node=key,
                            body_digest=body_digest,
                            index=index,
                            data=data,
                        ),
                        complete=True,
                    )
                )
            # One commit publishes all parts and the manifest. A reader never
            # treats partially staged bytes as a completed selection.
            self._write(receipt(self._selection_key(key), manifest, complete=True))

    def finish(self) -> None:
        with self.storage.commit_scope():
            old = self._read(self.identity)
            if old is None:
                raise ProviderRequestGuardError("Shared compaction root missing")
            root = SharedReceipt.model_validate(old)
            if not root.complete:
                self._write({**old, "complete": True}, old)


class JournaledCompressionClient(LiteLLMClient):
    """Observe native dispatch; the superclass still owns capacity and deadlines."""

    def __init__(self, config: LiteLLMConfig, journal: SharedCompactionJournal):
        super().__init__(config)
        self.journal = journal

    def _completion_with_hard_timeout(
        self, params: dict[str, Any], hard_timeout: float
    ) -> Any:
        response = super()._completion_with_hard_timeout(params, hard_timeout)
        # Retain the exact returned object/ordinal before native parsing; do
        # not inspect it or raise ahead of native token-usage accounting.
        self.journal.returned.append((self.journal.active_attempt, response))
        return response

    def generate_chat_response_with_provenance(self, *args: Any, **kwargs: Any) -> Any:
        try:
            try:
                result = super().generate_chat_response_with_provenance(*args, **kwargs)
            finally:
                self.journal.flush_responses(self)
        except StructuredOutputRepairError as exc:
            if isinstance(exc.raw_content, str):
                self.journal.consume(_content_digest(exc.raw_content))
            raise
        else:
            if self.journal.last_response is None:
                raise ProviderRequestGuardError(
                    "Shared completion lacks an observed response"
                )
            observed = self.journal.last_response
            if (result.provenance.model_name, result.provenance.provider) != (
                observed.model,
                observed.provider,
            ):
                raise ProviderRequestGuardError(
                    "Shared completion response provenance changed"
                )
            self.journal.consume(observed.content_digest)
            return result
