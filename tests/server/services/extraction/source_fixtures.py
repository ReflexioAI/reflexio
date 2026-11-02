"""Persist exact synthetic prompt inputs before exercising retained extraction."""

from collections.abc import Sequence

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.services.storage.storage_base import BaseStorage


def persist_source_models(
    storage: BaseStorage, models: Sequence[RequestInteractionDataModel]
) -> None:
    with storage.commit_scope():
        for group in models:
            storage.add_request(group.request)
            storage.add_user_interactions_bulk(
                group.request.user_id, group.interactions, embeddings_prepared=True
            )
