from reflexio.models.api_schema.domain.entities import PlaybookRetrievalLog


class RetrievalLogMixin:
    """Mixin for playbook retrieval log storage methods.

    These methods are forward scaffolding for the offline playbook tuner and are
    not yet implemented by any storage backend. They are intentionally concrete
    (not ``@abstractmethod``) so concrete storage classes remain instantiable;
    each raises ``NotImplementedError`` until a backend provides a real
    implementation.
    """

    def save_playbook_retrieval_log(self, log: PlaybookRetrievalLog) -> int:
        """Persist a retrieval log entry and return its assigned id.

        Args:
            log (PlaybookRetrievalLog): The log entry to save. ``retrieval_log_id``
                may be 0; the storage layer assigns a real id on insert.

        Returns:
            int: The assigned ``retrieval_log_id``.
        """
        raise NotImplementedError

    def get_playbook_retrieval_logs(
        self,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> list[PlaybookRetrievalLog]:
        """Retrieve playbook retrieval log entries, optionally filtered.

        Args:
            session_id (str | None): Filter to logs for this session. If None,
                no session filter is applied.
            request_id (str | None): Filter to logs for this request. If None,
                no request filter is applied.

        Returns:
            list[PlaybookRetrievalLog]: Matching log entries, ordered by
                ``created_at`` ascending.
        """
        raise NotImplementedError
