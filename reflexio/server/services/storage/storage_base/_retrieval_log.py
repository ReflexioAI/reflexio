from reflexio.models.api_schema.domain.entities import PlaybookRetrievalLog


class RetrievalLogMixin:
    """Mixin for playbook retrieval log storage methods.

    These methods are optional retrieval-capture storage hooks. They are
    intentionally concrete (not ``@abstractmethod``) so concrete storage classes
    remain instantiable; each raises ``NotImplementedError`` until a backend
    provides a real implementation.
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
        interaction_id: int | None = None,
        user_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[PlaybookRetrievalLog]:
        """Retrieve playbook retrieval log entries, optionally filtered.

        Args:
            session_id (str | None): Filter to logs for this session. If None,
                no session filter is applied.
            request_id (str | None): Filter to logs for this request. If None,
                no request filter is applied.
            interaction_id (int | None): Filter to logs for this interaction.
            user_id (str | None): Filter to logs for this user.
            start_time (int | None): Inclusive lower bound on ``created_at``.
            end_time (int | None): Inclusive upper bound on ``created_at``.

        Returns:
            list[PlaybookRetrievalLog]: Matching log entries, ordered by
                ``created_at`` ascending.
        """
        raise NotImplementedError
