"""Tests for DiskStorage file I/O (serialize/deserialize) and QMD client."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    Interaction,
    PlaybookStatus,
    Request,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import EMBEDDING_DIMENSIONS, SearchMode
from reflexio.server.services.storage.disk_storage._file_io import (
    deserialize_embedding,
    deserialize_entity,
    serialize_embedding,
    serialize_entity,
)
from reflexio.server.services.storage.disk_storage._qmd_client import QMDClient
from reflexio.server.services.storage.error import StorageError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = int(datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC).timestamp())


def _fake_embedding(seed: float = 0.1) -> list[float]:
    """Generate a valid 512-dimensional embedding vector for test fixtures."""
    return [seed + i * 0.001 for i in range(EMBEDDING_DIMENSIONS)]


@pytest.fixture()
def user_profile() -> UserProfile:
    return UserProfile(
        profile_id="prof_001",
        user_id="user_42",
        content="Prefers dark-mode UIs and concise answers.",
        last_modified_timestamp=_NOW,
        generated_from_request_id="req_abc",
        embedding=_fake_embedding(0.1),
    )


@pytest.fixture()
def interaction() -> Interaction:
    return Interaction(
        interaction_id=7,
        user_id="user_42",
        request_id="req_abc",
        created_at=_NOW,
        role="User",
        content="How do I enable dark mode?",
        embedding=_fake_embedding(0.4),
    )


@pytest.fixture()
def user_playbook() -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=3,
        user_id="user_42",
        agent_version="v2",
        request_id="req_xyz",
        playbook_name="tone-guide",
        created_at=_NOW,
        content="Always be concise and professional.",
        rationale="User explicitly asked for brevity.",
        trigger="User says 'be brief'",
        embedding=_fake_embedding(0.6),
    )


@pytest.fixture()
def agent_playbook() -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=10,
        playbook_name="escalation-protocol",
        agent_version="v3",
        created_at=_NOW,
        content="When the user is frustrated, escalate to a human.",
        rationale="Prevents customer churn.",
        trigger="Negative sentiment detected.",
        playbook_status=PlaybookStatus.APPROVED,
        playbook_metadata="meta-abc",
        embedding=_fake_embedding(0.9),
    )


@pytest.fixture()
def request_entity() -> Request:
    return Request(
        request_id="req_abc",
        user_id="user_42",
        created_at=_NOW,
        source="api",
        agent_version="v1",
        session_id="sess_01",
    )


@pytest.fixture()
def evaluation_result() -> AgentSuccessEvaluationResult:
    return AgentSuccessEvaluationResult(
        result_id=5,
        agent_version="v2",
        session_id="sess_01",
        is_success=True,
        failure_type=None,
        failure_reason=None,
        evaluation_name="accuracy",
        created_at=_NOW,
        embedding=_fake_embedding(1.1),
    )


# ===================================================================
# File I/O: serialize_entity / deserialize_entity
# ===================================================================


class TestUserProfileRoundTrip:
    """Round-trip test: serialize then deserialize a UserProfile."""

    def test_round_trip_preserves_fields(self, user_profile: UserProfile) -> None:
        md = serialize_entity(user_profile)
        restored = deserialize_entity(md, UserProfile)

        assert restored.profile_id == user_profile.profile_id
        assert restored.user_id == user_profile.user_id
        assert restored.content == user_profile.content
        assert restored.last_modified_timestamp == user_profile.last_modified_timestamp
        assert (
            restored.generated_from_request_id == user_profile.generated_from_request_id
        )

    def test_embedding_is_empty_after_round_trip(
        self, user_profile: UserProfile
    ) -> None:
        """Embedding is excluded during serialization and defaults to [] on deserialization."""
        md = serialize_entity(user_profile)
        restored = deserialize_entity(md, UserProfile)
        assert restored.embedding == []


class TestInteractionRoundTrip:
    """Round-trip test for Interaction entity."""

    def test_round_trip_preserves_fields(self, interaction: Interaction) -> None:
        md = serialize_entity(interaction)
        restored = deserialize_entity(md, Interaction)

        assert restored.interaction_id == interaction.interaction_id
        assert restored.user_id == interaction.user_id
        assert restored.request_id == interaction.request_id
        assert restored.created_at == interaction.created_at
        assert restored.role == interaction.role
        assert restored.content == interaction.content
        assert restored.embedding == []


class TestUserPlaybookRoundTrip:
    """Round-trip test for UserPlaybook, including flat structured fields."""

    def test_round_trip_preserves_fields(self, user_playbook: UserPlaybook) -> None:
        md = serialize_entity(user_playbook)
        restored = deserialize_entity(md, UserPlaybook)

        assert restored.user_playbook_id == user_playbook.user_playbook_id
        assert restored.user_id == user_playbook.user_id
        assert restored.agent_version == user_playbook.agent_version
        assert restored.request_id == user_playbook.request_id
        assert restored.playbook_name == user_playbook.playbook_name
        assert restored.content == user_playbook.content
        assert restored.embedding == []

    def test_flat_structured_fields(self, user_playbook: UserPlaybook) -> None:
        md = serialize_entity(user_playbook)
        restored = deserialize_entity(md, UserPlaybook)

        assert restored.rationale == user_playbook.rationale
        assert restored.trigger == user_playbook.trigger


class TestAgentPlaybookRoundTrip:
    """Round-trip test for AgentPlaybook, including PlaybookStatus enum."""

    def test_round_trip_preserves_fields(self, agent_playbook: AgentPlaybook) -> None:
        md = serialize_entity(agent_playbook)
        restored = deserialize_entity(md, AgentPlaybook)

        assert restored.agent_playbook_id == agent_playbook.agent_playbook_id
        assert restored.playbook_name == agent_playbook.playbook_name
        assert restored.agent_version == agent_playbook.agent_version
        assert restored.content == agent_playbook.content
        assert restored.playbook_metadata == agent_playbook.playbook_metadata
        assert restored.embedding == []

    def test_playbook_status_enum_preserved(
        self, agent_playbook: AgentPlaybook
    ) -> None:
        md = serialize_entity(agent_playbook)
        restored = deserialize_entity(md, AgentPlaybook)
        assert restored.playbook_status == PlaybookStatus.APPROVED


class TestRequestRoundTrip:
    """Round-trip test for Request -- a metadata-only entity (no content field)."""

    def test_round_trip_preserves_fields(self, request_entity: Request) -> None:
        md = serialize_entity(request_entity)
        restored = deserialize_entity(md, Request)

        assert restored.request_id == request_entity.request_id
        assert restored.user_id == request_entity.user_id
        assert restored.created_at == request_entity.created_at
        assert restored.source == request_entity.source
        assert restored.agent_version == request_entity.agent_version
        assert restored.session_id == request_entity.session_id

    def test_no_body_section_for_metadata_only(self, request_entity: Request) -> None:
        """Request has no content field, so the serialized output should not have a body."""
        md = serialize_entity(request_entity)
        # The md should be frontmatter-only: starts and ends with ---
        lines = md.strip().split("\n")
        assert lines[0] == "---"
        assert lines[-1] == "---"


class TestEvaluationResultRoundTrip:
    """Round-trip test for AgentSuccessEvaluationResult -- metadata-only."""

    def test_round_trip_preserves_fields(
        self, evaluation_result: AgentSuccessEvaluationResult
    ) -> None:
        md = serialize_entity(evaluation_result)
        restored = deserialize_entity(md, AgentSuccessEvaluationResult)

        assert restored.result_id == evaluation_result.result_id
        assert restored.agent_version == evaluation_result.agent_version
        assert restored.session_id == evaluation_result.session_id
        assert restored.is_success == evaluation_result.is_success
        assert restored.evaluation_name == evaluation_result.evaluation_name
        assert restored.embedding == []


class TestEmbeddingExclusion:
    """Verify that the top-level embedding field never appears in serialized output.

    We check for the YAML key ``embedding:`` at the start of a line (top-level
    frontmatter key) to avoid false positives.
    """

    @staticmethod
    def _has_toplevel_embedding_key(md: str) -> bool:
        """Return True if 'embedding:' appears as a top-level YAML key."""
        return any(line.strip().startswith("embedding:") for line in md.split("\n"))

    def test_embedding_not_in_serialized_output(
        self, user_profile: UserProfile
    ) -> None:
        md = serialize_entity(user_profile)
        assert not self._has_toplevel_embedding_key(md)

    def test_embedding_not_in_interaction_output(
        self, interaction: Interaction
    ) -> None:
        md = serialize_entity(interaction)
        assert not self._has_toplevel_embedding_key(md)

    def test_embedding_not_in_agent_playbook_output(
        self, agent_playbook: AgentPlaybook
    ) -> None:
        md = serialize_entity(agent_playbook)
        assert not self._has_toplevel_embedding_key(md)


class TestNoneStatusPreserved:
    """Verify that status=None is written as ``null`` in YAML and preserved on round-trip."""

    def test_none_status_written_as_null(self) -> None:
        profile = UserProfile(
            profile_id="p1",
            user_id="u1",
            content="test",
            last_modified_timestamp=_NOW,
            generated_from_request_id="r1",
            status=None,
        )
        md = serialize_entity(profile)
        restored = deserialize_entity(md, UserProfile)
        assert restored.status is None

    def test_none_status_in_user_playbook(self) -> None:
        playbook = UserPlaybook(
            user_playbook_id=1,
            user_id="u1",
            agent_version="v1",
            request_id="r1",
            content="test",
            status=None,
        )
        md = serialize_entity(playbook)
        restored = deserialize_entity(md, UserPlaybook)
        assert restored.status is None


class TestDeserializeErrors:
    """Verify that deserialize_entity raises ValueError for invalid input."""

    def test_no_frontmatter_delimiters(self) -> None:
        with pytest.raises(ValueError, match="does not start with YAML frontmatter"):
            deserialize_entity("no frontmatter here", UserProfile)

    def test_missing_closing_delimiter(self) -> None:
        with pytest.raises(ValueError, match="missing closing ---"):
            deserialize_entity("---\nfoo: bar\n", UserProfile)

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="does not start with YAML frontmatter"):
            deserialize_entity("", UserProfile)

    def test_only_opening_delimiter(self) -> None:
        with pytest.raises(ValueError, match="missing closing ---"):
            deserialize_entity("---\n", UserProfile)


class TestEmbeddingSidecar:
    """Tests for serialize_embedding / deserialize_embedding."""

    def test_round_trip(self) -> None:
        vec = [0.1, 0.2, 0.3, -0.5, 0.0]
        text = serialize_embedding(vec)
        restored = deserialize_embedding(text)
        assert restored == vec

    def test_empty_vector(self) -> None:
        text = serialize_embedding([])
        assert deserialize_embedding(text) == []

    def test_output_is_valid_json(self) -> None:
        vec = [1.0, 2.0]
        text = serialize_embedding(vec)
        parsed = json.loads(text)
        assert parsed == vec


# ===================================================================
# QMD Client
# ===================================================================


def _make_qmd_client(
    collection_path: Path, collection_name: str = "test_col"
) -> QMDClient:
    """Helper to construct a QMDClient with all subprocess calls mocked during init."""
    with patch("subprocess.run") as mock_run:
        # _check_installed: qmd --version succeeds
        version_result = MagicMock()
        version_result.returncode = 0

        # _ensure_collection: collection list returns empty
        list_result = MagicMock()
        list_result.returncode = 0
        list_result.stdout = "[]"
        list_result.stderr = ""

        # _ensure_collection: collection add succeeds
        add_result = MagicMock()
        add_result.returncode = 0
        add_result.stderr = ""

        # update_index: qmd update succeeds
        update_result = MagicMock()
        update_result.returncode = 0
        update_result.stderr = ""

        mock_run.side_effect = [version_result, list_result, add_result, update_result]

        return QMDClient(
            collection_path=collection_path, collection_name=collection_name
        )


class TestQMDCheckInstalled:
    """Tests for QMDClient._check_installed."""

    def test_returns_true_when_binary_exists(self, tmp_path: Path) -> None:
        client = _make_qmd_client(tmp_path)
        # If we got here without StorageError, _check_installed returned True
        assert client._available is True

    def test_auto_installs_when_not_found(self, tmp_path: Path) -> None:
        """When qmd is not found, auto-install is attempted; raises if all methods fail."""
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            pytest.raises(StorageError, match="automatic installation failed"),
        ):
            QMDClient(collection_path=tmp_path, collection_name="test_col")


class TestQMDSearch:
    """Tests for QMDClient.search."""

    def test_parses_json_output_correctly(self, tmp_path: Path) -> None:
        client = _make_qmd_client(tmp_path)
        search_output = json.dumps(
            {
                "results": [
                    {
                        "filepath": "/data/profiles/p1.md",
                        "score": 0.95,
                        "title": "Profile 1",
                        "snippet": "Likes sushi",
                        "source": "fts",
                    },
                    {
                        "filepath": "/data/profiles/p2.md",
                        "score": 0.80,
                        "title": "Profile 2",
                        "snippet": "Likes pizza",
                        "source": "fts",
                    },
                ]
            }
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = search_output

        with patch("subprocess.run", return_value=mock_result):
            results = client.search("sushi", mode=SearchMode.FTS)

        assert len(results) == 2
        # Paths are resolved relative to collection_path (tmp_path)
        assert results[0].filepath == str(tmp_path / "data" / "profiles" / "p1.md")
        assert results[0].score == 0.95
        assert results[0].title == "Profile 1"
        assert results[0].snippet == "Likes sushi"
        assert results[0].source == "fts"
        assert results[1].filepath == str(tmp_path / "data" / "profiles" / "p2.md")

    def test_returns_empty_on_subprocess_failure(self, tmp_path: Path) -> None:
        client = _make_qmd_client(tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "search index not found"

        with patch("subprocess.run", return_value=mock_result):
            results = client.search("query")

        assert results == []

    def test_returns_empty_on_file_not_found(self, tmp_path: Path) -> None:
        client = _make_qmd_client(tmp_path)

        with patch("subprocess.run", side_effect=FileNotFoundError):
            results = client.search("query")

        assert results == []

    def test_dispatches_fts_subcommand(self, tmp_path: Path) -> None:
        client = _make_qmd_client(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"results": []}'

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.search("test query", mode=SearchMode.FTS)
            args_passed = mock_run.call_args[0][0]
            assert args_passed[1] == "search"

    def test_dispatches_vector_subcommand(self, tmp_path: Path) -> None:
        client = _make_qmd_client(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"results": []}'

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.search("test query", mode=SearchMode.VECTOR)
            args_passed = mock_run.call_args[0][0]
            assert args_passed[1] == "vsearch"

    def test_dispatches_hybrid_subcommand(self, tmp_path: Path) -> None:
        client = _make_qmd_client(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"results": []}'

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.search("test query", mode=SearchMode.HYBRID)
            args_passed = mock_run.call_args[0][0]
            assert args_passed[1] == "query"


class TestQMDParseResults:
    """Tests for QMDClient._parse_results with edge cases."""

    def test_empty_string(self) -> None:
        assert QMDClient._parse_results("") == []

    def test_whitespace_only(self) -> None:
        assert QMDClient._parse_results("   \n  ") == []

    def test_malformed_json(self) -> None:
        assert QMDClient._parse_results("{not valid json") == []

    def test_missing_results_key(self) -> None:
        assert QMDClient._parse_results('{"data": []}') == []

    def test_skips_entries_without_filepath(self) -> None:
        output = json.dumps(
            {
                "results": [
                    {"filepath": "/valid.md", "score": 0.9},
                    {"score": 0.5, "title": "No path"},
                    {"filepath": "", "score": 0.3},
                ]
            }
        )
        results = QMDClient._parse_results(output)
        assert len(results) == 1
        assert results[0].filepath == "/valid.md"

    def test_defaults_for_missing_fields(self) -> None:
        output = json.dumps(
            {
                "results": [
                    {"filepath": "/minimal.md"},
                ]
            }
        )
        results = QMDClient._parse_results(output)
        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].title == ""
        assert results[0].snippet == ""
        assert results[0].source == ""


class TestQMDUpdateIndex:
    """Tests for QMDClient.update_index."""

    def test_calls_qmd_update(self, tmp_path: Path) -> None:
        client = _make_qmd_client(tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.update_index()
            args_passed = mock_run.call_args[0][0]
            assert "update" in args_passed


class TestQMDEnsureCollection:
    """Regression tests for stale-path detection in ``_ensure_collection``."""

    def test_register_uses_caller_provided_path_not_hardcoded(
        self, tmp_path: Path
    ) -> None:
        """When QMD has no existing collection, ``collection add`` is invoked
        with the caller's ``collection_path`` — never a hardcoded ``/tmp``
        path. Regression for the leak where production logs showed
        ``/tmp/e2e-disk/...`` instead of the configured directory."""
        client = _make_qmd_client(tmp_path, collection_name="test_col")
        # The constructor already invoked `collection add`; assert the path
        # baked into the client matches the directory we passed in.
        assert client.collection_path == tmp_path
        assert "/tmp/e2e-disk" not in str(client.collection_path)

    def test_re_registers_when_existing_path_is_stale(self, tmp_path: Path) -> None:
        """When QMD's registry holds the same name pointed at a stale path
        (e.g. a prior test run's ``/tmp/e2e-disk/...``), the client removes
        the stale entry and re-adds the collection at the real path."""
        from reflexio.server.services.storage.disk_storage._qmd_client import (
            QMDClient,
        )

        # Sequence of subprocess.run invocations during construction:
        #   1. _check_installed: qmd --version
        #   2. _ensure_collection: collection list --json (returns stale path)
        #   3. _ensure_collection: collection remove (idempotent)
        #   4. _ensure_collection: collection add with the new path
        #   5. update_index: qmd update
        version_result = MagicMock(returncode=0)
        list_result = MagicMock(
            returncode=0,
            stdout=json.dumps([{"name": "test_col", "path": "/tmp/e2e-disk/disk_x"}]),
            stderr="",
        )
        remove_result = MagicMock(returncode=0, stderr="")
        add_result = MagicMock(returncode=0, stderr="")
        update_result = MagicMock(returncode=0, stderr="")

        seq = [
            version_result,
            list_result,
            remove_result,
            add_result,
            update_result,
        ]
        with patch("subprocess.run", side_effect=seq) as mock_run:
            QMDClient(collection_path=tmp_path, collection_name="test_col")

        # Find the `collection add` invocation and assert it uses our
        # tmp_path, not the stale /tmp/e2e-disk path.
        add_calls = [
            call
            for call in mock_run.call_args_list
            if "collection" in call.args[0] and "add" in call.args[0]
        ]
        assert add_calls, "expected `collection add` invocation"
        argv = add_calls[0].args[0]
        assert str(tmp_path) in argv
        assert "/tmp/e2e-disk" not in " ".join(argv)
        # And the stale entry was removed first.
        remove_calls = [
            call
            for call in mock_run.call_args_list
            if "collection" in call.args[0] and "remove" in call.args[0]
        ]
        assert remove_calls, "expected `collection remove` invocation for stale path"

    def test_keeps_existing_collection_when_path_matches(self, tmp_path: Path) -> None:
        """When the registered path matches, no re-register and no
        ``update_index`` are issued — that path was the no-op we always
        had."""
        from reflexio.server.services.storage.disk_storage._qmd_client import (
            QMDClient,
        )

        version_result = MagicMock(returncode=0)
        list_result = MagicMock(
            returncode=0,
            stdout=json.dumps([{"name": "test_col", "path": str(tmp_path.resolve())}]),
            stderr="",
        )

        with patch(
            "subprocess.run", side_effect=[version_result, list_result]
        ) as mock_run:
            QMDClient(collection_path=tmp_path, collection_name="test_col")

        # No collection add / remove / update_index calls.
        invoked = [call.args[0] for call in mock_run.call_args_list]
        assert not any("add" in cmd and "collection" in cmd for cmd in invoked)
        assert not any("remove" in cmd and "collection" in cmd for cmd in invoked)


class TestQMDProbeTimeout:
    """Tests for the configurable probe timeout."""

    def test_status_uses_short_probe_timeout(self, tmp_path: Path) -> None:
        """``status`` is a fast probe and must use the short timeout, not 60s."""
        client = _make_qmd_client(tmp_path)
        client.probe_timeout_seconds = 3

        mock_result = MagicMock(returncode=0, stdout="{}", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.status()
        kwargs = mock_run.call_args.kwargs
        assert kwargs.get("timeout") == 3

    def test_status_returns_empty_when_probe_times_out(self, tmp_path: Path) -> None:
        """If the qmd subprocess wedges, status() bails with an empty dict
        rather than blocking the caller for 60s."""
        client = _make_qmd_client(tmp_path)

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="qmd", timeout=5),
        ):
            assert client.status() == {}

    def test_default_probe_timeout_is_short(self, tmp_path: Path) -> None:
        """Default probe timeout must be small enough that wedged probes
        don't block requests for tens of seconds."""
        from reflexio.server.services.storage.disk_storage._qmd_client import (
            DEFAULT_PROBE_TIMEOUT_SECONDS,
        )

        assert DEFAULT_PROBE_TIMEOUT_SECONDS <= 10
        client = _make_qmd_client(tmp_path)
        assert client.probe_timeout_seconds == DEFAULT_PROBE_TIMEOUT_SECONDS
