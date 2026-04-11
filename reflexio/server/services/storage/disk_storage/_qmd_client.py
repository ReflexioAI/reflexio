"""QMD CLI client — subprocess wrapper for search, collection, and index operations.

QMD (tobi/qmd) is a local markdown search engine supporting BM25, vector,
and hybrid search with LLM reranking.  This module wraps its CLI so that
DiskStorage can delegate ``search_*`` methods to it.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from reflexio.models.config_schema import SearchMode
from reflexio.server.services.storage.error import StorageError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QMDResult:
    """A single search result returned by QMD."""

    filepath: str
    score: float = 0.0
    title: str = ""
    snippet: str = ""
    source: str = ""  # "fts" | "vec" | "reranked"


@dataclass
class QMDClient:
    """Thin wrapper around the ``qmd`` CLI binary.

    Args:
        collection_path: Root directory of the QMD collection.
        collection_name: Name to register with QMD.
        qmd_binary: Path or command name for the qmd executable.
    """

    collection_path: Path
    collection_name: str
    qmd_binary: str = "qmd"
    _available: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._available = self._check_installed()
        if not self._available:
            logger.info("QMD CLI not found — attempting automatic installation...")
            self._auto_install()
            self._available = self._check_installed()
            if not self._available:
                raise StorageError(
                    "QMD CLI automatic installation failed. "
                    "Please install manually: npm install -g @tobilu/qmd"
                )
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.FTS,
        top_k: int = 10,
    ) -> list[QMDResult]:
        """Run a QMD search and return parsed results.

        Args:
            query: The search query string.
            mode: Search mode — FTS (BM25), VECTOR, or HYBRID.
            top_k: Maximum number of results to return.

        Returns:
            List of QMDResult ordered by relevance score.
        """
        cmd_map = {
            SearchMode.FTS: "search",
            SearchMode.VECTOR: "vsearch",
            SearchMode.HYBRID: "query",
        }
        subcommand = cmd_map.get(mode, "search")

        args = [
            self.qmd_binary,
            subcommand,
            "--json",
            "-n",
            str(top_k),
            "-c",
            self.collection_name,
            query,
        ]

        try:
            result = subprocess.run(  # noqa: S603
                args,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except FileNotFoundError:
            logger.error("QMD binary not found: %s", self.qmd_binary)
            return []
        except subprocess.TimeoutExpired:
            logger.error("QMD search timed out after 60s for query: %s", query)
            return []

        if result.returncode != 0:
            logger.warning(
                "QMD search failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return []

        return self._parse_results(result.stdout, self.collection_path)

    def update_index(self) -> None:
        """Re-index changed files via ``qmd update``."""
        self._run_qmd(["update"], timeout=120)

    def embed(self, force: bool = False) -> None:
        """Generate vector embeddings via ``qmd embed``.

        Args:
            force: If True, re-embed all documents (not just new ones).
        """
        args = ["embed"]
        if force:
            args.append("-f")
        self._run_qmd(args, timeout=300)

    def status(self) -> dict:
        """Return QMD index health information."""
        result = self._run_qmd(["status", "--json"], capture=True)
        if result and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {}
        return {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_installed(self) -> bool:
        """Check if the qmd binary is available."""
        try:
            subprocess.run(  # noqa: S603
                [self.qmd_binary, "--version"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _auto_install(self) -> None:
        """Attempt to install QMD automatically via npm or npx."""
        install_commands: list[list[str]] = [
            ["npm", "install", "-g", "@tobilu/qmd"],
            ["npx", "@tobilu/qmd", "--version"],  # npx auto-downloads on first run
        ]
        for cmd in install_commands:
            try:
                result = subprocess.run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode == 0:
                    logger.info("QMD installed successfully via %s", cmd[0])
                    return
                logger.warning(
                    "QMD install via %s failed (rc=%d): %s",
                    cmd[0],
                    result.returncode,
                    result.stderr.strip(),
                )
            except FileNotFoundError:
                logger.warning("%s not found — skipping this install method", cmd[0])
            except subprocess.TimeoutExpired:
                logger.warning("QMD install via %s timed out after 120s", cmd[0])
        logger.error("All QMD automatic installation methods failed")

    def _ensure_collection(self) -> None:
        """Register the storage directory as a QMD collection if not already registered."""
        # Check existing collections
        result = self._run_qmd(["collection", "list", "--json"], capture=True)
        if result and result.stdout.strip():
            try:
                collections = json.loads(result.stdout)
                # If collection already exists, skip
                if isinstance(collections, list):
                    for col in collections:
                        if (
                            isinstance(col, dict)
                            and col.get("name") == self.collection_name
                        ):
                            logger.info(
                                "QMD collection '%s' already registered",
                                self.collection_name,
                            )
                            return
                elif (
                    isinstance(collections, dict)
                    and self.collection_name in collections
                ):
                    logger.info(
                        "QMD collection '%s' already registered", self.collection_name
                    )
                    return
            except json.JSONDecodeError:
                pass

        # Register new collection
        logger.info(
            "Registering QMD collection '%s' at %s",
            self.collection_name,
            self.collection_path,
        )
        self._run_qmd(
            [
                "collection",
                "add",
                str(self.collection_path),
                "--name",
                self.collection_name,
            ]
        )

        # Initial index build
        self.update_index()

    def _run_qmd(
        self,
        args: list[str],
        timeout: int = 60,
        capture: bool = False,
    ) -> subprocess.CompletedProcess | None:
        """Run a qmd subcommand.

        Args:
            args: Arguments to pass after the qmd binary name.
            timeout: Subprocess timeout in seconds.
            capture: Whether to capture stdout/stderr.

        Returns:
            CompletedProcess if capture=True, else None.
        """
        cmd = [self.qmd_binary, *args]
        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=capture,
                text=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip() if capture and result.stderr else ""
                logger.warning(
                    "QMD command %s failed (rc=%d): %s",
                    args[0],
                    result.returncode,
                    stderr,
                )
            return result if capture else None
        except FileNotFoundError:
            logger.error("QMD binary not found: %s", self.qmd_binary)
            return None
        except subprocess.TimeoutExpired:
            logger.error("QMD command %s timed out after %ds", args[0], timeout)
            return None

    @staticmethod
    def _parse_results(
        stdout: str, collection_path: Path | None = None
    ) -> list[QMDResult]:
        """Parse QMD JSON output into QMDResult objects.

        Args:
            stdout: Raw JSON output from QMD CLI.
            collection_path: Base directory of the QMD collection, used to resolve
                relative paths from ``qmd://`` URIs to absolute filesystem paths.
        """
        if not stdout.strip():
            return []
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            logger.warning("Failed to parse QMD JSON output")
            return []

        # QMD may return a list directly or a dict with a "results" key
        results_raw = data if isinstance(data, list) else data.get("results", [])

        parsed = []
        for r in results_raw:
            # QMD uses "file" (qmd:// URI) or "filepath" depending on version
            raw_path = r.get("file", "") or r.get("filepath", "")
            if not raw_path:
                continue

            # Strip qmd:// URI scheme: "qmd://collection/path" → "path"
            if raw_path.startswith("qmd://"):
                parts = raw_path.split("/", 3)  # ["qmd:", "", "collection", "rest..."]
                raw_path = parts[3] if len(parts) > 3 else ""
            if not raw_path:
                continue

            # Resolve to absolute filesystem path.
            # QMD hyphenates directory names (user_playbooks → user-playbooks),
            # so try the path as-is first, then try with hyphens replaced by
            # underscores in directory components only (not the filename, which
            # may contain UUIDs with legitimate hyphens).
            if collection_path:
                abs_path = collection_path / raw_path
                if not abs_path.exists():
                    parts = raw_path.split("/")
                    # Replace hyphens with underscores in directory parts only
                    fixed_parts = [p.replace("-", "_") for p in parts[:-1]]
                    fixed_parts.append(parts[-1])  # Keep filename as-is
                    abs_path = collection_path / Path(*fixed_parts)
                raw_path = str(abs_path)

            parsed.append(
                QMDResult(
                    filepath=raw_path,
                    score=float(r.get("score", 0.0)),
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    source=r.get("source", ""),
                )
            )
        return parsed
