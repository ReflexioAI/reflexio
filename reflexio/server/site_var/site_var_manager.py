import json
import logging
from pathlib import Path

import redis

logger = logging.getLogger(__name__)


class SiteVarManager:
    """
    Manages site variables by loading JSON or text files from a source directory and caching them in Redis.
    Each file in the source directory represents a site variable with the filename as the key.
    Supports both .json and .txt files.
    """

    def __init__(self, source_dir: str | None = None, enable_redis: bool = False):
        """
        Initialize the SiteVarManager with a source directory for site variable files.

        Args:
            source_dir (Optional[str]): Directory containing JSON or text files for site variables
            enable_redis (bool): Whether to enable Redis caching (default: False)
        """
        self.source_dir = source_dir or str(Path(__file__).parent / "site_var_sources")
        self.enable_redis = enable_redis
        self.redis = (
            redis.Redis(host="localhost", port=6379, db=0) if enable_redis else None
        )

        # In-memory cache to avoid repeated file/Redis access
        self._cache: dict[str, str | dict] = {}

        # Ensure source directory exists
        if not Path(self.source_dir).exists():
            logger.warning(
                "Source directory %s does not exist, creating it", self.source_dir
            )
            Path(self.source_dir).mkdir(parents=True, exist_ok=True)

    def get_site_var(self, name: str) -> str | dict | None:
        """
        Get a site variable by name. Checks in-memory cache first, then Redis cache (if enabled), then loads from file.
        Supports both JSON and text files.

        Args:
            name (str): Name of the site variable to retrieve

        Returns:
            Optional[str | dict]: The site variable value as JSON dict (for JSON files) or plain text (for text files), or None if not found
        """
        try:
            # First check in-memory cache
            if name in self._cache:
                logger.debug("Retrieved site var %s from in-memory cache", name)
                return self._cache[name]

            # Then try to get from Redis cache if enabled
            if self.enable_redis and self.redis is not None:
                cached_value = self.redis.get(name)
                if cached_value is not None:
                    logger.debug("Retrieved site var %s from Redis cache", name)
                    decoded_value = cached_value.decode("utf-8")  # type: ignore[reportAttributeAccessIssue]
                    # Try to parse as JSON for dict values
                    try:
                        parsed_value = json.loads(decoded_value)
                        self._cache[name] = parsed_value
                        return parsed_value
                    except json.JSONDecodeError:
                        # It's a plain text value
                        self._cache[name] = decoded_value
                        return decoded_value

            # If not in cache or Redis disabled, try to load from file
            content = self._load_from_file(name)
            if content is not None:
                # Cache in memory
                self._cache[name] = content

                # Cache in Redis for future use if enabled
                if self.enable_redis and self.redis is not None:
                    # Convert dict to JSON string for Redis storage
                    redis_value = (
                        json.dumps(content) if isinstance(content, dict) else content
                    )
                    self.redis.set(name, redis_value)
                    logger.debug("Cached site var %s in Redis", name)
                return content
            logger.warning("Site var %s not found in Redis or files", name)
            return None

        except redis.RedisError as e:
            logger.error("Redis error while getting site var %s: %s", name, str(e))
            # Fallback to file-based loading
            content = self._load_from_file(name)
            if content is not None:
                self._cache[name] = content
            return content
        except Exception as e:
            logger.error("Error loading site var %s: %s", name, str(e))
            return None

    def evict_cache(self) -> None:
        """
        Evict (clear) all cached site variables from in-memory cache.
        This forces fresh loading from Redis or files on next access.
        """
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info("Evicted %d site variables from in-memory cache", cache_size)

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get statistics about the in-memory cache.

        Returns:
            dict[str, int]: Cache statistics including size and key count
        """
        return {
            "cache_size": len(self._cache),
            "cached_keys": len(list(self._cache.keys())),
        }

    def load_all_site_vars(self) -> dict[str, str | dict]:
        """
        Load all site variables from files in the source directory and cache them in memory and Redis (if enabled).
        Supports both JSON and text files.

        Returns:
            dict[str, str | dict]: Dictionary mapping site var names to their values
        """
        site_vars = {}

        try:
            if not Path(self.source_dir).exists():
                logger.warning("Source directory %s does not exist", self.source_dir)
                return site_vars

            for filename in [p.name for p in Path(self.source_dir).iterdir()]:
                if filename.endswith((".json", ".txt")):
                    name = self._get_name_from_filename(filename)
                    value = self.get_site_var(name)  # This will use cache automatically
                    if value is not None:
                        site_vars[name] = value

            logger.info(
                "Loaded %d site variables, %d now in cache",
                len(site_vars),
                len(self._cache),
            )
            return site_vars

        except Exception as e:
            logger.error("Error loading site variables from directory: %s", str(e))
            return site_vars

    def list_site_vars(self) -> list[str]:
        """
        List all available site variable names from the source directory.
        Supports both .json and .txt files.

        Returns:
            list[str]: List of site variable names
        """
        try:
            if not Path(self.source_dir).exists():
                return []

            return [
                entry.stem
                for entry in Path(self.source_dir).iterdir()
                if entry.name.endswith((".json", ".txt"))
            ]

        except Exception as e:
            logger.error("Error listing site variables: %s", str(e))
            return []

    def _find_file_path(self, name: str) -> str | None:
        """
        Find the file path for a site variable, checking both .json and .txt extensions.

        Args:
            name (str): Name of the site variable

        Returns:
            Optional[str]: Path to the file if found, None otherwise
        """
        json_path = Path(self.source_dir) / f"{name}.json"
        txt_path = Path(self.source_dir) / f"{name}.txt"

        if json_path.exists():
            return str(json_path)
        if txt_path.exists():
            return str(txt_path)
        return None

    def _load_file_content(self, file_path: str) -> str | dict | None:
        """
        Load content from a file, handling both JSON and text files.

        Args:
            file_path (str): Path to the file to load

        Returns:
            Optional[str]: File content as string, or None if error
        """
        try:
            if not Path(file_path).exists():
                logger.error("File %s does not exist", file_path)
                return None

            if file_path.endswith(".txt"):
                with Path(file_path).open(encoding="utf-8") as f:
                    return f.read().strip()
            if file_path.endswith(".json"):
                try:
                    # Parse to validate JSON, then return as string
                    with Path(file_path).open(encoding="utf-8") as f:
                        return json.load(f)
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON in file %s: %s", file_path, str(e))
                    return None
            else:
                logger.error(
                    "Unsupported file type: %s. Use 'json' or 'txt'", file_path
                )
                return None

        except OSError as e:
            logger.error("Error reading file %s: %s", file_path, str(e))
            return None

    def _load_from_file(self, name: str) -> str | dict | None:
        """
        Load a site variable directly from file (fallback method).

        Args:
            name (str): Name of the site variable

        Returns:
            Optional[str]: The site variable value, or None if not found
        """
        file_path = self._find_file_path(name)
        if file_path:
            return self._load_file_content(file_path)
        return None

    def _get_name_from_filename(self, filename: str) -> str:
        """
        Extract the site variable name from a filename by removing the extension.

        Args:
            filename (str): Filename with extension

        Returns:
            str: Site variable name without extension
        """
        if filename.endswith(".json"):
            return filename[:-5]
        if filename.endswith(".txt"):
            return filename[:-4]
        return filename
