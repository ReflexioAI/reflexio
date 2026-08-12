# Minimal stub for the optional `mem0ai` dependency (not installed in dev/CI).
# Without it, pyright resolves the top-level name `mem0` from inside
# `reflexio/mem0/` to that package itself, making the wrapper appear to
# derive from itself. Only the surface `reflexio.mem0` touches is declared.
from typing import Any

class Memory: ...
class AsyncMemory: ...

class AsyncMemoryClient:
    api_key: str | None
    host: str
    client: Any
    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        client: Any = None,
    ) -> None: ...
    async def add(
        self, messages: Any, options: Any = None, **kwargs: Any
    ) -> dict[str, Any]: ...
    async def search(
        self, query: str, options: Any = None, **kwargs: Any
    ) -> dict[str, Any]: ...
    async def get(self, memory_id: str) -> dict[str, Any]: ...
    async def get_all(self, options: Any = None, **kwargs: Any) -> dict[str, Any]: ...
    async def delete_all(
        self, options: Any = None, **kwargs: Any
    ) -> dict[str, str]: ...
    async def delete(self, memory_id: str) -> dict[str, Any]: ...
    async def delete_users(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, str]: ...
    async def reset(self) -> dict[str, str]: ...

class MemoryClient:
    api_key: str | None
    host: str
    client: Any
    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        client: Any = None,
    ) -> None: ...
    def add(
        self, messages: Any, options: Any = None, **kwargs: Any
    ) -> dict[str, Any]: ...
    def search(
        self, query: str, options: Any = None, **kwargs: Any
    ) -> dict[str, Any]: ...
    def get(self, memory_id: str) -> dict[str, Any]: ...
    def get_all(self, options: Any = None, **kwargs: Any) -> dict[str, Any]: ...
    def delete_all(self, options: Any = None, **kwargs: Any) -> dict[str, str]: ...
    def delete(self, memory_id: str) -> dict[str, Any]: ...
    def delete_users(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, str]: ...
    def reset(self) -> dict[str, str]: ...
