"""Isolated publish/read probe for #295; invoked by a bounded parent test.

Real ASGI handlers and temporary SQLite. Inference and background extraction
are excluded, so a pass cannot establish that the original Windows hang is fixed.
"""

import asyncio
import faulthandler
import json
import platform
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import httpx

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.api import create_app
from reflexio.server.services.configurator.configurator import DefaultConfigurator


async def probe(client, storage):
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=4))
    durations = []

    async def publish(index):
        start = time.monotonic()
        response = await client.post(
            "/api/publish_interaction",
            json={
                "user_id": f"probe-{index}",
                "request_id": f"probe-{index}",
                "session_id": f"probe-{index}",
                "agent_version": "isolated-probe",
                "interaction_data_list": [
                    {
                        "content": "Isolated concurrent publish probe.",
                        "created_at": int(time.time()),
                    }
                ],
            },
        )
        durations.append(time.monotonic() - start)
        assert response.status_code == 200, response.text
        assert response.json()["success"], response.text

    async def reads():
        for _ in range(20):
            response = await client.get("/api/get_all_profiles?limit=5")
            assert response.status_code == 200, response.text
            await asyncio.sleep(0.001)

    await asyncio.wait_for(
        asyncio.gather(*(publish(i) for i in range(20)), reads()), 20
    )
    persisted = sum(storage.get_request(f"probe-{i}") is not None for i in range(20))
    assert persisted == 20
    legacy = await client.post("/", json={})
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "reflexio_version": version("reflexio-ai"),
                "scope": "ASGI + SQLite; inference and extraction excluded",
                "concurrent_publishes": 20,
                "concurrent_read_requests": 20,
                "persisted_requests": persisted,
                "slowest_publish_seconds": round(max(durations), 3),
                "legacy_root_post_status": legacy.status_code,
            }
        )
    )


faulthandler.enable()
faulthandler.dump_traceback_later(20, repeat=True)

with tempfile.TemporaryDirectory(prefix="reflexio-295-") as tmp:
    org_id = "isolated-publish-probe"
    configurator = DefaultConfigurator(org_id=org_id, base_dir=tmp)
    configurator.set_config(
        Config(storage_config=StorageConfigSQLite(db_path=str(Path(tmp) / "probe.db")))
    )
    reflexio = Reflexio(org_id=org_id, storage_base_dir=tmp, configurator=configurator)
    storage = reflexio.get_storage()
    try:
        with (
            patch(
                "reflexio.server.api_endpoints.publisher_api.get_reflexio",
                return_value=reflexio,
            ),
            patch(
                "reflexio.server.cache.reflexio_cache.get_reflexio",
                return_value=reflexio,
            ),
            patch(
                "reflexio.server.services.durable_learning.local.ensure_local_extraction"
            ),
        ):

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(
                        app=create_app(get_org_id=lambda: org_id)
                    ),
                    base_url="http://localhost",
                ) as client:
                    await probe(client, storage)

            asyncio.run(run())
    finally:
        storage.conn.close()
        faulthandler.cancel_dump_traceback_later()
