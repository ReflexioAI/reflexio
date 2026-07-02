"""Domain route modules for the OSS FastAPI app (Tier3 A2 split of api.py).

Each module owns its own ``APIRouter`` and is aggregated into
``reflexio.server.api.core_router`` so the single ``core_router`` remains the
data-plane router surface the enterprise composition mounts and iterates.
"""
