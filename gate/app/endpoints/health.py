"""Health check and system endpoints."""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.state import state

router = APIRouter()


@router.get("/v1/healthz")
async def healthz():
    """Basic health check."""
    return {"ok": True}


@router.get("/v1/readyz")
async def readyz(response: Response):
    """Readiness check including retrieval service status."""
    if state.config_error:
        response.status_code = 503
        return {"ready": False, "config_error": state.config_error}
    assert state.retrieval is not None
    r = await state.retrieval.readyz()
    ready = bool(r.get("ready"))
    if not ready:
        response.status_code = 503
    return {"ready": ready, "retrieval": r}


@router.get("/v1/version")
async def version():
    """Service version and configuration summary."""
    if state.settings is None:
        return {"service": {"name": "rag-gate"}, "config_error": state.config_error}
    return {"service": {"name": state.settings.service_name}, "config": state.settings.safe_summary()}


@router.get("/v1/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

