"""
Health Check Router.

Provides endpoints for monitoring application health and readiness.
"""

from fastapi import APIRouter, Depends, status

from app.api.deps import get_embedder, get_llm_service, get_vector_store
from app.config import get_settings
from app.models.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check if the service is running and all components are healthy.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is degraded or unhealthy"},
    },
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Checks:
    - Application status
    - Vector store connectivity
    - Embedding model status

    Returns:
        HealthResponse: Service health status and version info.
    """
    settings = get_settings()

    # Check vector store
    try:
        vector_store = await get_vector_store()
        vector_store_status = "connected" if vector_store.is_connected() else "disconnected"
    except Exception:
        vector_store_status = "error"

    # Check embedding model
    try:
        embedder = await get_embedder()
        embedding_status = "loaded" if embedder.is_loaded() else "not_loaded"
    except Exception:
        embedding_status = "error"

    # Determine overall status
    if vector_store_status == "connected" and embedding_status == "loaded":
        overall_status = "healthy"
    elif vector_store_status == "error" or embedding_status == "error":
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        vector_store=vector_store_status,
        embedding_model=embedding_status,
    )


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Check if the service is ready to accept requests.",
)
async def readiness_check() -> dict:
    """
    Kubernetes-style readiness probe.

    Returns 200 if ready, 503 if not ready.
    """
    try:
        vector_store = await get_vector_store()
        embedder = await get_embedder()

        if vector_store.is_connected() and embedder.is_loaded():
            return {"ready": True}

    except Exception:
        pass

    return {"ready": False}


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Check",
    description="Check if the service is alive.",
)
async def liveness_check() -> dict:
    """
    Kubernetes-style liveness probe.

    Simple check that the server is responding.
    """
    return {"alive": True}
