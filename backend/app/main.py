"""
EndeeSync - Local RAG-based Semantic Memory Engine.

FastAPI application entry point. Configures middleware, routers,
static files, templates, and application lifecycle events.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.deps import cleanup_services, get_embedder, get_vector_store
from app.config import get_settings
from app.routers import health, ingest, query, search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Initialize embedding model, connect to vector store
    - Shutdown: Clean up resources, close connections

    Args:
        app: FastAPI application instance.

    Yields:
        None
    """
    # Startup
    logger.info("Starting EndeeSync...")

    try:
        # Pre-initialize services for faster first request
        settings = get_settings()

        logger.info(f"Initializing embedder: {settings.embedding_model}")
        await get_embedder()

        logger.info(f"Connecting to vector store: {settings.endee_db_path}")
        await get_vector_store()

        logger.info("EndeeSync started successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Continue anyway - services will be initialized on first request

    yield

    # Shutdown
    logger.info("Shutting down EndeeSync...")
    await cleanup_services()
    logger.info("EndeeSync shutdown complete")


def create_app() -> FastAPI:
    """
    Application factory.

    Creates and configures the FastAPI application with:
    - CORS middleware
    - Static file serving
    - Jinja2 templates
    - Exception handlers
    - API routers
    - Lifespan events

    Returns:
        FastAPI: Configured application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Local RAG-based semantic memory engine with vector search and LLM-powered Q&A.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        logger.info(f"Static files mounted from: {STATIC_DIR}")

    # Initialize templates
    templates = None
    if TEMPLATES_DIR.exists():
        templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
        logger.info(f"Templates loaded from: {TEMPLATES_DIR}")

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred"},
        )

    # Register API routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
    app.include_router(search.router, prefix="/api/v1", tags=["Search"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])

    # Serve frontend
    @app.get("/", response_class=HTMLResponse, tags=["Frontend"])
    async def serve_frontend(request: Request):
        """Serve the main frontend application."""
        if templates:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "app_name": settings.app_name,
                    "app_version": settings.app_version,
                },
            )
        # Fallback if templates not configured
        return HTMLResponse(
            content="<h1>EndeeSync</h1><p>Templates not configured. Visit <a href='/docs'>/docs</a> for API.</p>",
            status_code=200,
        )

    return app


# Create application instance
app = create_app()
