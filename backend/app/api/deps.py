"""
API Dependencies.

Dependency injection for FastAPI routes.
Provides configured service instances.
"""

import logging
from functools import lru_cache
from typing import AsyncGenerator

from app.config import get_settings
from app.core.embedder import Embedder
from app.db.vector_store import VectorStore
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

# Global service instances (initialized on first use)
_embedder: Embedder | None = None
_vector_store: VectorStore | None = None
_llm_service: LLMService | None = None


async def get_embedder() -> Embedder:
    """Get or create the embedder instance."""
    global _embedder

    if _embedder is None:
        settings = get_settings()
        _embedder = Embedder(settings.embedding_model)
        _embedder.load_model()
        logger.info(f"Embedder initialized: {settings.embedding_model}")

    return _embedder


async def get_vector_store() -> VectorStore:
    """Get or create the vector store instance."""
    global _vector_store

    if _vector_store is None:
        settings = get_settings()
        _vector_store = VectorStore(
            db_path=settings.endee_db_path,
            collection_name=settings.endee_collection_name,
            dimension=settings.embedding_dimension,
        )
        await _vector_store.connect()
        logger.info(f"Vector store connected: {settings.endee_db_path}")

    return _vector_store


async def get_llm_service() -> LLMService:
    """Get or create the LLM service instance."""
    global _llm_service

    if _llm_service is None:
        try:
            _llm_service = LLMService()
            await _llm_service.initialize()
            logger.info("LLM service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise

    return _llm_service


async def get_ingestion_service() -> IngestionService:
    """Get a configured ingestion service."""
    embedder = await get_embedder()
    vector_store = await get_vector_store()
    return IngestionService(embedder, vector_store)


async def get_retrieval_service() -> RetrievalService:
    """Get a configured retrieval service."""
    embedder = await get_embedder()
    vector_store = await get_vector_store()
    return RetrievalService(embedder, vector_store)


async def get_rag_service() -> RAGService:
    """Get a configured RAG service."""
    retrieval_service = await get_retrieval_service()
    llm_service = await get_llm_service()
    return RAGService(retrieval_service, llm_service)


async def cleanup_services() -> None:
    """Cleanup all services on shutdown."""
    global _vector_store, _embedder, _llm_service

    if _vector_store is not None:
        await _vector_store.disconnect()
        _vector_store = None

    if _embedder is not None:
        _embedder.unload_model()
        _embedder = None

    _llm_service = None

    logger.info("Services cleaned up")
