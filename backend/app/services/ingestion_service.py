"""
Ingestion Service.

Orchestrates the full ingestion pipeline:
text → chunking → embedding → storage
"""

import logging
from datetime import datetime, timezone

from app.config import get_settings
from app.core.chunker import ChunkConfig, Chunker
from app.core.embedder import Embedder
from app.core.exceptions import IngestionError
from app.core.indexer import IndexConfig, Indexer, IndexResult
from app.db.vector_store import VectorStore
from app.models.domain import Document
from app.models.schemas.ingest import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Service for ingesting text documents into the vector store.

    Handles the complete pipeline from raw text to stored vectors.
    This is the main entry point for the ingestion API endpoints.

    Example:
        service = IngestionService(embedder, vector_store)
        response = await service.ingest(request)
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Initialize the ingestion service.

        Args:
            embedder: Embedding generation component.
            vector_store: Vector storage backend.
        """
        self._embedder = embedder
        self._vector_store = vector_store

        # Create indexer with settings
        settings = get_settings()
        config = IndexConfig(
            chunk_size=settings.chunk_size * 4,  # Convert tokens to chars
            chunk_overlap=settings.chunk_overlap * 4,
            embedding_model=settings.embedding_model,
        )
        self._indexer = Indexer(embedder, vector_store, config)

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """
        Ingest a text document.

        Pipeline:
        1. Validate input
        2. Chunk text with recursive splitting
        3. Generate embeddings for all chunks
        4. Store chunks with metadata in vector store

        Args:
            request: Ingestion request with text and metadata.

        Returns:
            IngestResponse: Result with document ID and chunk count.

        Raises:
            IngestionError: If any pipeline step fails.
        """
        logger.info(f"Ingesting document: source={request.source}, tags={request.tags}")

        try:
            # Use indexer for the heavy lifting
            result: IndexResult = await self._indexer.index_text(
                text=request.text,
                source=request.source,
                tags=request.tags,
                metadata=request.metadata,
            )

            return IngestResponse(
                document_id=result.document_id,
                chunk_count=result.chunk_count,
                created_at=result.created_at,
                message=f"Successfully ingested {result.chunk_count} chunks",
            )

        except IngestionError:
            raise
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise IngestionError(
                f"Failed to ingest document: {e}",
                details={"source": request.source, "error": str(e)},
            ) from e

    async def delete(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: Document identifier to delete.

        Returns:
            bool: True if deleted, False if not found.

        Raises:
            StorageError: If deletion fails.
        """
        logger.info(f"Deleting document: {document_id}")

        deleted_count = await self._indexer.delete_document(document_id)

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return True

        logger.warning(f"Document not found: {document_id}")
        return False

    async def get_document(self, document_id: str) -> Document | None:
        """
        Retrieve a document by ID.

        Reconstructs the document from stored chunks.

        Args:
            document_id: Document identifier.

        Returns:
            Document if found, None otherwise.
        """
        chunks = await self._vector_store.get_by_document(document_id)

        if not chunks:
            return None

        # Reconstruct document from chunks
        first_chunk = chunks[0]
        metadata = first_chunk.metadata

        return Document(
            id=document_id,
            source=metadata.get("source", ""),
            chunks=chunks,
            created_at=datetime.fromisoformat(
                metadata.get("created_at", datetime.now(timezone.utc).isoformat())
            ),
            tags=metadata.get("tags", []),
            metadata={k: v for k, v in metadata.items() if k not in ("source", "tags", "created_at")},
        )

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        List all documents.

        Args:
            limit: Maximum documents to return.
            offset: Number of documents to skip.

        Returns:
            List of document summaries.
        """
        # Get all chunks and group by document
        all_chunks = await self._vector_store.filter_by_metadata(limit=10000)

        # Group by document_id
        documents: dict[str, dict] = {}
        for chunk in all_chunks:
            doc_id = chunk.document_id
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "source": chunk.metadata.get("source"),
                    "tags": chunk.metadata.get("tags", []),
                    "created_at": chunk.metadata.get("created_at"),
                    "chunk_count": 0,
                }
            documents[doc_id]["chunk_count"] += 1

        # Sort by created_at (newest first)
        sorted_docs = sorted(
            documents.values(),
            key=lambda d: d.get("created_at", ""),
            reverse=True,
        )

        return sorted_docs[offset:offset + limit]

    async def get_stats(self) -> dict:
        """
        Get ingestion statistics.

        Returns:
            Dictionary with stats.
        """
        chunk_count = await self._vector_store.count()

        return {
            "total_chunks": chunk_count,
            "embedding_model": self._embedder.model_name,
            "embedding_dimension": self._embedder.dimension,
        }
