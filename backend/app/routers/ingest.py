"""
Ingestion Router.

Handles text note ingestion, chunking, embedding, and storage.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_ingestion_service
from app.core.exceptions import IngestionError
from app.models.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Text Note",
    description="Ingest a text note with optional tags. The note will be chunked, embedded, and stored.",
    responses={
        201: {"description": "Document ingested successfully"},
        400: {"description": "Invalid request data"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def ingest_note(
    request: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """
    Ingest a text note into the memory store.

    Pipeline:
    1. Validate input
    2. Chunk text recursively with overlap
    3. Generate embeddings for each chunk
    4. Store vectors with metadata (timestamp, tags, source)

    Args:
        request: Ingestion request containing text and metadata.
        service: Injected ingestion service.

    Returns:
        IngestResponse: Ingestion result with document ID and chunk count.

    Raises:
        HTTPException: If ingestion fails.
    """
    logger.info(f"Ingest request: source={request.source}, tags={request.tags}")

    try:
        response = await service.ingest(request)
        logger.info(f"Ingested document {response.document_id}: {response.chunk_count} chunks")
        return response

    except IngestionError as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e.message),
        )
    except Exception as e:
        logger.exception(f"Unexpected error during ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during ingestion",
        )


@router.delete(
    "/ingest/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document",
    description="Delete a document and all its chunks from the memory store.",
    responses={
        204: {"description": "Document deleted successfully"},
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"},
    },
)
async def delete_document(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> None:
    """
    Delete a document from the memory store.

    Args:
        document_id: Unique identifier of the document to delete.
        service: Injected ingestion service.

    Raises:
        HTTPException: If document not found or deletion fails.
    """
    logger.info(f"Delete request: document_id={document_id}")

    try:
        deleted = await service.delete(document_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        logger.info(f"Deleted document: {document_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during deletion",
        )


@router.get(
    "/ingest/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Get Document",
    description="Retrieve a document by its ID.",
    responses={
        200: {"description": "Document found"},
        404: {"description": "Document not found"},
    },
)
async def get_document(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> dict:
    """
    Get a document by ID.

    Args:
        document_id: Document identifier.
        service: Injected ingestion service.

    Returns:
        Document data with chunks.
    """
    document = await service.get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    return {
        "document_id": document.id,
        "source": document.source,
        "tags": document.tags,
        "created_at": document.created_at.isoformat(),
        "chunk_count": len(document.chunks),
        "chunks": [
            {
                "id": chunk.id,
                "text": chunk.text,
                "index": chunk.index,
            }
            for chunk in document.chunks
        ],
    }


@router.get(
    "/ingest",
    status_code=status.HTTP_200_OK,
    summary="List Documents",
    description="List all ingested documents.",
)
async def list_documents(
    limit: int = 100,
    offset: int = 0,
    service: IngestionService = Depends(get_ingestion_service),
) -> dict:
    """
    List all documents.

    Args:
        limit: Maximum documents to return.
        offset: Number to skip.
        service: Injected ingestion service.

    Returns:
        List of document summaries.
    """
    documents = await service.list_documents(limit=limit, offset=offset)

    return {
        "documents": documents,
        "total": len(documents),
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/ingest/stats",
    status_code=status.HTTP_200_OK,
    summary="Get Stats",
    description="Get ingestion statistics.",
)
async def get_stats(
    service: IngestionService = Depends(get_ingestion_service),
) -> dict:
    """Get ingestion statistics."""
    return await service.get_stats()
