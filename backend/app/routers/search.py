"""
Search Router.

Handles semantic similarity search with metadata filtering.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_retrieval_service
from app.core.exceptions import RetrievalError
from app.models.schemas.search import SearchRequest, SearchResponse
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Search",
    description="Search for similar content using vector similarity with optional metadata filters.",
    responses={
        200: {"description": "Search completed successfully"},
        400: {"description": "Invalid request data"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def search(
    request: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """
    Perform semantic similarity search.

    Pipeline:
    1. Embed query text
    2. Search vector store for similar chunks
    3. Apply metadata filters (time range, tags)
    4. Return top-k results with scores

    Args:
        request: Search request with query and filters.
        service: Injected retrieval service.

    Returns:
        SearchResponse: Ranked search results with similarity scores.

    Raises:
        HTTPException: If search fails.
    """
    logger.info(f"Search request: query='{request.query[:50]}...', top_k={request.top_k}")

    try:
        response = await service.search(request)
        logger.info(f"Search returned {response.total} results")
        return response

    except RetrievalError as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e.message),
        )
    except Exception as e:
        logger.exception(f"Unexpected error during search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during search",
        )


@router.post(
    "/search/similar/{chunk_id}",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Find Similar Chunks",
    description="Find chunks similar to a given chunk ID.",
)
async def find_similar(
    chunk_id: str,
    top_k: int = 5,
    exclude_same_document: bool = True,
    service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """
    Find chunks similar to a given chunk.

    Useful for "related notes" functionality.

    Args:
        chunk_id: Source chunk ID.
        top_k: Number of similar chunks.
        exclude_same_document: Exclude chunks from same document.
        service: Injected retrieval service.

    Returns:
        SearchResponse with similar chunks.
    """
    try:
        results = await service.get_similar_chunks(
            chunk_id=chunk_id,
            top_k=top_k,
            exclude_same_document=exclude_same_document,
        )

        # Convert to response format
        from datetime import datetime
        from app.models.schemas.search import SearchResultItem

        items = []
        for r in results:
            metadata = r.chunk.metadata
            created_at = None
            if created_at_str := metadata.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except (ValueError, TypeError):
                    pass

            items.append(SearchResultItem(
                chunk_id=r.chunk.id,
                document_id=r.chunk.document_id,
                text=r.chunk.text,
                score=r.score,
                source=metadata.get("source"),
                tags=metadata.get("tags", []),
                created_at=created_at,
            ))

        return SearchResponse(
            query=f"similar_to:{chunk_id}",
            results=items,
            total=len(items),
        )

    except Exception as e:
        logger.exception(f"Error finding similar chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find similar chunks",
        )
