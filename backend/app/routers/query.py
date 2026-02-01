"""
Query Router.

Handles RAG-based Q&A and summarization endpoints.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_rag_service
from app.core.exceptions import LLMError, RAGError
from app.models.schemas.query import (
    QueryRequest,
    QueryResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask Question",
    description="Ask a question and get an answer based on stored memories using RAG.",
    responses={
        200: {"description": "Question answered successfully"},
        400: {"description": "Invalid request data"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
        503: {"description": "LLM service unavailable"},
    },
)
async def ask_question(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """
    Answer a question using RAG (Retrieval-Augmented Generation).

    Pipeline:
    1. Retrieve relevant chunks via semantic search
    2. Build context prompt with retrieved chunks
    3. Generate answer via external LLM
    4. Return answer with source references

    Args:
        request: Query request with question and optional filters.
        service: Injected RAG service.

    Returns:
        QueryResponse: Generated answer with source chunks.

    Raises:
        HTTPException: If query processing fails.
    """
    logger.info(f"Query request: '{request.question[:50]}...'")

    try:
        response = await service.query(request)
        logger.info(f"Query answered with {len(response.sources)} sources")
        return response

    except LLMError as e:
        logger.error(f"LLM error: {e.message} - Details: {e.details}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {e.message}",
        )
    except RAGError as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e.message),
        )
    except Exception as e:
        logger.exception(f"Unexpected error during query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        )


@router.post(
    "/query/stream",
    status_code=status.HTTP_200_OK,
    summary="Ask Question (Streaming)",
    description="Ask a question and stream the answer in real-time.",
)
async def ask_question_stream(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """
    Stream a question answer in real-time.

    Returns a text/event-stream response for real-time UI updates.

    Args:
        request: Query request.
        service: Injected RAG service.

    Returns:
        StreamingResponse with text chunks.
    """
    logger.info(f"Streaming query: '{request.question[:50]}...'")

    async def generate():
        try:
            async for chunk in service.query_stream(request):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize Memories",
    description="Generate a summary of stored memories matching the given criteria.",
    responses={
        200: {"description": "Summary generated successfully"},
        400: {"description": "Invalid request data"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
        503: {"description": "LLM service unavailable"},
    },
)
async def summarize(
    request: SummarizeRequest,
    service: RAGService = Depends(get_rag_service),
) -> SummarizeResponse:
    """
    Summarize stored memories.

    Pipeline:
    1. Retrieve chunks matching filters
    2. Build summarization prompt
    3. Generate summary via external LLM
    4. Return summary with coverage stats

    Args:
        request: Summarization request with filters.
        service: Injected RAG service.

    Returns:
        SummarizeResponse: Generated summary.

    Raises:
        HTTPException: If summarization fails.
    """
    logger.info(f"Summarize request: topic='{request.topic}'")

    try:
        response = await service.summarize(request)
        logger.info(f"Summary generated from {response.chunk_count} chunks")
        return response

    except LLMError as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is unavailable. Please try again later.",
        )
    except RAGError as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e.message),
        )
    except Exception as e:
        logger.exception(f"Unexpected error during summarization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        )
