"""
Retrieval Service.

High-performance semantic search with vector similarity and metadata filtering.
Optimized for low latency and minimal memory usage.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from app.core.embedder import Embedder
from app.core.exceptions import RetrievalError
from app.db.vector_store import VectorStore
from app.models.domain import Chunk, SearchResult
from app.models.schemas.search import MetadataFilter, SearchRequest, SearchResponse, SearchResultItem
from app.utils.metrics import RequestMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """
    Configuration for retrieval operations.

    Using frozen dataclass with slots for memory efficiency.

    Attributes:
        default_top_k: Default number of results.
        max_top_k: Maximum allowed top_k.
        default_threshold: Default similarity threshold.
        enable_reranking: Whether to enable cross-encoder reranking.
        rerank_top_n: Number of candidates for reranking.
    """

    default_top_k: int = 5
    max_top_k: int = 50
    default_threshold: float = 0.0
    enable_reranking: bool = False
    rerank_top_n: int = 20


class RetrievalService:
    """
    High-performance semantic search and retrieval service.

    Core capabilities:
    - Embed query and search for similar vectors
    - Apply metadata filters (date range, tags, source)
    - Return ranked results with similarity scores
    - Optional cross-encoder reranking for improved relevance

    Performance optimizations:
    - Lazy embedding model loading
    - Minimal object allocation
    - Pre-filtering before similarity computation
    - Batch operations where possible

    Example:
        service = RetrievalService(embedder, vector_store)
        response = await service.search(request)
    """

    __slots__ = ("_embedder", "_vector_store", "_config", "_query_cache")

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        """
        Initialize the retrieval service.

        Args:
            embedder: Embedding generation component.
            vector_store: Vector storage backend.
            config: Optional configuration.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._config = config or RetrievalConfig()

        # Simple LRU-style query cache for repeated queries
        self._query_cache: dict[str, list[float]] = {}

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform semantic similarity search.

        Pipeline:
        1. Validate and normalize request
        2. Embed query (with caching)
        3. Build metadata filters
        4. Execute vector similarity search
        5. Optional: Rerank top results
        6. Format and return response with timings

        Args:
            request: Search request with query and options.

        Returns:
            SearchResponse: Ranked search results with timing metrics.

        Raises:
            RetrievalError: If search fails.
        """
        metrics = RequestMetrics()

        try:
            # Validate top_k
            top_k = min(request.top_k, self._config.max_top_k)
            threshold = request.threshold or self._config.default_threshold

            logger.debug(f"Search: query='{request.query[:50]}...', top_k={top_k}")

            # Embed query (with timing)
            with metrics.time("embedding_ms"):
                query_embedding = self._get_query_embedding(request.query)

            # Build filter dict for vector store
            filters = self._build_filters(request.filters)

            # Execute search (with timing)
            with metrics.time("retrieval_ms"):
                results = await self._vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k * 2 if self._config.enable_reranking else top_k,
                    threshold=threshold,
                    filters=filters,
                )

            # Optional reranking (with timing)
            if self._config.enable_reranking and len(results) > top_k:
                with metrics.time("reranking_ms"):
                    results = self._rerank(request.query, results, top_k)

            # Limit to requested top_k
            results = results[:top_k]

            # Format response
            with metrics.time("formatting_ms"):
                result_items = [
                    self._to_result_item(r) for r in results
                ]

            # Log metrics
            timings = metrics.get_timings()
            logger.info(
                f"Search completed: query='{request.query[:30]}...', "
                f"results={len(result_items)}, "
                f"embedding={timings.get('embedding_ms', 0):.1f}ms, "
                f"retrieval={timings.get('retrieval_ms', 0):.1f}ms, "
                f"total={timings.get('total_request_ms', 0):.1f}ms"
            )

            return SearchResponse(
                query=request.query,
                results=result_items,
                total=len(result_items),
                timings=timings,
            )

        except RetrievalError:
            raise
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RetrievalError(
                f"Search failed: {e}",
                details={"query": request.query[:100], "error": str(e)},
            ) from e

    async def search_simple(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        tags: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> tuple[list[SearchResult], dict[str, float]]:
        """
        Simplified search interface for internal use.

        Args:
            query: Search query text.
            top_k: Number of results.
            threshold: Minimum similarity score.
            tags: Optional tag filter.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Tuple of (results, timings).
        """
        metrics = RequestMetrics()

        with metrics.time("embedding_ms"):
            query_embedding = self._get_query_embedding(query)

        filters = {}
        if tags:
            filters["tags"] = tags
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date

        with metrics.time("retrieval_ms"):
            results = await self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold,
                filters=filters if filters else None,
            )

        return results, metrics.get_timings()

    async def retrieve_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """
        Retrieve specific chunks by their IDs.

        Optimized for batch retrieval with minimal overhead.

        Args:
            chunk_ids: List of chunk identifiers.

        Returns:
            List of chunks (preserving order, skipping not found).
        """
        chunks = []
        for chunk_id in chunk_ids:
            chunk = await self._vector_store.get_by_id(chunk_id)
            if chunk:
                chunks.append(chunk)
        return chunks

    async def retrieve_by_filter(
        self,
        filters: MetadataFilter,
        limit: int = 100,
    ) -> list[Chunk]:
        """
        Retrieve chunks matching metadata filters without similarity search.

        Useful for browsing notes by tag or date range.

        Args:
            filters: Metadata filter criteria.
            limit: Maximum results to return.

        Returns:
            List of matching chunks.
        """
        return await self._vector_store.filter_by_metadata(
            tags=filters.tags,
            start_date=filters.start_date,
            end_date=filters.end_date,
            source=filters.source,
            limit=limit,
        )

    async def get_similar_chunks(
        self,
        chunk_id: str,
        top_k: int = 5,
        exclude_same_document: bool = True,
    ) -> list[SearchResult]:
        """
        Find chunks similar to a given chunk.

        Useful for "related notes" functionality.

        Args:
            chunk_id: Source chunk ID.
            top_k: Number of similar chunks.
            exclude_same_document: Exclude chunks from same document.

        Returns:
            List of similar chunks.
        """
        # Get the source chunk
        chunk = await self._vector_store.get_by_id(chunk_id)
        if not chunk:
            return []

        # Search for similar
        results = await self._vector_store.search(
            query_embedding=chunk.embedding,
            top_k=top_k + 10,  # Get extra for filtering
            threshold=0.0,
        )

        # Filter and limit
        filtered = []
        for r in results:
            # Skip self
            if r.chunk.id == chunk_id:
                continue

            # Skip same document if requested
            if exclude_same_document and r.chunk.document_id == chunk.document_id:
                continue

            filtered.append(r)
            if len(filtered) >= top_k:
                break

        return filtered

    def filter_metadata(
        self,
        results: list[SearchResult],
        filters: MetadataFilter,
    ) -> list[SearchResult]:
        """
        Post-filter search results by metadata.

        Use when filters weren't applied during search,
        or for additional filtering layers.

        Args:
            results: Search results to filter.
            filters: Metadata filter criteria.

        Returns:
            Filtered results (preserving order).
        """
        filtered = []

        for result in results:
            metadata = result.chunk.metadata

            # Tag filter (OR logic)
            if filters.tags:
                chunk_tags = set(metadata.get("tags", []))
                if not set(filters.tags).intersection(chunk_tags):
                    continue

            # Source filter
            if filters.source:
                if metadata.get("source") != filters.source:
                    continue

            # Date range
            created_at_str = metadata.get("created_at")
            if created_at_str and (filters.start_date or filters.end_date):
                try:
                    created_at = datetime.fromisoformat(created_at_str)

                    if filters.start_date and created_at < filters.start_date:
                        continue

                    if filters.end_date and created_at > filters.end_date:
                        continue
                except (ValueError, TypeError):
                    pass

            filtered.append(result)

        return filtered

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
    ) -> list[SearchResult]:
        """
        Rerank results using a cross-encoder or custom reranker.

        Cross-encoder reranking can significantly improve relevance
        by computing query-document similarity more accurately.

        Args:
            query: Original search query.
            results: Initial search results.
            top_k: Number of results to return after reranking.
            reranker: Custom reranking function. If None, uses default.

        Returns:
            Reranked results.
        """
        if not results:
            return []

        top_k = top_k or self._config.rerank_top_n

        if reranker:
            # Custom reranker
            texts = [r.chunk.text for r in results]
            scores = reranker(query, texts)
        else:
            # Default: use embedding similarity (already computed)
            # For true reranking, integrate a cross-encoder here
            scores = [r.score for r in results]

        # Sort by new scores
        ranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Update ranks and return
        reranked = []
        for i, (result, score) in enumerate(ranked[:top_k]):
            reranked.append(SearchResult(
                chunk=result.chunk,
                score=score,
                rank=i + 1,
            ))

        return reranked

    def _rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Internal reranking using default strategy."""
        return self.rerank(query, results, top_k)

    def _get_query_embedding(self, query: str) -> list[float]:
        """
        Get embedding for query with simple caching.

        Cache helps when the same query is used multiple times
        (e.g., during pagination or filter changes).

        Args:
            query: Query text.

        Returns:
            Query embedding vector.
        """
        # Simple cache (limit size to prevent memory bloat)
        cache_key = query.strip().lower()[:100]

        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        embedding = self._embedder.embed(query)

        # Keep cache small (LRU-style eviction)
        if len(self._query_cache) >= 100:
            # Remove oldest entry
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]

        self._query_cache[cache_key] = embedding
        return embedding

    def _build_filters(
        self,
        filters: MetadataFilter | None,
    ) -> dict | None:
        """
        Convert MetadataFilter to dict for vector store.

        Args:
            filters: API filter model.

        Returns:
            Filter dict or None.
        """
        if not filters:
            return None

        result = {}

        if filters.tags:
            result["tags"] = filters.tags
        if filters.source:
            result["source"] = filters.source
        if filters.start_date:
            result["start_date"] = filters.start_date
        if filters.end_date:
            result["end_date"] = filters.end_date

        return result if result else None

    def _to_result_item(self, result: SearchResult) -> SearchResultItem:
        """
        Convert internal SearchResult to API response item.

        Args:
            result: Internal search result.

        Returns:
            API response model.
        """
        metadata = result.chunk.metadata
        created_at = None

        if created_at_str := metadata.get("created_at"):
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, TypeError):
                pass

        return SearchResultItem(
            chunk_id=result.chunk.id,
            document_id=result.chunk.document_id,
            text=result.chunk.text,
            score=result.score,
            source=metadata.get("source"),
            tags=metadata.get("tags", []),
            created_at=created_at,
        )

    def clear_cache(self) -> None:
        """Clear the query embedding cache."""
        self._query_cache.clear()

    @property
    def config(self) -> RetrievalConfig:
        """Get retrieval configuration."""
        return self._config
