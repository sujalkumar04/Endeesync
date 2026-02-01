"""
RAG Service.

Retrieval-Augmented Generation pipeline with token-efficient context building.
Separates retrieval from generation for maximum flexibility.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.models.domain import SearchResult
from app.models.schemas.query import (
    QueryRequest,
    QueryResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.models.schemas.search import MetadataFilter, SearchResultItem
from app.services.llm_service import LLMResponse, LLMService
from app.services.retrieval_service import RetrievalService
from app.utils.metrics import RequestMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """
    RAG pipeline configuration.

    Attributes:
        max_context_tokens: Maximum tokens for context.
        chunk_separator: Separator between chunks in context.
        include_metadata: Whether to include chunk metadata in context.
        default_top_k: Default chunks to retrieve.
    """

    max_context_tokens: int = 4000
    chunk_separator: str = "\n\n---\n\n"
    include_metadata: bool = True
    default_top_k: int = 5


class PromptBuilder:
    """
    Token-efficient prompt construction.

    Builds prompts that maximize relevant context while
    staying within token budgets.
    """

    # System prompts optimized for clarity and token efficiency
    QA_SYSTEM_PROMPT = """You are a helpful assistant answering questions based on provided context.

RULES:
- Answer ONLY using the provided context
- If the answer isn't in the context, say "I don't have that information"
- Be concise and direct
- Cite specific parts of the context when relevant"""

    SUMMARIZE_SYSTEM_PROMPT = """You are a helpful assistant creating summaries.

RULES:
- Summarize ONLY the provided content
- Include key points and main themes
- Be concise but comprehensive
- Organize by topic if multiple subjects are present"""

    def __init__(self, llm_service: LLMService) -> None:
        """
        Initialize the prompt builder.

        Args:
            llm_service: LLM service for token estimation.
        """
        self._llm = llm_service

    def build_qa_prompt(
        self,
        question: str,
        chunks: list[SearchResult],
        max_tokens: int = 4000,
        include_metadata: bool = True,
    ) -> tuple[str, str, list[SearchResult]]:
        """
        Build a Q&A prompt with retrieved context.

        Truncates and selects chunks to fit token budget.

        Args:
            question: User's question.
            chunks: Retrieved chunks (ranked by relevance).
            max_tokens: Maximum tokens for context.
            include_metadata: Include source/date metadata.

        Returns:
            Tuple of (system_prompt, user_prompt, used_chunks).
        """
        # Reserve tokens for question and response
        question_tokens = self._llm.estimate_tokens(question)
        system_tokens = self._llm.estimate_tokens(self.QA_SYSTEM_PROMPT)
        overhead = question_tokens + system_tokens + 200  # Buffer for formatting

        available_tokens = max_tokens - overhead
        if available_tokens < 100:
            available_tokens = 500  # Minimum context

        # Build context by adding chunks until budget exhausted
        context_parts = []
        used_chunks = []
        current_tokens = 0

        for i, result in enumerate(chunks):
            chunk_text = self._format_chunk(result, i + 1, include_metadata)
            chunk_tokens = self._llm.estimate_tokens(chunk_text)

            if current_tokens + chunk_tokens > available_tokens:
                # Try truncating if this is the first chunk
                if not context_parts:
                    truncated = self._llm.truncate_to_tokens(
                        chunk_text, available_tokens
                    )
                    context_parts.append(truncated)
                    used_chunks.append(result)
                break

            context_parts.append(chunk_text)
            used_chunks.append(result)
            current_tokens += chunk_tokens

        context = "\n\n---\n\n".join(context_parts)

        user_prompt = f"""CONTEXT:
{context}

QUESTION: {question}

Please answer based on the context above."""

        return self.QA_SYSTEM_PROMPT, user_prompt, used_chunks

    def build_summarize_prompt(
        self,
        chunks: list[SearchResult],
        topic: str | None = None,
        max_tokens: int = 4000,
    ) -> tuple[str, str, list[SearchResult]]:
        """
        Build a summarization prompt.

        Args:
            chunks: Chunks to summarize.
            topic: Optional topic focus.
            max_tokens: Maximum tokens for context.

        Returns:
            Tuple of (system_prompt, user_prompt, used_chunks).
        """
        system_tokens = self._llm.estimate_tokens(self.SUMMARIZE_SYSTEM_PROMPT)
        overhead = system_tokens + 200

        available_tokens = max_tokens - overhead
        if available_tokens < 100:
            available_tokens = 500

        # Build content to summarize
        content_parts = []
        used_chunks = []
        current_tokens = 0

        for result in chunks:
            text = result.chunk.text
            tokens = self._llm.estimate_tokens(text)

            if current_tokens + tokens > available_tokens:
                # Try truncating
                if not content_parts:
                    truncated = self._llm.truncate_to_tokens(text, available_tokens)
                    content_parts.append(truncated)
                    used_chunks.append(result)
                break

            content_parts.append(text)
            used_chunks.append(result)
            current_tokens += tokens

        content = "\n\n".join(content_parts)

        if topic:
            user_prompt = f"""CONTENT TO SUMMARIZE:
{content}

Please summarize the above content, focusing on: {topic}"""
        else:
            user_prompt = f"""CONTENT TO SUMMARIZE:
{content}

Please provide a comprehensive summary of the above content."""

        return self.SUMMARIZE_SYSTEM_PROMPT, user_prompt, used_chunks

    def _format_chunk(
        self,
        result: SearchResult,
        index: int,
        include_metadata: bool,
    ) -> str:
        """
        Format a chunk for inclusion in prompt.

        Args:
            result: Search result with chunk.
            index: Chunk index (1-based).
            include_metadata: Include source/date info.

        Returns:
            Formatted chunk text.
        """
        chunk = result.chunk
        metadata = chunk.metadata

        if not include_metadata:
            return f"[{index}] {chunk.text}"

        # Build minimal metadata string
        meta_parts = []

        if source := metadata.get("source"):
            meta_parts.append(f"Source: {source}")

        if created_at := metadata.get("created_at"):
            try:
                dt = datetime.fromisoformat(created_at)
                meta_parts.append(f"Date: {dt.strftime('%Y-%m-%d')}")
            except (ValueError, TypeError):
                pass

        if meta_parts:
            meta_str = " | ".join(meta_parts)
            return f"[{index}] ({meta_str})\n{chunk.text}"

        return f"[{index}] {chunk.text}"


class RAGService:
    """
    Retrieval-Augmented Generation pipeline.

    Seamlessly combines retrieval and generation:
    1. Retrieve relevant chunks via semantic search
    2. Build token-efficient context
    3. Generate response via LLM
    4. Return answer with source attribution and timing metrics

    Example:
        service = RAGService(retrieval_service, llm_service)
        response = await service.query(request)
    """

    __slots__ = ("_retrieval", "_llm", "_prompt_builder", "_config")

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
        config: RAGConfig | None = None,
    ) -> None:
        """
        Initialize the RAG service.

        Args:
            retrieval_service: Service for semantic retrieval.
            llm_service: Service for LLM generation.
            config: Optional configuration.
        """
        self._retrieval = retrieval_service
        self._llm = llm_service
        self._prompt_builder = PromptBuilder(llm_service)
        self._config = config or RAGConfig()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Answer a question using RAG with timing metrics.

        Pipeline:
        1. Retrieve relevant chunks via semantic search
        2. Build context-aware prompt
        3. Generate answer with LLM
        4. Return answer with sources and timing metrics

        Args:
            request: Query request with question and options.

        Returns:
            QueryResponse: Generated answer with sources and timings.
        """
        metrics = RequestMetrics()

        logger.info(f"RAG query: '{request.question[:50]}...'")

        # Step 1: Retrieve relevant chunks
        with metrics.time("retrieval_ms"):
            search_results, search_timings = await self._retrieval.search_simple(
                query=request.question,
                top_k=request.top_k or self._config.default_top_k,
                tags=request.filters.tags if request.filters else None,
                start_date=request.filters.start_date if request.filters else None,
                end_date=request.filters.end_date if request.filters else None,
            )

        # Add search sub-timings
        metrics.add_timing("embedding_ms", search_timings.get("embedding_ms", 0))
        metrics.add_timing("vector_search_ms", search_timings.get("retrieval_ms", 0))

        if not search_results:
            return QueryResponse(
                question=request.question,
                answer="I don't have any relevant information to answer this question.",
                sources=[],
                model=self._llm.get_model_name(),
                timings=metrics.get_timings(),
            )

        # Step 2: Build prompt
        with metrics.time("prompt_build_ms"):
            system_prompt, user_prompt, used_chunks = self._prompt_builder.build_qa_prompt(
                question=request.question,
                chunks=search_results,
                max_tokens=self._config.max_context_tokens,
                include_metadata=self._config.include_metadata,
            )

        # Step 3: Generate answer
        with metrics.time("llm_generation_ms"):
            response: LLMResponse = await self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

        # Log token usage for cost tracking
        if response.usage:
            metrics.add_metadata("prompt_tokens", response.usage.prompt_tokens)
            metrics.add_metadata("completion_tokens", response.usage.completion_tokens)
            logger.info(
                f"RAG query tokens: prompt={response.usage.prompt_tokens}, "
                f"completion={response.usage.completion_tokens}"
            )

        # Step 4: Format response with sources
        sources = []
        if request.include_sources:
            sources = [self._to_source_item(r) for r in used_chunks]

        timings = metrics.get_timings()
        logger.info(
            f"RAG query completed: "
            f"retrieval={timings.get('retrieval_ms', 0):.1f}ms, "
            f"llm={timings.get('llm_generation_ms', 0):.1f}ms, "
            f"total={timings.get('total_request_ms', 0):.1f}ms"
        )

        return QueryResponse(
            question=request.question,
            answer=response.content,
            sources=sources,
            model=response.model,
            timings=timings,
        )

    async def summarize(self, request: SummarizeRequest) -> SummarizeResponse:
        """
        Summarize stored memories with timing metrics.

        Pipeline:
        1. Retrieve chunks matching filters
        2. Build summarization context
        3. Generate summary via LLM
        4. Return summary with stats and timings

        Args:
            request: Summarization request with options.

        Returns:
            SummarizeResponse: Generated summary with timings.
        """
        metrics = RequestMetrics()

        logger.info(f"RAG summarize: topic='{request.topic}'")

        # Step 1: Retrieve chunks (by filter or recent)
        if request.filters:
            filter_model = request.filters
        else:
            filter_model = MetadataFilter()

        with metrics.time("retrieval_ms"):
            search_results, search_timings = await self._retrieval.search_simple(
                query=request.topic or "summary of all content",
                top_k=request.max_chunks,
                tags=filter_model.tags,
                start_date=filter_model.start_date,
                end_date=filter_model.end_date,
            )

        metrics.add_timing("embedding_ms", search_timings.get("embedding_ms", 0))
        metrics.add_timing("vector_search_ms", search_timings.get("retrieval_ms", 0))

        if not search_results:
            return SummarizeResponse(
                summary="No content found matching the specified criteria.",
                chunk_count=0,
                model=self._llm.get_model_name(),
                timings=metrics.get_timings(),
            )

        # Step 2: Build prompt
        with metrics.time("prompt_build_ms"):
            system_prompt, user_prompt, used_chunks = self._prompt_builder.build_summarize_prompt(
                chunks=search_results,
                topic=request.topic,
                max_tokens=self._config.max_context_tokens,
            )

        # Step 3: Generate summary
        with metrics.time("llm_generation_ms"):
            response = await self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

        # Log token usage
        if response.usage:
            metrics.add_metadata("prompt_tokens", response.usage.prompt_tokens)
            metrics.add_metadata("completion_tokens", response.usage.completion_tokens)
            logger.info(
                f"RAG summarize tokens: prompt={response.usage.prompt_tokens}, "
                f"completion={response.usage.completion_tokens}"
            )

        timings = metrics.get_timings()
        logger.info(
            f"RAG summarize completed: "
            f"retrieval={timings.get('retrieval_ms', 0):.1f}ms, "
            f"llm={timings.get('llm_generation_ms', 0):.1f}ms, "
            f"total={timings.get('total_request_ms', 0):.1f}ms"
        )

        return SummarizeResponse(
            summary=response.content,
            chunk_count=len(used_chunks),
            model=response.model,
            timings=timings,
        )

    async def query_stream(
        self,
        request: QueryRequest,
    ):
        """
        Stream a RAG query response.

        Useful for real-time UI updates.

        Args:
            request: Query request.

        Yields:
            Text chunks as generated.
        """
        # Retrieve
        search_results, _ = await self._retrieval.search_simple(
            query=request.question,
            top_k=request.top_k or self._config.default_top_k,
            tags=request.filters.tags if request.filters else None,
            start_date=request.filters.start_date if request.filters else None,
            end_date=request.filters.end_date if request.filters else None,
        )

        if not search_results:
            yield "I don't have any relevant information to answer this question."
            return

        # Build prompt
        system_prompt, user_prompt, _ = self._prompt_builder.build_qa_prompt(
            question=request.question,
            chunks=search_results,
            max_tokens=self._config.max_context_tokens,
        )

        # Stream response
        async for chunk in self._llm.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
        ):
            yield chunk

    def _to_source_item(self, result: SearchResult) -> SearchResultItem:
        """Convert SearchResult to API response item."""
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

    @property
    def config(self) -> RAGConfig:
        """Get RAG configuration."""
        return self._config
