"""
Indexer.

Handles the complete indexing pipeline:
- Text cleaning and preprocessing
- Chunking
- Embedding generation
- Metadata attachment
- Storage into Endee vector store
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.chunker import ChunkConfig, Chunker, TextChunk
from app.core.embedder import Embedder
from app.core.exceptions import IngestionError
from app.db.vector_store import VectorStore
from app.models.domain import Chunk, Document

logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """
    Configuration for the indexer.

    Attributes:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.
        embedding_model: Model to use for embeddings.
        batch_size: Batch size for embedding generation.
    """

    chunk_size: int = 1600  # ~400 tokens
    chunk_overlap: int = 200  # ~50 tokens
    embedding_model: str = "all-minilm"
    batch_size: int = 32


@dataclass
class IndexResult:
    """
    Result of an indexing operation.

    Attributes:
        document_id: Unique document identifier.
        chunk_count: Number of chunks created.
        chunk_ids: List of chunk identifiers.
        created_at: Indexing timestamp.
        source: Source file or identifier.
        metadata: Additional metadata.
    """

    document_id: str
    chunk_count: int
    chunk_ids: list[str]
    created_at: datetime
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Indexer:
    """
    Document indexer for the EndeeSync ingestion pipeline.

    Orchestrates the complete process of ingesting text documents:
    1. Read and clean text
    2. Split into overlapping chunks
    3. Generate embeddings
    4. Attach metadata
    5. Store in vector database

    Example:
        indexer = Indexer(embedder, vector_store)
        result = await indexer.index_text(
            text="My note content...",
            source="notes.txt",
            tags=["personal", "ideas"],
        )
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        config: IndexConfig | None = None,
    ) -> None:
        """
        Initialize the indexer.

        Args:
            embedder: Embedding generator instance.
            vector_store: Vector store instance.
            config: Indexer configuration.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._config = config or IndexConfig()

        # Initialize chunker with config
        chunk_config = ChunkConfig(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        self._chunker = Chunker(chunk_config)

    async def index_text(
        self,
        text: str,
        source: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IndexResult:
        """
        Index a text document.

        Args:
            text: Text content to index.
            source: Optional source identifier.
            tags: Optional list of tags.
            metadata: Optional additional metadata.

        Returns:
            IndexResult with document info and chunk IDs.

        Raises:
            IngestionError: If indexing fails.
        """
        if not text or not text.strip():
            raise IngestionError(
                "Cannot index empty text",
                details={"source": source},
            )

        try:
            # Generate document ID
            document_id = self._generate_document_id()
            created_at = datetime.now(timezone.utc)

            # Clean text
            cleaned_text = self._clean_text(text)

            logger.info(
                f"Indexing document {document_id}: "
                f"{len(cleaned_text)} chars, source={source}"
            )

            # Chunk text
            text_chunks = self._chunker.chunk(cleaned_text)

            if not text_chunks:
                raise IngestionError(
                    "Chunking produced no chunks",
                    details={"text_length": len(text), "source": source},
                )

            logger.debug(f"Created {len(text_chunks)} chunks")

            # Generate embeddings in batch
            chunk_texts = [chunk.text for chunk in text_chunks]
            embeddings = self._embedder.embed_batch(
                chunk_texts,
                batch_size=self._config.batch_size,
            )

            # Create Chunk objects with metadata
            chunks = self._create_chunks(
                document_id=document_id,
                text_chunks=text_chunks,
                embeddings=embeddings,
                source=source,
                tags=tags or [],
                created_at=created_at,
                metadata=metadata or {},
            )

            # Store in vector database
            chunk_ids = await self._vector_store.insert_batch(chunks)

            logger.info(
                f"Indexed document {document_id}: "
                f"{len(chunk_ids)} chunks stored"
            )

            return IndexResult(
                document_id=document_id,
                chunk_count=len(chunk_ids),
                chunk_ids=chunk_ids,
                created_at=created_at,
                source=source,
                metadata=metadata or {},
            )

        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(
                f"Failed to index text: {e}",
                details={"source": source, "error": str(e)},
            ) from e

    async def index_file(
        self,
        file_path: str | Path,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        encoding: str = "utf-8",
    ) -> IndexResult:
        """
        Index a text file (.txt or .md).

        Args:
            file_path: Path to the file.
            tags: Optional list of tags.
            metadata: Optional additional metadata.
            encoding: File encoding.

        Returns:
            IndexResult with document info.

        Raises:
            IngestionError: If file reading or indexing fails.
        """
        path = Path(file_path)

        # Validate file
        if not path.exists():
            raise IngestionError(
                f"File not found: {file_path}",
                details={"path": str(file_path)},
            )

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise IngestionError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}",
                details={"path": str(file_path), "extension": path.suffix},
            )

        try:
            # Read file content
            text = path.read_text(encoding=encoding)

            # Include file info in metadata
            file_metadata = {
                "file_name": path.name,
                "file_path": str(path.absolute()),
                "file_size": path.stat().st_size,
                **(metadata or {}),
            }

            return await self.index_text(
                text=text,
                source=path.name,
                tags=tags,
                metadata=file_metadata,
            )

        except UnicodeDecodeError as e:
            raise IngestionError(
                f"Failed to decode file: {e}",
                details={"path": str(file_path), "encoding": encoding},
            ) from e
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(
                f"Failed to index file: {e}",
                details={"path": str(file_path), "error": str(e)},
            ) from e

    async def index_files(
        self,
        file_paths: list[str | Path],
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[IndexResult]:
        """
        Index multiple files.

        Args:
            file_paths: List of file paths.
            tags: Tags to apply to all files.
            metadata: Metadata to apply to all files.

        Returns:
            List of IndexResult objects.
        """
        results = []
        for path in file_paths:
            try:
                result = await self.index_file(path, tags, metadata)
                results.append(result)
            except IngestionError as e:
                logger.error(f"Failed to index {path}: {e}")
                # Continue with other files

        return results

    async def delete_document(self, document_id: str) -> int:
        """
        Delete a document and all its chunks.

        Args:
            document_id: Document identifier.

        Returns:
            Number of chunks deleted.
        """
        return await self._vector_store.delete_by_document(document_id)

    def _generate_document_id(self) -> str:
        """Generate a unique document ID."""
        return f"doc_{uuid.uuid4().hex[:12]}"

    def _generate_chunk_id(self, document_id: str, index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{document_id}_chunk_{index:04d}"

    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for indexing.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text.
        """
        # Basic cleaning
        text = text.strip()

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove null bytes
        text = text.replace("\x00", "")

        # Handle markdown-specific cleaning
        # Remove excessive blank lines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text

    def _create_chunks(
        self,
        document_id: str,
        text_chunks: list[TextChunk],
        embeddings: list[list[float]],
        source: str | None,
        tags: list[str],
        created_at: datetime,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """
        Create Chunk domain objects with metadata.

        Args:
            document_id: Parent document ID.
            text_chunks: List of text chunks.
            embeddings: Corresponding embeddings.
            source: Source identifier.
            tags: List of tags.
            created_at: Timestamp.
            metadata: Additional metadata.

        Returns:
            List of Chunk objects.
        """
        chunks = []

        for i, (text_chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk_id = self._generate_chunk_id(document_id, i)

            chunk_metadata = {
                "source": source,
                "tags": tags,
                "created_at": created_at.isoformat(),
                "chunk_index": i,
                "start_char": text_chunk.start_char,
                "end_char": text_chunk.end_char,
                **metadata,
            }

            chunks.append(Chunk(
                id=chunk_id,
                document_id=document_id,
                text=text_chunk.text,
                embedding=embedding,
                index=i,
                metadata=chunk_metadata,
            ))

        return chunks

    @property
    def supported_extensions(self) -> set[str]:
        """Get supported file extensions."""
        return self.SUPPORTED_EXTENSIONS.copy()
