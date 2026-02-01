"""
EndeeSync Core Modules.

Low-level components for text processing and embedding:
- Chunker: Recursive text splitting
- Embedder: Vector embedding generation
- Indexer: Document indexing pipeline
- Exceptions: Custom exception classes
"""

from app.core.chunker import ChunkConfig, Chunker, TextChunk
from app.core.embedder import Embedder, get_embedder
from app.core.exceptions import (
    ChunkingError,
    EmbeddingError,
    EndeeError,
    IngestionError,
    LLMError,
    RAGError,
    RetrievalError,
    StorageError,
)
from app.core.indexer import IndexConfig, Indexer, IndexResult

__all__ = [
    "Chunker",
    "ChunkConfig",
    "TextChunk",
    "Embedder",
    "get_embedder",
    "Indexer",
    "IndexConfig",
    "IndexResult",
    "EndeeError",
    "ChunkingError",
    "EmbeddingError",
    "IngestionError",
    "RetrievalError",
    "StorageError",
    "LLMError",
    "RAGError",
]
