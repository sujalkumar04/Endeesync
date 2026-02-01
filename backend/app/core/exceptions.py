"""
Custom Exceptions.

Hierarchical exception classes for error handling across the application.
"""


class EndeeError(Exception):
    """Base exception for all EndeeSync errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message.
            details: Optional additional details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ChunkingError(EndeeError):
    """Raised when text chunking fails."""

    pass


class EmbeddingError(EndeeError):
    """Raised when embedding generation fails."""

    pass


class StorageError(EndeeError):
    """Raised when vector store operations fail."""

    pass


class IngestionError(EndeeError):
    """Raised when the ingestion pipeline fails."""

    pass


class RetrievalError(EndeeError):
    """Raised when search/retrieval fails."""

    pass


class LLMError(EndeeError):
    """Raised when LLM API calls fail."""

    pass


class RAGError(EndeeError):
    """Raised when RAG pipeline fails."""

    pass
