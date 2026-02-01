"""
Embedding Generator.

Generates vector embeddings using sentence-transformers.
Supports BGE-small and all-MiniLM models.
"""

import logging
from typing import Literal

import numpy as np

from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class Embedder:
    """
    Vector embedding generator.

    Uses sentence-transformers for local embedding generation.
    Supports batch processing for efficiency.

    Example:
        embedder = Embedder("all-minilm")
        embedder.load_model()
        vector = embedder.embed("Hello world")
    """

    # Supported models and their configurations
    MODEL_CONFIGS: dict[str, dict] = {
        "bge-small": {
            "name": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "max_length": 512,
            "normalize": True,
        },
        "all-minilm": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "max_length": 256,
            "normalize": True,
        },
    }

    def __init__(
        self,
        model_name: Literal["bge-small", "all-minilm"] = "all-minilm",
    ) -> None:
        """
        Initialize the embedder.

        Args:
            model_name: Name of the embedding model to use.

        Raises:
            ValueError: If model_name is not supported.
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported: {list(self.MODEL_CONFIGS.keys())}"
            )

        self._model_name = model_name
        self._config = self.MODEL_CONFIGS[model_name]
        self._model = None

    def load_model(self) -> None:
        """
        Load the embedding model into memory.

        Call this during application startup for faster first inference.

        Raises:
            EmbeddingError: If model loading fails.
        """
        if self._model is not None:
            logger.debug("Model already loaded")
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self._config['name']}")
            self._model = SentenceTransformer(
                self._config["name"],
                trust_remote_code=True,
            )
            logger.info(f"Model loaded successfully: {self._model_name}")

        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers not installed. Run: pip install sentence-transformers",
                details={"error": str(e)},
            ) from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load model {self._model_name}: {e}",
                details={"model": self._model_name, "error": str(e)},
            ) from e

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded, loading lazily if needed."""
        if self._model is None:
            self.load_model()

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        self._ensure_loaded()

        if not text or not text.strip():
            raise EmbeddingError(
                "Cannot embed empty text",
                details={"text": text},
            )

        try:
            # Truncate if needed
            text = self._truncate_text(text)

            # Generate embedding
            embedding = self._model.encode(
                text,
                normalize_embeddings=self._config["normalize"],
                show_progress_bar=False,
            )

            return embedding.tolist()

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embedding: {e}",
                details={"text_length": len(text), "error": str(e)},
            ) from e

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        More efficient than calling embed() multiple times due to batching.

        Args:
            texts: List of input texts.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        self._ensure_loaded()

        if not texts:
            return []

        try:
            # Filter and truncate texts
            processed_texts = []
            valid_indices = []

            for i, text in enumerate(texts):
                if text and text.strip():
                    processed_texts.append(self._truncate_text(text))
                    valid_indices.append(i)

            if not processed_texts:
                return [[] for _ in texts]

            # Generate embeddings in batch
            embeddings = self._model.encode(
                processed_texts,
                batch_size=batch_size,
                normalize_embeddings=self._config["normalize"],
                show_progress_bar=show_progress,
            )

            # Map back to original indices
            result: list[list[float]] = [[] for _ in texts]
            for idx, embedding in zip(valid_indices, embeddings):
                result[idx] = embedding.tolist()

            return result

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate batch embeddings: {e}",
                details={"batch_size": len(texts), "error": str(e)},
            ) from e

    def embed_numpy(self, text: str) -> np.ndarray:
        """
        Generate embedding as numpy array.

        Args:
            text: Input text to embed.

        Returns:
            Embedding as numpy array.
        """
        embedding = self.embed(text)
        return np.array(embedding, dtype=np.float32)

    def embed_batch_numpy(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate batch embeddings as numpy array.

        Args:
            texts: List of input texts.
            batch_size: Batch size for processing.

        Returns:
            Embeddings as 2D numpy array (n_texts, dimension).
        """
        embeddings = self.embed_batch(texts, batch_size)
        return np.array(embeddings, dtype=np.float32)

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to model's max length.

        Uses a simple character-based truncation.
        The model handles actual tokenization internally.

        Args:
            text: Input text.

        Returns:
            Truncated text.
        """
        max_chars = self._config["max_length"] * 4  # Rough estimate
        if len(text) > max_chars:
            logger.debug(f"Truncating text from {len(text)} to {max_chars} chars")
            return text[:max_chars]
        return text

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity score (0.0 to 1.0).
        """
        emb1 = self.embed_numpy(text1)
        emb2 = self.embed_numpy(text2)

        # Cosine similarity (embeddings are already normalized)
        return float(np.dot(emb1, emb2))

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._config["dimension"]

    @property
    def model_id(self) -> str:
        """Get the HuggingFace model identifier."""
        return self._config["name"]

    @property
    def model_name(self) -> str:
        """Get the short model name."""
        return self._model_name

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        logger.info(f"Model unloaded: {self._model_name}")


# Singleton instance for convenience
_default_embedder: Embedder | None = None


def get_embedder(model_name: str = "all-minilm") -> Embedder:
    """
    Get or create a singleton embedder instance.

    Args:
        model_name: Model name to use.

    Returns:
        Embedder instance.
    """
    global _default_embedder
    if _default_embedder is None or _default_embedder.model_name != model_name:
        _default_embedder = Embedder(model_name)
    return _default_embedder
