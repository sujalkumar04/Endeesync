"""
Vector Store.

Abstraction layer for Endee vector database operations.
Handles CRUD operations and similarity search.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.exceptions import StorageError
from app.models.domain import Chunk, SearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Endee vector database client.

    Provides a clean interface for vector storage, retrieval,
    and similarity search operations.

    Note: This implementation uses a file-based approach for
    the Endee vector store. Replace with actual Endee client
    when the official SDK is available.

    Example:
        store = VectorStore("./data/endee", "memories")
        await store.connect()
        await store.insert(chunk)
        results = await store.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        dimension: int = 384,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            db_path: Path to the Endee database directory.
            collection_name: Name of the collection to use.
            dimension: Vector dimension (must match embedding model).
        """
        self._db_path = Path(db_path)
        self._collection_name = collection_name
        self._dimension = dimension
        self._connected = False

        # In-memory storage (replace with Endee client)
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}
        self._texts: dict[str, str] = {}

    async def connect(self) -> None:
        """
        Establish connection to the vector store.

        Creates the database and collection if they don't exist.

        Raises:
            StorageError: If connection fails.
        """
        try:
            # Create database directory
            self._db_path.mkdir(parents=True, exist_ok=True)

            # Load existing data if available
            await self._load_from_disk()

            self._connected = True
            logger.info(
                f"Connected to vector store: {self._db_path}/{self._collection_name}"
            )

        except Exception as e:
            raise StorageError(
                f"Failed to connect to vector store: {e}",
                details={"db_path": str(self._db_path), "error": str(e)},
            ) from e

    async def disconnect(self) -> None:
        """
        Close the vector store connection.

        Ensures all pending writes are flushed.
        """
        if self._connected:
            await self._save_to_disk()
            self._connected = False
            logger.info("Disconnected from vector store")

    async def insert(self, chunk: Chunk) -> str:
        """
        Insert a single chunk into the store.

        Args:
            chunk: Chunk with embedding and metadata.

        Returns:
            Chunk ID.

        Raises:
            StorageError: If insertion fails.
        """
        self._ensure_connected()

        try:
            self._vectors[chunk.id] = np.array(chunk.embedding, dtype=np.float32)
            self._texts[chunk.id] = chunk.text
            self._metadata[chunk.id] = {
                "document_id": chunk.document_id,
                "index": chunk.index,
                **chunk.metadata,
            }

            return chunk.id

        except Exception as e:
            raise StorageError(
                f"Failed to insert chunk: {e}",
                details={"chunk_id": chunk.id, "error": str(e)},
            ) from e

    async def insert_batch(self, chunks: list[Chunk]) -> list[str]:
        """
        Insert multiple chunks in a batch.

        More efficient than calling insert() multiple times.

        Args:
            chunks: List of chunks to insert.

        Returns:
            List of chunk IDs.

        Raises:
            StorageError: If insertion fails.
        """
        self._ensure_connected()

        chunk_ids = []
        try:
            for chunk in chunks:
                self._vectors[chunk.id] = np.array(chunk.embedding, dtype=np.float32)
                self._texts[chunk.id] = chunk.text
                self._metadata[chunk.id] = {
                    "document_id": chunk.document_id,
                    "index": chunk.index,
                    **chunk.metadata,
                }
                chunk_ids.append(chunk.id)

            # Persist to disk after batch insert
            await self._save_to_disk()

            logger.debug(f"Inserted {len(chunk_ids)} chunks")
            return chunk_ids

        except Exception as e:
            raise StorageError(
                f"Failed to insert batch: {e}",
                details={"batch_size": len(chunks), "error": str(e)},
            ) from e

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            threshold: Minimum similarity score.
            filters: Optional metadata filters.

        Returns:
            List of search results ordered by similarity.

        Raises:
            StorageError: If search fails.
        """
        self._ensure_connected()

        if not self._vectors:
            return []

        try:
            query_vec = np.array(query_embedding, dtype=np.float32)

            # Normalize query vector for cosine similarity
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

            # Compute similarities
            similarities: list[tuple[str, float]] = []

            for chunk_id, vector in self._vectors.items():
                # Apply metadata filters
                if filters and not self._matches_filters(chunk_id, filters):
                    continue

                # Normalize stored vector
                vec_norm = vector / (np.linalg.norm(vector) + 1e-9)

                # Cosine similarity
                similarity = float(np.dot(query_norm, vec_norm))

                if similarity >= threshold:
                    similarities.append((chunk_id, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Take top-k
            top_results = similarities[:top_k]

            # Create SearchResult objects
            results = []
            for rank, (chunk_id, score) in enumerate(top_results, 1):
                chunk = await self.get_by_id(chunk_id)
                if chunk:
                    results.append(SearchResult(
                        chunk=chunk,
                        score=score,
                        rank=rank,
                    ))

            return results

        except Exception as e:
            raise StorageError(
                f"Search failed: {e}",
                details={"top_k": top_k, "error": str(e)},
            ) from e

    async def get_by_id(self, chunk_id: str) -> Chunk | None:
        """
        Retrieve a chunk by ID.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            Chunk if found, None otherwise.
        """
        self._ensure_connected()

        if chunk_id not in self._vectors:
            return None

        metadata = self._metadata.get(chunk_id, {})

        return Chunk(
            id=chunk_id,
            document_id=metadata.get("document_id", ""),
            text=self._texts.get(chunk_id, ""),
            embedding=self._vectors[chunk_id].tolist(),
            index=metadata.get("index", 0),
            metadata=metadata,
        )

    async def get_by_document(self, document_id: str) -> list[Chunk]:
        """
        Retrieve all chunks for a document.

        Args:
            document_id: Document identifier.

        Returns:
            List of chunks belonging to the document.
        """
        self._ensure_connected()

        chunks = []
        for chunk_id, metadata in self._metadata.items():
            if metadata.get("document_id") == document_id:
                chunk = await self.get_by_id(chunk_id)
                if chunk:
                    chunks.append(chunk)

        # Sort by index
        chunks.sort(key=lambda c: c.index)
        return chunks

    async def delete(self, chunk_id: str) -> bool:
        """
        Delete a chunk by ID.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            True if deleted, False if not found.
        """
        self._ensure_connected()

        if chunk_id not in self._vectors:
            return False

        del self._vectors[chunk_id]
        del self._texts[chunk_id]
        del self._metadata[chunk_id]

        await self._save_to_disk()
        return True

    async def delete_by_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document identifier.

        Returns:
            Number of chunks deleted.
        """
        self._ensure_connected()

        chunk_ids_to_delete = [
            cid for cid, meta in self._metadata.items()
            if meta.get("document_id") == document_id
        ]

        for chunk_id in chunk_ids_to_delete:
            del self._vectors[chunk_id]
            del self._texts[chunk_id]
            del self._metadata[chunk_id]

        if chunk_ids_to_delete:
            await self._save_to_disk()

        return len(chunk_ids_to_delete)

    async def filter_by_metadata(
        self,
        tags: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        """
        Filter chunks by metadata without similarity search.

        Args:
            tags: Filter by tags (OR logic).
            start_date: Minimum timestamp.
            end_date: Maximum timestamp.
            source: Filter by source.
            limit: Maximum results.

        Returns:
            List of matching chunks.
        """
        self._ensure_connected()

        filters = {}
        if tags:
            filters["tags"] = tags
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date
        if source:
            filters["source"] = source

        matching_chunks = []
        for chunk_id in self._metadata:
            if self._matches_filters(chunk_id, filters):
                chunk = await self.get_by_id(chunk_id)
                if chunk:
                    matching_chunks.append(chunk)

                if len(matching_chunks) >= limit:
                    break

        return matching_chunks

    async def count(self) -> int:
        """
        Get total number of chunks in the store.

        Returns:
            Total chunk count.
        """
        return len(self._vectors)

    def is_connected(self) -> bool:
        """Check if connected to the store."""
        return self._connected

    def _ensure_connected(self) -> None:
        """Ensure connection is established."""
        if not self._connected:
            raise StorageError(
                "Not connected to vector store",
                details={"db_path": str(self._db_path)},
            )

    def _matches_filters(
        self,
        chunk_id: str,
        filters: dict[str, Any],
    ) -> bool:
        """
        Check if a chunk matches the given filters.

        Args:
            chunk_id: Chunk identifier.
            filters: Filter criteria.

        Returns:
            True if matches all filters.
        """
        metadata = self._metadata.get(chunk_id, {})

        # Tag filter (OR logic)
        if "tags" in filters:
            filter_tags = set(filters["tags"])
            chunk_tags = set(metadata.get("tags", []))
            if not filter_tags.intersection(chunk_tags):
                return False

        # Source filter
        if "source" in filters:
            if metadata.get("source") != filters["source"]:
                return False

        # Date range filters
        if "start_date" in filters or "end_date" in filters:
            created_at_str = metadata.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)

                    if "start_date" in filters:
                        if created_at < filters["start_date"]:
                            return False

                    if "end_date" in filters:
                        if created_at > filters["end_date"]:
                            return False
                except (ValueError, TypeError):
                    pass

        return True

    async def _save_to_disk(self) -> None:
        """Persist data to disk."""
        try:
            collection_dir = self._db_path / self._collection_name
            collection_dir.mkdir(parents=True, exist_ok=True)

            # Save vectors
            vectors_path = collection_dir / "vectors.npy"
            if self._vectors:
                ids = list(self._vectors.keys())
                vectors = np.array([self._vectors[id_] for id_ in ids])
                np.save(vectors_path, vectors)

                # Save IDs
                ids_path = collection_dir / "ids.json"
                with open(ids_path, "w") as f:
                    json.dump(ids, f)
            else:
                # Remove files if empty
                if vectors_path.exists():
                    vectors_path.unlink()

            # Save metadata
            metadata_path = collection_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self._metadata, f, default=str)

            # Save texts
            texts_path = collection_dir / "texts.json"
            with open(texts_path, "w") as f:
                json.dump(self._texts, f)

            logger.debug(f"Saved {len(self._vectors)} vectors to disk")

        except Exception as e:
            logger.error(f"Failed to save to disk: {e}")
            raise StorageError(f"Failed to persist data: {e}") from e

    async def _load_from_disk(self) -> None:
        """Load data from disk if available."""
        collection_dir = self._db_path / self._collection_name

        if not collection_dir.exists():
            return

        try:
            # Load vectors and IDs
            vectors_path = collection_dir / "vectors.npy"
            ids_path = collection_dir / "ids.json"

            if vectors_path.exists() and ids_path.exists():
                vectors = np.load(vectors_path)
                with open(ids_path) as f:
                    ids = json.load(f)

                for i, id_ in enumerate(ids):
                    self._vectors[id_] = vectors[i]

            # Load metadata
            metadata_path = collection_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self._metadata = json.load(f)

            # Load texts
            texts_path = collection_dir / "texts.json"
            if texts_path.exists():
                with open(texts_path) as f:
                    self._texts = json.load(f)

            logger.info(f"Loaded {len(self._vectors)} vectors from disk")

        except Exception as e:
            logger.warning(f"Failed to load from disk: {e}")
            # Start fresh
            self._vectors = {}
            self._metadata = {}
            self._texts = {}
