"""
Text Chunker.

Recursive text splitting with configurable chunk size and overlap.
Preserves semantic boundaries where possible.
"""

from dataclasses import dataclass, field
from typing import Callable

from app.core.exceptions import ChunkingError


@dataclass
class ChunkConfig:
    """
    Configuration for text chunking.

    Attributes:
        chunk_size: Target size for each chunk in characters (~400 tokens ≈ 1600 chars).
        chunk_overlap: Overlap between consecutive chunks in characters (~50 tokens ≈ 200 chars).
        separators: List of separators to split on, in order of preference.
        length_function: Function to measure text length (default: character count).
    """

    chunk_size: int = 1600  # ~400 tokens
    chunk_overlap: int = 200  # ~50 tokens
    separators: list[str] = field(default_factory=lambda: [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        "? ",    # Questions
        "! ",    # Exclamations
        "; ",    # Clauses
        ", ",    # Phrases
        " ",     # Words
        "",      # Characters (last resort)
    ])
    length_function: Callable[[str], int] = field(default=len)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


@dataclass
class TextChunk:
    """
    A text chunk with metadata.

    Attributes:
        text: The chunk text content.
        index: Position in the original document (0-indexed).
        start_char: Starting character position in original text.
        end_char: Ending character position in original text.
    """

    text: str
    index: int
    start_char: int
    end_char: int

    @property
    def char_count(self) -> int:
        """Get character count of the chunk."""
        return len(self.text)


class Chunker:
    """
    Recursive text chunker.

    Splits text into chunks of approximately equal size while
    respecting semantic boundaries (paragraphs, sentences, words).

    Uses a recursive approach:
    1. Try to split on the highest-priority separator
    2. If resulting segments are still too large, split recursively
    3. Merge small segments until target size is reached
    4. Add overlap between consecutive chunks
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        """
        Initialize the chunker.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self._config = config or ChunkConfig()

    def chunk(self, text: str) -> list[TextChunk]:
        """
        Split text into chunks with overlap.

        Args:
            text: Input text to chunk.

        Returns:
            List of TextChunk objects with metadata.

        Raises:
            ChunkingError: If chunking fails.
        """
        if not text or not text.strip():
            return []

        try:
            # Clean and normalize text
            text = self._normalize_text(text)

            # Recursively split into segments
            segments = self._split_recursive(text, self._config.separators)

            # Merge segments into chunks with target size
            merged = self._merge_segments(segments)

            # Add overlap and create TextChunk objects
            chunks = self._add_overlap(merged, text)

            return chunks

        except Exception as e:
            raise ChunkingError(
                f"Failed to chunk text: {e}",
                details={"text_length": len(text)},
            ) from e

    def chunk_texts(self, texts: list[str]) -> list[list[TextChunk]]:
        """
        Chunk multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of chunk lists, one per input text.
        """
        return [self.chunk(text) for text in texts]

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text before chunking.

        Args:
            text: Input text.

        Returns:
            Normalized text.
        """
        # Replace various whitespace with standard space
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\t", " ")

        # Remove null bytes
        text = text.replace("\x00", "")

        # Collapse multiple spaces (but preserve newlines)
        lines = text.split("\n")
        lines = [" ".join(line.split()) for line in lines]
        text = "\n".join(lines)

        return text.strip()

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """
        Recursively split text using separators.

        Args:
            text: Text to split.
            separators: Ordered list of separators to try.

        Returns:
            List of text segments.
        """
        if not text:
            return []

        length_fn = self._config.length_function
        chunk_size = self._config.chunk_size

        # Base case: text is small enough
        if length_fn(text) <= chunk_size:
            return [text]

        # No separators left - force split by characters
        if not separators:
            return self._force_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means character-level split
            splits = list(text)

        # Process each split
        result: list[str] = []
        current_segment = ""

        for i, split in enumerate(splits):
            # Add separator back (except for last split)
            segment = split
            if separator and i < len(splits) - 1:
                segment = split + separator

            # Check if adding this segment exceeds chunk size
            test_segment = current_segment + segment

            if length_fn(test_segment) <= chunk_size:
                current_segment = test_segment
            else:
                # Save current segment if not empty
                if current_segment:
                    result.append(current_segment)

                # Check if this segment alone is too large
                if length_fn(segment) > chunk_size:
                    # Recursively split with remaining separators
                    sub_splits = self._split_recursive(segment, remaining_separators)
                    result.extend(sub_splits)
                    current_segment = ""
                else:
                    current_segment = segment

        # Don't forget the last segment
        if current_segment:
            result.append(current_segment)

        return result

    def _force_split(self, text: str) -> list[str]:
        """
        Force split text by character count when no separator works.

        Args:
            text: Text to split.

        Returns:
            List of segments of approximately chunk_size.
        """
        chunk_size = self._config.chunk_size
        result = []

        for i in range(0, len(text), chunk_size):
            result.append(text[i:i + chunk_size])

        return result

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """
        Merge small segments until they reach target size.

        Args:
            segments: List of text segments.

        Returns:
            List of merged segments.
        """
        if not segments:
            return []

        length_fn = self._config.length_function
        chunk_size = self._config.chunk_size

        merged: list[str] = []
        current = ""

        for segment in segments:
            if not segment.strip():
                continue

            test = current + segment if current else segment

            if length_fn(test) <= chunk_size:
                current = test
            else:
                if current:
                    merged.append(current)
                current = segment

        if current:
            merged.append(current)

        return merged

    def _add_overlap(
        self,
        segments: list[str],
        original_text: str,
    ) -> list[TextChunk]:
        """
        Add overlap between chunks and create TextChunk objects.

        Args:
            segments: List of text segments.
            original_text: Original text for position tracking.

        Returns:
            List of TextChunk objects with overlap.
        """
        if not segments:
            return []

        overlap = self._config.chunk_overlap
        chunks: list[TextChunk] = []
        current_pos = 0

        for i, segment in enumerate(segments):
            # Find position in original text
            start_pos = original_text.find(segment[:50], current_pos)
            if start_pos == -1:
                start_pos = current_pos

            # Add overlap from previous chunk
            if i > 0 and overlap > 0:
                prev_text = segments[i - 1]
                overlap_text = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
                text = overlap_text + segment
                start_char = max(0, start_pos - len(overlap_text))
            else:
                text = segment
                start_char = start_pos

            end_char = start_char + len(text)

            chunks.append(TextChunk(
                text=text.strip(),
                index=i,
                start_char=start_char,
                end_char=end_char,
            ))

            current_pos = start_pos + len(segment)

        return chunks

    @property
    def chunk_size(self) -> int:
        """Get configured chunk size."""
        return self._config.chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Get configured overlap size."""
        return self._config.chunk_overlap


# Convenience function for simple use cases
def chunk_text(
    text: str,
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Simple function to chunk text.

    Args:
        text: Input text.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunk text strings.
    """
    config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunker = Chunker(config)
    chunks = chunker.chunk(text)
    return [c.text for c in chunks]
