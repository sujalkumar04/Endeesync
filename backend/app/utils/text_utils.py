"""
Text Utilities.

Helper functions for text processing and manipulation.
"""

import re
import unicodedata
from typing import Pattern

# Precompiled regex patterns for performance
WHITESPACE_PATTERN: Pattern = re.compile(r"\s+")
MULTIPLE_NEWLINES: Pattern = re.compile(r"\n{3,}")
NON_PRINTABLE: Pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.

    Operations:
    - Strip leading/trailing whitespace
    - Normalize unicode characters (NFC)
    - Remove non-printable characters
    - Normalize line endings
    - Collapse multiple spaces (preserving single newlines)

    Args:
        text: Input text to clean.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    # Normalize unicode to composed form
    text = unicodedata.normalize("NFC", text)

    # Remove non-printable characters (except newline, tab)
    text = NON_PRINTABLE.sub("", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes
    text = text.replace("\x00", "")

    # Collapse multiple newlines to max 2
    text = MULTIPLE_NEWLINES.sub("\n\n", text)

    # Collapse multiple spaces on each line
    lines = text.split("\n")
    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in lines]
    text = "\n".join(lines)

    return text.strip()


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
    word_boundary: bool = True,
) -> str:
    """
    Truncate text to a maximum length.

    Tries to break at word boundaries when possible.

    Args:
        text: Input text.
        max_length: Maximum length (including suffix).
        suffix: String to append when truncated.
        word_boundary: Whether to break at word boundaries.

    Returns:
        Truncated text.
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    # Account for suffix length
    truncate_at = max_length - len(suffix)

    if truncate_at <= 0:
        return suffix[:max_length]

    truncated = text[:truncate_at]

    # Try to break at word boundary
    if word_boundary:
        # Find last space
        last_space = truncated.rfind(" ")
        if last_space > truncate_at // 2:  # Only if reasonable
            truncated = truncated[:last_space]

    return truncated.rstrip() + suffix


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.

    Uses tiktoken for accurate counting when available,
    falls back to character-based estimation.

    Args:
        text: Input text.
        model: Model to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    except ImportError:
        # Fallback: rough estimate (1 token ≈ 4 chars for English)
        return len(text) // 4


def estimate_chars_from_tokens(tokens: int) -> int:
    """
    Estimate character count from token count.

    Args:
        tokens: Number of tokens.

    Returns:
        Estimated character count.
    """
    return tokens * 4  # Rough estimate


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Args:
        text: Input text.

    Returns:
        List of sentences.
    """
    if not text:
        return []

    # Simple sentence splitting
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text)

    return [s.strip() for s in sentences if s.strip()]


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """
    Extract key terms from text using simple frequency analysis.

    Args:
        text: Input text.
        max_keywords: Maximum keywords to return.

    Returns:
        List of extracted keywords.
    """
    if not text:
        return []

    # Common stop words to filter out
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "it", "its", "this", "that", "these", "those", "i", "you", "he",
        "she", "we", "they", "what", "which", "who", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "also",
    }

    # Tokenize
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    # Filter and count
    word_counts: dict[str, int] = {}
    for word in words:
        if word not in stop_words:
            word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    return [word for word, _ in sorted_words[:max_keywords]]


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace to single spaces.

    Args:
        text: Input text.

    Returns:
        Text with normalized whitespace.
    """
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_markdown(text: str) -> str:
    """
    Remove markdown formatting from text.

    Args:
        text: Markdown text.

    Returns:
        Plain text.
    """
    if not text:
        return ""

    # Remove headers
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)

    # Remove links but keep text
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)

    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`(.+?)`", r"\1", text)

    # Remove images
    text = re.sub(r"!\[.*?\]\(.+?\)", "", text)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)

    # Remove blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    return clean_text(text)
