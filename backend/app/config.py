"""
EndeeSync Configuration.

Centralized configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables
    or a .env file in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="EndeeSync", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Embedding
    embedding_model: str = Field(
        default="all-minilm",
        description="Embedding model (bge-small or all-minilm)",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Embedding vector dimension",
    )

    # Chunking
    chunk_size: int = Field(
        default=400,
        description="Target chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks in tokens",
    )

    # Vector Store (Endee)
    endee_db_path: str = Field(
        default="./data/endee",
        description="Path to Endee database directory",
    )
    endee_collection_name: str = Field(
        default="memories",
        description="Collection name in Endee",
    )

    # Search
    default_top_k: int = Field(
        default=5,
        description="Default number of search results",
    )
    similarity_threshold: float = Field(
        default=0.0,
        description="Minimum similarity score threshold",
    )

    # LLM
    llm_provider: Literal["groq", "openai", "anthropic", "local"] = Field(
        default="groq",
        description="LLM provider",
    )
    llm_api_key: str = Field(
        default="",
        description="LLM API key",
    )
    llm_model: str = Field(
        default="llama-3.1-8b-instant",
        description="LLM model identifier",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Custom LLM API base URL",
    )
    llm_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens in LLM response",
    )
    llm_temperature: float = Field(
        default=0.7,
        description="LLM sampling temperature",
    )

    # CORS
    cors_origins: str = Field(
        default="*",
        description="Comma-separated CORS origins",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Settings are cached after first load for performance.

    Returns:
        Settings: Application configuration.
    """
    return Settings()
