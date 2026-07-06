"""Centralized configuration using Pydantic Settings."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All app settings loaded from .env with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    default_llm_provider: Literal["openai", "google", "groq", "openrouter"] = Field(
        default="groq", alias="DEFAULT_LLM_PROVIDER"
    )
    default_llm_model: str = Field(default="llama3-8b-8192", alias="DEFAULT_LLM_MODEL")

    # Embeddings
    jina_api_key: Optional[str] = Field(default=None, alias="JINA_API_KEY")

    # Vector DB
    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="rag_documents", alias="QDRANT_COLLECTION_NAME")

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: Optional[str] = Field(default=None, alias="LANGCHAIN_PROJECT")

    # App
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")

    # RAG params
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=10, alias="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, alias="TOP_K_RERANK")

    @field_validator("upload_dir")
    @classmethod
    def ensure_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v

    def get_llm_key(self, provider: Optional[str] = None) -> Optional[str]:
        provider = provider or self.default_llm_provider
        return {
            "openai": self.openai_api_key,
            "google": self.google_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
        }.get(provider)


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings — loaded once per process."""
    return Settings()