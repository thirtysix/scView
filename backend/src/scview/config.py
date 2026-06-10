"""Application configuration via Pydantic BaseSettings."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings are populated from environment variables or a .env file."""

    DATA_DIR: str = "/data"
    MAX_UPLOAD_SIZE_MB: int = 2048
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    CONVERTER_URL: str = "http://converter:8001"
    DEEPINFRA_API_KEY: str = ""
    MSIGDB_DIR: str = ""

    # --- RAG co-pilot (literature + tutorials corpora over pgvector) ---------
    # Postgres/Neon connection string with the pgvector extension. Empty =
    # RAG retrieval disabled (the co-pilot still grounds in in-app facts).
    RAG_DATABASE_URL: str = ""
    # DeepInfra embedding model (768-d), matching the WntHub corpus pipeline.
    RAG_EMBED_MODEL: str = "BAAI/bge-base-en-v1.5"
    RAG_EMBED_DIM: int = 768
    # DeepInfra reranker (optional; empty disables reranking).
    RAG_RERANK_MODEL: str = "Qwen/Qwen3-Reranker-4B"
    # When reranking, fetch this many hybrid candidates, then rerank down to TOP_K.
    RAG_RERANK_CANDIDATES: int = 24
    # Chat/generation + router model on DeepInfra.
    RAG_CHAT_MODEL: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # Retrieval tuning.
    RAG_TOP_K: int = 6
    RAG_VECTOR_WEIGHT: float = 0.7
    RAG_TEXT_WEIGHT: float = 0.3

    @property
    def rag_enabled(self) -> bool:
        return bool(self.RAG_DATABASE_URL and self.DEEPINFRA_API_KEY)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
