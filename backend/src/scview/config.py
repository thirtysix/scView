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

    # --- Deployment posture --------------------------------------------------
    # "private" (default): assume a single trusted user on localhost — the only
    # supported posture out of the box. "public": you are exposing scView beyond
    # localhost; this turns on input hardening + a startup self-check and makes
    # the optional shared-secret gate (ACCESS_TOKEN) advisable. Public, multi-user
    # hosting still REQUIRES auth + rate limiting + quotas in front (see SECURITY).
    DEPLOYMENT_MODE: str = "private"
    # Optional shared-secret gate. When set, every /api request must present it as
    # `Authorization: Bearer <token>` or `X-Access-Token`. A coarse gate (one
    # secret for everyone), not per-user auth — pair it with a reverse proxy.
    ACCESS_TOKEN: str = ""
    # Always-on input caps for LLM-bound text (cheap hygiene, both modes).
    MAX_QUERY_CHARS: int = 4000
    MAX_QUERY_WORDS: int = 400
    MAX_HISTORY_MESSAGES: int = 50
    # Reject oversized JSON bodies on non-upload routes before parsing.
    MAX_JSON_BODY_KB: int = 512

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
    def is_public(self) -> bool:
        return self.DEPLOYMENT_MODE.strip().lower() == "public"

    def deployment_warnings(self) -> list[str]:
        """Posture problems to surface loudly at startup in public mode."""
        if not self.is_public:
            return []
        warns: list[str] = []
        if not self.ACCESS_TOKEN:
            warns.append(
                "ACCESS_TOKEN is unset: every endpoint (LLM + compute) is OPEN to "
                "anyone who can reach this port. Set ACCESS_TOKEN and/or front scView "
                "with an authenticating reverse proxy."
            )
        if any("localhost" in o or "127.0.0.1" in o for o in self.cors_origins_list):
            warns.append(
                "CORS_ORIGINS still allows localhost — set it to your real public origin(s)."
            )
        warns.append(
            "Public mode does NOT add rate limiting or per-user quotas. A single client "
            "can still exhaust your LLM budget and CPU. Add those before real multi-user use "
            "(see SECURITY.md)."
        )
        return warns

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
