"""Application configuration via Pydantic BaseSettings."""

from functools import lru_cache

from pydantic_settings import BaseSettings

NO_UPLOAD_CAP = 1 << 62  # bytes: no upload is refused for its size


class Settings(BaseSettings):
    """Settings are populated from environment variables or a .env file."""

    DATA_DIR: str = "/data"
    # Per uploaded file, in public mode only (``max_upload_bytes``).
    MAX_UPLOAD_SIZE_MB: int = 2048
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    CONVERTER_URL: str = "http://converter:8001"
    DEEPINFRA_API_KEY: str = ""
    MSIGDB_DIR: str = ""
    # Fetch missing MSigDB collections from the Broad Institute on startup.
    MSIGDB_AUTO_DOWNLOAD: bool = True
    # Fetch the PanglaoDB + CellMarker 2.0 marker databases (and the Cell Ontology)
    # into DATA_DIR/marker_db on first use (core/marker_db.py).
    MARKER_DB_AUTO_DOWNLOAD: bool = True
    # Fetch the CytoSig cytokine-response signature into DATA_DIR/networks/cytosig on
    # first use (core/cytokine_activity.py).
    CYTOSIG_AUTO_DOWNLOAD: bool = True
    # Import public datasets from CELLxGENE Discover (core/public_import.py). Unset:
    # on in private mode, off in public mode (the server would download for anyone).
    PUBLIC_IMPORT_ENABLED: bool | None = None

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
    # Joint-embedding caps: datasets per integration and total merged cells. The
    # pipeline's scaling step densifies all genes, so memory grows ~cells x genes.
    MAX_INTEGRATE_DATASETS: int = 8
    MAX_INTEGRATE_CELLS: int = 150_000

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
    RAG_CHAT_MODEL: str = "deepseek-ai/DeepSeek-V4.1-Flash"
    # Retrieval tuning.
    RAG_TOP_K: int = 6
    RAG_VECTOR_WEIGHT: float = 0.7
    RAG_TEXT_WEIGHT: float = 0.3

    @property
    def rag_enabled(self) -> bool:
        return bool(self.RAG_DATABASE_URL and self.DEEPINFRA_API_KEY)

    @property
    def public_import_enabled(self) -> bool:
        if self.PUBLIC_IMPORT_ENABLED is not None:
            return self.PUBLIC_IMPORT_ENABLED
        return not self.is_public

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
        """The upload cap: MAX_UPLOAD_SIZE_MB in public mode. Private mode (one user, their
        own machine and disk) has none: the 10x 1.3M-cell brain is one 4.2 GB file."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024 if self.is_public else NO_UPLOAD_CAP

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
