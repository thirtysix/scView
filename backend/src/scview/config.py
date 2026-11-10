"""Application configuration via Pydantic BaseSettings."""

from functools import lru_cache
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings

NO_UPLOAD_CAP = 1 << 62  # bytes: no upload is refused for its size
# The default language-model provider (OpenAI-compatible endpoint). Literature RAG's
# embeddings and reranker always use DeepInfra (core/rag/).
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


class Settings(BaseSettings):
    """Settings are populated from environment variables or a .env file."""

    DATA_DIR: str = "/data"
    # Per uploaded file, in public mode only (``max_upload_bytes``).
    MAX_UPLOAD_SIZE_MB: int = 2048
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    CONVERTER_URL: str = "http://converter:8001"
    DEEPINFRA_API_KEY: str = ""

    # --- Language model provider ----------------------------------------------
    # Every AI feature (co-pilot, QC report, advisor, LLM annotation, claims check)
    # calls one OpenAI-compatible chat endpoint. Empty: DeepInfra. For Privatemode
    # (confidential computing), run Edgeless's privatemode-proxy beside scView and
    # set e.g. http://privatemode-proxy:8080/v1 (docs/AI_ASSISTANT.md).
    LLM_BASE_URL: str = ""
    # Key sent to LLM_BASE_URL. Empty: DEEPINFRA_API_KEY, but only while the URL is
    # DeepInfra's, so a DeepInfra key is never sent to another provider.
    LLM_API_KEY: str = ""
    # Chat model for every AI feature (the annotator's default too). Empty:
    # RAG_CHAT_MODEL. Set it when the provider names its models differently.
    LLM_CHAT_MODEL: str = ""
    # Provider name shown in the UI and the startup log. Empty: "DeepInfra" for
    # DeepInfra's URL, else the URL's host.
    LLM_PROVIDER_NAME: str = ""
    MSIGDB_DIR: str = ""
    # Fetch missing MSigDB collections from the Broad Institute on startup.
    MSIGDB_AUTO_DOWNLOAD: bool = True
    # Fetch the PanglaoDB + CellMarker 2.0 marker databases (and the Cell Ontology)
    # into DATA_DIR/marker_db on first use (core/marker_db.py).
    MARKER_DB_AUTO_DOWNLOAD: bool = True
    ANNOTATION_AUTO_DOWNLOAD: bool = True
    GENOME_AUTO_DOWNLOAD: bool = True
    # Fetch the Human Protein Atlas tissue / cell-type RNA tables into DATA_DIR/hpa
    # on first use (core/offtarget.py).
    HPA_AUTO_DOWNLOAD: bool = True
    # Fetch the CytoSig cytokine-response signature into DATA_DIR/networks/cytosig on
    # first use (core/cytokine_activity.py).
    CYTOSIG_AUTO_DOWNLOAD: bool = True
    # Fetch STRING v12.0 physical links per species into DATA_DIR/networks on
    # first use (core/ppi.py).
    PPI_AUTO_DOWNLOAD: bool = True
    # Fetch the HGNC HCOP human-mouse ortholog table into DATA_DIR/networks on first
    # use, to map human resources to mouse datasets (core/orthologs.py).
    ORTHOLOGS_AUTO_DOWNLOAD: bool = True
    # Fetch Reactome pathway diagrams (and the pathway hierarchy) into DATA_DIR/reactome
    # on first use; only the pathway id is sent (core/reactome.py).
    REACTOME_AUTO_DOWNLOAD: bool = True
    # Fetch JASPAR 2024 motifs and UCSC promoter sequences into DATA_DIR/tf_motifs on
    # first use, for upstream TF candidates (core/tf_motifs.py).
    TF_MOTIFS_AUTO_DOWNLOAD: bool = True
    # Import public datasets from CELLxGENE Discover (core/public_import.py). Unset:
    # on in private mode, off in public mode (the server would download for anyone).
    PUBLIC_IMPORT_ENABLED: bool | None = None
    # A folder of Discover files fetched beforehand ({version_id}.h5ad, and optionally its
    # listing record {version_id}.json): an import of such a version copies it from there
    # instead of downloading (batch runs on nodes without internet). Empty: always download.
    PUBLIC_IMPORT_LOCAL_DIR: str = ""
    # The MCP server at /mcp (scview/mcp/): an AI agent's tools over these routes. Unset:
    # on in private mode, off in public mode (an agent can run anything the UI can).
    MCP_ENABLED: bool | None = None
    # Browser origins allowed to call /mcp besides localhost / 127.0.0.1 / [::1] on any
    # port (comma-separated, e.g. "http://192.168.1.5:8080"); their hosts are allowed too.
    MCP_ALLOWED_ORIGINS: str = ""

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
    # Chat/generation + router model on DeepInfra (LLM_CHAT_MODEL overrides it).
    RAG_CHAT_MODEL: str = "deepseek-ai/DeepSeek-V4.1-Flash"
    # Retrieval tuning.
    RAG_TOP_K: int = 6
    RAG_VECTOR_WEIGHT: float = 0.7
    RAG_TEXT_WEIGHT: float = 0.3

    @property
    def llm_base_url(self) -> str:
        return self.LLM_BASE_URL.strip().rstrip("/") or DEEPINFRA_BASE_URL

    @property
    def llm_is_deepinfra(self) -> bool:
        host = urlsplit(self.llm_base_url).hostname or ""
        return host == "deepinfra.com" or host.endswith(".deepinfra.com")

    @property
    def llm_api_key(self) -> str:
        """The key for ``llm_base_url``: LLM_API_KEY, else DEEPINFRA_API_KEY for DeepInfra
        only. Empty: no language model (the AI features fall back)."""
        key = self.LLM_API_KEY.strip()
        if key:
            return key
        return self.DEEPINFRA_API_KEY if self.llm_is_deepinfra else ""

    @property
    def chat_model(self) -> str:
        return self.LLM_CHAT_MODEL.strip() or self.RAG_CHAT_MODEL

    @property
    def llm_provider(self) -> str:
        if self.LLM_PROVIDER_NAME.strip():
            return self.LLM_PROVIDER_NAME.strip()
        if self.llm_is_deepinfra:
            return "DeepInfra"
        return urlsplit(self.llm_base_url).hostname or self.llm_base_url

    @property
    def rag_disabled_reason(self) -> str | None:
        """Why literature RAG is off, or None when it is on. Its embeddings and reranker
        run on DeepInfra, so with another chat provider it is off: the questions would
        otherwise still go to DeepInfra."""
        if not self.RAG_DATABASE_URL:
            return "no_database"
        if not self.llm_is_deepinfra:
            return "provider"
        if not self.DEEPINFRA_API_KEY:
            return "no_api_key"
        return None

    @property
    def rag_enabled(self) -> bool:
        return self.rag_disabled_reason is None

    @property
    def public_import_enabled(self) -> bool:
        if self.PUBLIC_IMPORT_ENABLED is not None:
            return self.PUBLIC_IMPORT_ENABLED
        return not self.is_public

    @property
    def mcp_enabled(self) -> bool:
        if self.MCP_ENABLED is not None:
            return self.MCP_ENABLED
        return not self.is_public

    @property
    def mcp_allowed_origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.MCP_ALLOWED_ORIGINS.split(",") if o.strip()]

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
