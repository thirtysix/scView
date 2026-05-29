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
