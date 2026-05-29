"""FastAPI dependency injection helpers."""

from scview.config import Settings, get_settings
from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion.session import IngestSessionManager

_dataset_manager: DatasetManager | None = None
_ingest_session_manager: IngestSessionManager | None = None


def get_settings_dep() -> Settings:
    """Return the cached Settings instance (usable as a FastAPI dependency)."""
    return get_settings()


def get_dataset_manager() -> DatasetManager:
    """Return a singleton DatasetManager instance."""
    global _dataset_manager
    if _dataset_manager is None:
        settings = get_settings()
        _dataset_manager = DatasetManager(data_dir=settings.DATA_DIR)
    return _dataset_manager


def get_ingest_session_manager() -> IngestSessionManager:
    """Return a singleton IngestSessionManager instance."""
    global _ingest_session_manager
    if _ingest_session_manager is None:
        settings = get_settings()
        _ingest_session_manager = IngestSessionManager(data_dir=settings.DATA_DIR)
    return _ingest_session_manager
