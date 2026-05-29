"""FastAPI dependency injection helpers."""

from functools import lru_cache

from scview.config import Settings, get_settings
from scview.core.dataset_manager import DatasetManager

_dataset_manager: DatasetManager | None = None


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
