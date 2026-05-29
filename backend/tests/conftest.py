"""Shared pytest fixtures for scView backend tests."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from scview.config import Settings, get_settings
from scview.dependencies import get_dataset_manager as _get_dm
from scview.core.dataset_manager import DatasetManager
from scview.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Use a single event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with required subdirectories."""
    for subdir in ("uploads", "converted", "cache"):
        (tmp_path / subdir).mkdir()
    return tmp_path


@pytest.fixture()
def test_settings(tmp_data_dir: Path) -> Settings:
    """Settings wired to the temporary data directory."""
    return Settings(DATA_DIR=str(tmp_data_dir))


@pytest.fixture()
def dataset_manager(tmp_data_dir: Path) -> DatasetManager:
    return DatasetManager(data_dir=str(tmp_data_dir))


@pytest_asyncio.fixture()
async def client(test_settings: Settings, dataset_manager: DatasetManager):
    """httpx AsyncClient bound to the FastAPI app with overridden dependencies."""

    def _override_settings():
        return test_settings

    def _override_dm():
        return dataset_manager

    app.dependency_overrides[_get_dm] = _override_dm

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    app.dependency_overrides.clear()
