"""glibc malloc arenas (v0.78): the image and the compose file run the backend with two.

Measured at 2 CPUs on a 39,934 x 24,551 file, three pages opening Data Assessment at once: memory
held afterwards 3.98 -> 3.50 GiB (anon 3.89 -> 3.30), peak 4.40 -> 4.22; a Kang run as fast."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    if not path.exists():  # a test container mounts src and tests only
        pytest.skip(f"{path.name} is not here")
    return path.read_text()


def test_the_image_sets_two_arenas_for_every_target():
    text = _read(BACKEND / "Dockerfile")
    base = text.split("FROM base AS dev")[0]  # the base stage: dev and production both inherit it
    assert re.search(r"^ENV MALLOC_ARENA_MAX=2$", base, re.M)


def test_compose_gives_the_backend_two_arenas():
    text = _read(BACKEND.parent / "docker-compose.yml")
    backend = text.split("\n  backend:")[1].split("\n  frontend:")[0]
    assert re.search(r"^\s+- MALLOC_ARENA_MAX=2\b", backend, re.M)
