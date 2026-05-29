"""Pydantic response / request schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """Full dataset metadata."""

    id: str
    name: str
    filename: str
    n_cells: int | None = None
    n_genes: int | None = None
    status: str = "pending"
    available_embeddings: list[str] = Field(default_factory=list)
    obs_columns: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class DatasetUploadResponse(BaseModel):
    """Returned immediately after a successful upload."""

    id: str
    name: str
    status: str


class EmbeddingInfo(BaseModel):
    """Describes a single embedding."""

    name: str
    dimensions: int = 2


class ObsColumnInfo(BaseModel):
    """Describes a single obs column."""

    name: str
    dtype: str
    n_unique: int | None = None
    values: list[Any] | None = None  # populated for categoricals
