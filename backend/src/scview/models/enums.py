"""Enumerations used across the application."""

from enum import Enum


class DatasetStatus(str, Enum):
    pending = "pending"
    converting = "converting"
    ready = "ready"
    error = "error"


class EmbeddingType(str, Enum):
    umap = "umap"
    tsne = "tsne"
    pca = "pca"
    other = "other"
