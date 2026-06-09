"""Cell-type annotation endpoints.

CellTypist models are tissue/system specific (there is no single universal model),
so the UI offers the catalog below and the user picks the one matching their sample.
The default, ``Immune_All_Low``, suits the common PBMC/immune case.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_CELLTYPIST_MODEL = "Immune_All_Low.pkl"


class CellTypistModel(BaseModel):
    model: str
    description: str


class CellTypistModelsResponse(BaseModel):
    default: str
    models: list[CellTypistModel]


@lru_cache(maxsize=1)
def _model_catalog() -> list[tuple[str, str]]:
    """(name, description) for every available CellTypist model. Cached; may hit the
    network once to fetch the model index, then reads the local copy."""
    from celltypist import models as ct_models

    df = ct_models.models_description()
    return [(str(r["model"]), str(r["description"])) for _, r in df.iterrows()]


@router.get("/annotation/celltypist-models", response_model=CellTypistModelsResponse)
def list_celltypist_models() -> CellTypistModelsResponse:
    """List CellTypist models (name + description) for the annotation picker."""
    try:
        items = [CellTypistModel(model=m, description=d) for m, d in _model_catalog()]
    except Exception as e:  # offline / celltypist unavailable: still return the default
        logger.warning("Could not list CellTypist models (%s); returning default only.", e)
        items = [CellTypistModel(
            model=DEFAULT_CELLTYPIST_MODEL,
            description="immune sub-populations combined from 20 tissues of 18 studies",
        )]
    return CellTypistModelsResponse(default=DEFAULT_CELLTYPIST_MODEL, models=items)
