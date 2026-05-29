"""Top-level API router – aggregates all v1 sub-routers."""

from fastapi import APIRouter

from scview.api.v1 import (
    datasets,
    embeddings,
    metadata,
    expression,
    markers,
    genesets,
    enrichment,
    trajectory,
    assessment,
    export,
    ingest,
    ws,
)

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(datasets.router, tags=["datasets"])
api_router.include_router(embeddings.router, tags=["embeddings"])
api_router.include_router(metadata.router, tags=["metadata"])
api_router.include_router(expression.router, tags=["expression"])
api_router.include_router(markers.router, tags=["markers"])
api_router.include_router(genesets.router, tags=["gene-sets"])
api_router.include_router(enrichment.router, tags=["enrichment"])
api_router.include_router(trajectory.router, tags=["trajectory"])
api_router.include_router(assessment.router, tags=["assessment"])
api_router.include_router(export.router, tags=["export"])
api_router.include_router(ingest.router, tags=["ingest"])
api_router.include_router(ws.router, tags=["websocket"])
