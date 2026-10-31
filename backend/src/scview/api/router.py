"""Top-level API router – aggregates all v1 sub-routers."""

from fastapi import APIRouter

from scview.api.v1 import (
    activity,
    atac_fragments,
    datasets,
    embeddings,
    metadata,
    expression,
    markers,
    de,
    genesets,
    custom_genesets,
    enrichment,
    trajectory,
    assessment,
    annotation,
    labels,
    clustering,
    cnv,
    assistant,
    export,
    export_formats,
    export_rds,
    ingest,
    integration,
    cells,
    communication,
    composition,
    milo,
    milode,
    normalization,
    pca,
    planner,
    preflight,
    programs,
    system,
    report,
    layouts,
    batch_compare,
    seqqc,
    hashing,
    vdj,
    ambient,
    public_import,
    label_transfer,
    split_markers,
    marker_panel,
    structure,
    offtarget,
    cytokines,
    results,
    ppi,
    perturbation,
    ontology,
    dsb,
    spatial,
    spatial_autocorr,
    spatial_domains,
    pathways,
    tf_motifs,
    gwas,
    deconvolution,
    learned_gates,
    wnn,
    spot_deconvolution,
)

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(datasets.router, tags=["datasets"])
api_router.include_router(embeddings.router, tags=["embeddings"])
api_router.include_router(metadata.router, tags=["metadata"])
api_router.include_router(expression.router, tags=["expression"])
api_router.include_router(markers.router, tags=["markers"])
api_router.include_router(de.router, tags=["differential-expression"])
api_router.include_router(genesets.router, tags=["gene-sets"])
api_router.include_router(custom_genesets.router, tags=["gene-sets"])
api_router.include_router(enrichment.router, tags=["enrichment"])
api_router.include_router(trajectory.router, tags=["trajectory"])
api_router.include_router(assessment.router, tags=["assessment"])
api_router.include_router(annotation.router, tags=["annotation"])
api_router.include_router(labels.router, tags=["labels"])
api_router.include_router(clustering.router, tags=["clustering"])
api_router.include_router(assistant.router, tags=["assistant"])
api_router.include_router(export.router, tags=["export"])
api_router.include_router(export_formats.router, tags=["export"])
api_router.include_router(export_rds.router, tags=["export"])
api_router.include_router(ingest.router, tags=["ingest"])
api_router.include_router(integration.router, tags=["integration"])
api_router.include_router(normalization.router, tags=["assessment"])
api_router.include_router(activity.router, tags=["activity"])
api_router.include_router(cells.router, tags=["cells"])
api_router.include_router(composition.router, tags=["composition"])
api_router.include_router(communication.router, tags=["communication"])
api_router.include_router(milo.router, tags=["composition"])
api_router.include_router(milode.router, tags=["differential-expression"])
api_router.include_router(pca.router, tags=["assessment"])
api_router.include_router(system.router, tags=["system"])
api_router.include_router(preflight.router, tags=["system"])
api_router.include_router(report.router, tags=["report"])
api_router.include_router(planner.router, tags=["planner"])
api_router.include_router(programs.router, tags=["programs"])
api_router.include_router(cnv.router, tags=["cnv"])
api_router.include_router(layouts.router, tags=["embeddings"])
api_router.include_router(batch_compare.router, tags=["integration"])
api_router.include_router(seqqc.router, tags=["assessment"])
api_router.include_router(hashing.router, tags=["hashing"])
api_router.include_router(vdj.router, tags=["vdj"])
api_router.include_router(ambient.router, tags=["assessment"])
api_router.include_router(public_import.router, tags=["datasets"])
api_router.include_router(label_transfer.router, tags=["annotation"])
api_router.include_router(split_markers.router, tags=["markers"])
api_router.include_router(marker_panel.router, tags=["markers"])
api_router.include_router(structure.router, tags=["assessment"])
api_router.include_router(offtarget.router, tags=["markers"])
api_router.include_router(cytokines.router, tags=["activity"])
api_router.include_router(ppi.router, tags=["ppi"])
api_router.include_router(results.router, tags=["results"])
api_router.include_router(perturbation.router, tags=["perturbation"])
api_router.include_router(ontology.router, tags=["labels"])
api_router.include_router(dsb.router, tags=["protein"])
api_router.include_router(spatial.router, tags=["spatial"])
api_router.include_router(spatial_autocorr.router, tags=["spatial"])
api_router.include_router(spatial_domains.router, tags=["spatial"])
api_router.include_router(pathways.router, tags=["pathways"])
api_router.include_router(tf_motifs.router, tags=["activity"])
api_router.include_router(gwas.router, tags=["gwas"])
api_router.include_router(deconvolution.router, tags=["deconvolution"])
api_router.include_router(learned_gates.router, tags=["gates"])
api_router.include_router(wnn.router, tags=["protein"])
api_router.include_router(spot_deconvolution.router, tags=["spatial"])
api_router.include_router(atac_fragments.router, tags=["atac"])
