"""Every step kind scView can record in a dataset's provenance history.

``provenance.record_step(step=...)`` is called from some forty modules; this is the
one list of what it can be called with. ``tests/test_step_kinds.py`` reads every
call in the source and fails on a kind missing here, and checks that each kind is
sorted for staleness (``results_store.ADDITIVE_STEPS`` / ``LABEL_STEPS``, or
changes the data). The frontend's History labels (``lib/stepLabel.ts``) are
checked against this file by ``lib/stepLabel.test.ts``, so a new kind needs a
label there too.

A pipeline step that does not apply to the data is recorded as ``<step>_skipped``
(``pipeline.record_skip``); those are not listed one by one.
"""

from __future__ import annotations

RECORDED_STEPS: tuple[str, ...] = (
    # Bringing data in
    "ingest",
    "integrate",
    # The pipeline (pipeline.ALL_STEPS)
    "reset_to_counts",
    "qc_metrics",
    "doublet_detection",
    "filtering",
    "ambient_correction",
    "normalization",
    "log_transform",
    "highly_variable_genes",
    "scaling",
    "pca",
    "uncorrected_embedding",
    "batch_correction",
    "neighbors",
    "clustering",
    "embeddings",
    "marker_genes",
    "enrichment",
    "cell_cycle",
    "dissociation_stress",
    "cell_type_annotation",
    "run_errors",
    # Edits and choices kept
    "subcluster",
    "merge_clusters",
    "keep_resolution",
    "keep_hvg_setting",
    "keep_layout",
    "label_cells",
    "undo_label_cells",
    "doublet_threshold",
    "mito_refilter",
    "qc_state_label",
    # Annotation
    "label_transfer",
    "annotation_confidence",
    "annotation_confidence_unknown",
    "annotation_crosscheck",
    # Other modalities
    "protein_dsb",
    "wnn",
    "hashing_demux",
    "hashing_samples",
    "vdj_attach",
    "vdj_clonotypes",
    "vdj_remove",
    "crispr_assign",
    "mixscape",
    "atac_lsi",
    "atac_wnn",
    "atac_differential",
    "atac_fragment_qc",
    "atac_links",
    "atac_motif_enrichment",
    "atac_chromvar",
    "velocity",
    # Analyses kept with the data
    "pseudotime",
    "paga",
    "cnv",
    "gene_programs",
    "tf_motif_enrichment",
    "gwas_traits",
    "bulk_deconvolution",
    "learned_gate_train",
    "learned_gate_apply",
    # Spatial
    "spatial_autocorr",
    "spot_deconvolution",
    "spatial_domains",
    "spatial_niches",
    "nhood_enrichment",
    "co_occurrence",
)

SKIPPED_SUFFIX = "_skipped"
