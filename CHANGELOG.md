# Changelog

All notable changes to scView are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-06-10

Initial public release.

### Added
- **AI co-pilot** — grounded, cited chat over your dataset's results and provenance plus a
  dual-corpus scRNA-seq literature/tutorials RAG. Intent routing keeps it cheap, it's context-aware
  ("what is this cluster?" resolves to the highlighted cluster and its cell type), and it works even
  before a dataset is loaded.
- **AI-assisted data assessment** — a deterministic assessor reports the state of ~15 preprocessing
  steps; an LLM advisor recommends the next steps with reasons and sized parameters (you approve).
- **Provenance — "git for the h5ad"** — every analysis step is recorded into the data with
  dependency-aware *edit & re-run from here* and an exportable, replayable recipe.
- **Cell-type annotation** — tissue-agnostic LLM-from-markers (default, no reference model to pick)
  and reference-based CellTypist; runs on any chosen clustering, writing an explicit obs column.
- **Unified View** — one linked screen: scatter (2D/3D) + tabbed Markers/Expression/Gene Sets/
  Enrichment + violin, scaling to ~200k cells via a FastAPI backend, Apache Arrow, and deck.gl.
- **Multi-format ingestion** — h5ad, 10x MEX/HDF5, loom, zarr, dense CSV, Seurat `.rds`, and
  nf-core/scrnaseq outputs, via a guided import flow.
- **On-demand analysis pipeline** — QC, doublet detection, normalization, HVG, PCA, Harmony batch
  correction, Leiden/Louvain clustering, UMAP/t-SNE, marker genes, MSigDB enrichment, cell cycle.
- **Docker self-host** — `./start.sh` or `make` on Linux, macOS, and Windows.

[Unreleased]: https://github.com/thirtysix/scView/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thirtysix/scView/releases/tag/v0.1.0
