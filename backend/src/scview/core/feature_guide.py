"""What scView can do, and where each feature lives in the UI: the co-pilot's
feature guide.

``FEATURES`` holds one (path, hint, what) entry per place in the app. Each path
is the real navigation, " > "-separated, and every segment is a label the
frontend shows (a sidebar tab, a subtab, a mode or toolbar button, a card
heading or a menu item); ``frontend/src/lib/featureGuide.test.ts`` checks that
against the .tsx sources, so a renamed tab fails a test instead of sending the
model to a menu that does not exist. ``hint`` is the few words the compact
block keeps (it goes with every dataset answer: ``compact_guide``); ``what`` is
the fuller line app questions get (``full_guide``)."""

from __future__ import annotations

INTRO = "scView is a browser tool for single-cell RNA-seq."

IMPORTING = (
    "IMPORTING DATA: open the Data tab and drag files onto it, or use the file picker. "
    "Your own data is imported from local files only: there is no import from a URL; "
    "the one remote source is CELLxGENE Discover (Data > Import public dataset, when the server "
    "allows it). Supported formats: AnnData .h5ad; 10x Genomics MTX (matrix.mtx + barcodes.tsv + "
    "features/genes.tsv, optionally .gz) and 10x HDF5 (.h5); .loom; .zarr; dense CSV/TSV "
    "(optionally .gz); Seurat, SingleCellExperiment or SummarizedExperiment .rds "
    "(converted to .h5ad by the R converter service; an .rds dropped with other files is "
    "converted when the dataset is created and can be merged with them, e.g. with a 10x sample; "
    "a Seurat Visium object keeps its tissue image and spot positions); Seurat .h5Seurat "
    "(SeuratDisk; read directly, no R needed: counts, cell metadata, reductions, an ADT assay, "
    "Visium images); "
    "nf-core/scrnaseq mtx_conversions outputs (*.h5ad); and Visium spatial data: Space Ranger's "
    "filtered matrix with its spatial/ folder (or spatial.tar.gz; drop the whole outs/ folder), "
    "or an .h5ad with obsm['spatial'] (Visium HD and Xenium bundles are not read yet). A guided flow bundles multi-file "
    "inputs (e.g. the three 10x MTX files) and can merge several samples into one dataset. "
    "Each imported dataset then appears in the Data-tab library to open or manage."
)

FEATURES: tuple[tuple[str, str, str], ...] = (
    ("Data", "", "import files, the dataset library"),
    ("Data > Import public dataset", "CELLxGENE Discover",
     "search and import CELLxGENE Discover datasets"),
    ("Data Assessment", "QC + pipeline",
     "QC plots and the guided pipeline: QC, normalise, HVG, PCA, batch correction, "
     "clustering, UMAP, markers, enrichment; optional ambient RNA correction, doublet "
     "detection (Scrublet or scDblFinder-style), regress-out"),
    ("Data Assessment > Sequencing QC (Cell Ranger)", "", "Cell Ranger metrics, barcode ranks"),
    ("Data Assessment > Ambient RNA", "", "soup from empty droplets, contamination estimate"),
    ("Data Assessment > Cell hashing", "", "HTODemux / hashedDrops: singlets, doublets, negatives"),
    ("Data Assessment > QC Distributions", "doublets, sample QC, sex check",
     "per-cell QC plots, per-sample QC, doublet calls, pseudobulk sample QC, sex check"),
    ("Data Assessment > QC states: RP share vs mito", "", "damaged or stressed cells per type"),
    ("Data Assessment > PCA", "", "elbow, variance explained, loadings"),
    ("Data Assessment > Label from a reference dataset", "reference mapping",
     "label cells from another annotated library dataset; flags cells not in the reference"),
    ("Data Assessment > Annotation confidence", "", "label-fit flags per cell"),
    ("Data Assessment > Annotation cross-check", "", "agreement between two annotations"),
    ("Unified View", "", "linked scatter (UMAP/tSNE/PCA) with analysis tabs beside it"),
    ("Unified View > Embedding > Tissue", "spatial",
     "spots or cells on the tissue image (Visium); Section, Image on/off; colours, lasso as on UMAP"),
    ("Unified View > Markers", "", "marker table per group"),
    ("Unified View > Expression", "", "gene overlay, several genes, gene vs gene"),
    ("Unified View > Protein", "CITE-seq, CLR or DSB", "CITE-seq proteins; CLR or DSB normalisation"),
    ("Unified View > Gene Sets", "AUCell, ssGSEA, GSVA",
     "browse MSigDB sets; score them with AUCell, ssGSEA or GSVA"),
    ("Unified View > Enrichment", "ORA, GSEA", "pathway enrichment (ORA, preranked GSEA)"),
    ("Unified View > Enrichment > Diagram", "Reactome",
     "a Reactome row's pathway diagram coloured by a DE result's fold changes (also on the DE tab's "
     "enrichment); KEGG and WikiPathways rows link out"),
    ("Unified View > DE", "", "selection vs rest or between groups; cell-level and pseudobulk tests"),
    ("Unified View > DE > Audit this result", "donor shuffle", "donor-shuffle check of a DE result"),
    ("Unified View > Activity > TFs (CollecTRI)", "", "transcription factor activity"),
    ("Unified View > Activity > Pathways (PROGENy)", "", "pathway activity"),
    ("Unified View > Activity > Cytokine response", "CytoSig", "CytoSig cytokine response per group"),
    ("Unified View > Activity > Upstream TF candidates", "motifs",
     "TFs whose JASPAR motifs are enriched in each group's marker promoters (candidates)"),
    ("Unified View > Gene lists", "", "saved gene lists"),
    ("Unified View > Communication > One run", "ligand-receptor, LIANA",
     "ligand-receptor scores between groups (LIANA consensus)"),
    ("Unified View > Communication > Compare conditions", "",
     "interactions gained, lost or changed between two conditions"),
    ("Unified View > Report", "", "one page of evidence per cluster or gene"),
    ("Unified View > Programs", "NMF", "consensus NMF gene programs"),
    ("Unified View > Copy number", "CNV", "inferred copy-number score per cell"),
    ("Unified View > Clonotypes", "TCR/BCR", "TCR/BCR repertoire"),
    ("Unified View > Perturbation", "CRISPR, Mixscape", "CRISPR guide calls, Mixscape"),
    ("Unified View > Spatial genes", "Moran's I, spatial data only",
     "spatially variable genes (Moran's I) on Visium spots or imaging cells; click a gene to colour the tissue"),
    ("Unified View > Traits", "GWAS",
     "GWAS Catalog traits linked to each group's markers (leads, not causal genes); one gene's traits"),
    ("Unified View > Deconvolution", "bulk RNA-seq",
     "cell-type fractions of uploaded bulk RNA-seq samples, this dataset as the reference; self-check, "
     "reference download for MuSiC / CIBERSORTx"),
    ("Unified View > Spot composition", "Visium, RCTD-style",
     "each spot's cell-type shares from another library dataset as the reference; tissue overlay per type, "
     "dominant type, self-check"),
    ("Unified View > Resolutions", "cluster tree, merge check",
     "clustering resolution sweep, cluster tree, merge check"),
    ("Unified View > Variable genes", "", "re-cluster under other HVG / neighbour settings"),
    ("Unified View > Plot reliability", "", "colour cells by how well the plot keeps their neighbours"),
    ("Unified View > Layouts", "UMAP stability", "alternative UMAPs: which structure is stable"),
    ("Unified View > Filter", "", "select cells by gene and metadata conditions"),
    ("Observations", "", "cell metadata and group composition"),
    ("Gene Expression", "", "gene expression plots"),
    ("Gene Sets & Enrichment", "", "gene sets and enrichment on a full page"),
    ("Marker Genes", "", "marker table, dot plot and heatmap of the top markers per group"),
    ("Trajectory", "", "PAGA, pseudotime and the genes that change along it"),
    ("Compare > Joint embedding", "Harmony, Scanorama, BBKNN",
     "merge 2+ library datasets, corrected with Harmony, Scanorama or BBKNN"),
    ("Compare > Composition > Between conditions", "",
     "cell-type proportions between conditions, donors as replicates"),
    ("Compare > Composition > Neighbourhoods (Milo)", "", "neighbourhood differential abundance"),
    ("Compare > Composition > Neighbourhood DE (miloDE)", "", "DE within neighbourhoods"),
    ("Compare > Composition > Two datasets", "", "two datasets' proportions side by side"),
    ("Compare > Plan a study", "power", "donors per group needed, from this pilot"),
    ("History", "replay script", "what was done, edit and re-run, replay script"),
    ("History > Stored results", "", "every stored result, newest first; download or delete"),
    ("Download", "CELLxGENE, Vitessce, .rds",
     "h5ad, for CELLxGENE, for Vitessce, Seurat or SingleCellExperiment .rds, metadata, replay script"),
    ("AI Co-pilot", "", "this assistant; checks each claim against the fact it cites"),
)

CLOSING = "These are all of scView's tabs and tools; name a feature only by its path here."


def compact_guide() -> str:
    """Every feature's path, grouped by tab, with a few words where the label
    alone does not say it (the data route: kept small)."""
    heads: dict[str, str] = {}
    items: dict[str, list[str]] = {}
    for path, hint, _what in FEATURES:
        tab, _, rest = path.partition(" > ")
        heads.setdefault(tab, tab)
        if not rest:
            heads[tab] = f"{tab} ({hint})" if hint else tab
        else:
            items.setdefault(tab, []).append(f"{rest} ({hint})" if hint else rest)
    lines = [f"- {head}" + (f": {'; '.join(_shared_parents(items[tab]))}" if tab in items else "")
             for tab, head in heads.items()]
    return "\n".join([*lines, f"- {CLOSING}"])


def _shared_parents(entries: list[str]) -> list[str]:
    """Neighbouring entries under one parent written once: "Activity > TFs; Activity >
    Pathways" -> "Activity > TFs / Pathways" (keeps the compact guide under its cap)."""
    out: list[str] = []
    parent = None
    for e in entries:
        head, sep, rest = e.partition(" > ")
        if sep and head == parent:
            out[-1] += f" / {rest}"
        else:
            out.append(e)
            parent = head if sep else e.split(" (")[0]
    return out


def full_guide() -> str:
    """The intro, importing and every feature (app questions)."""
    return "\n".join([f"- {INTRO} {IMPORTING}",
                      *(f"- {path}: {what}" for path, _hint, what in FEATURES),
                      f"- {CLOSING}"])
