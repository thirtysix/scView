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
    "converted when the dataset is created and can be merged with them, e.g. with a 10x sample "
    "(Convert now in the wizard converts it first, so the merge plan shows its gene and cell "
    "overlap); "
    "a Seurat Visium object keeps its tissue image and spot positions); Seurat .h5Seurat "
    "(SeuratDisk; read directly, no R needed: counts, cell metadata, reductions, an ADT assay, "
    "Visium images); "
    "10x Multiome: its .h5 or folder, an .h5ad with Peaks features or an .h5mu (RNA in X, peaks in their own "
    "file on disk; per_barcode_metrics.csv beside it adds FRiP, TSS and mitochondrial fractions; ATAC-only data "
    "and a chromatin assay inside a Seurat object are not read); "
    "nf-core/scrnaseq mtx_conversions outputs (*.h5ad); and Visium spatial data: Space Ranger's "
    "filtered matrix with its spatial/ folder (or spatial.tar.gz; drop the whole outs/ folder), "
    "Visium HD's binned_outputs folder or one square_NNNum folder (8 or 16 um bins, one dataset per bin "
    "size; 2 um and its .tar.gz are refused) or Space Ranger 4's segmented_outputs folder (cells with their "
    "outlines; a run folder holding both reads as cells), "
    "a Xenium output folder (not its .zip; only the needed files are sent), or an .h5ad with "
    "obsm['spatial']. A guided flow bundles multi-file "
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
     "detection (Scrublet or scDblFinder-style), regress-out; a note when a file's curated cell_type "
     "seems to have lost the authors' meaning for some labels (cross-tab against the author column)"),
    ("Data Assessment > Cell/Gene Filtering", "% mito floor, max counts, ATAC cuts",
     "the filtering step's thresholds, MAD or fixed: under each sample's MAD % mito bound a floor ('Max % mito floor', "
     "2% by default, since nuclei have almost no mito reads and median + 3 MADs can fall near 0%), with a warning when "
     "% mito on nuclei removes over 5% of a sample; 'Max counts/cell (0 = off)'; on 10x Multiome data with ATAC QC "
     "columns, ATAC cut-offs (ATAC counts, nucleosome signal, TSS enrichment), off until set, that the Use Signac's "
     "cuts button fills with the multiome vignette's values. Reprocess from scratch can start 'from the cells as "
     "uploaded': every cell an earlier filtering removed comes back"),
    ("Data Assessment > Sequencing QC (Cell Ranger)", "", "Cell Ranger metrics, barcode ranks"),
    ("Data Assessment > Ambient RNA", "", "soup from empty droplets, contamination estimate"),
    ("Data Assessment > Cluster QC", "",
     "each cluster's genes, counts and % mito against the typical cluster, with flags for damaged, doublet-like, "
     "stressed or cycling clusters and cautions: mito RNA in nuclei (share of cells with mito reads), a cluster "
     "with another's markers but far less RNA, markers led by non-coding RNAs or by sex genes (XIST, Y genes)"),
    ("Data Assessment > Cell hashing", "", "HTODemux / hashedDrops: singlets, doublets, negatives"),
    ("Data Assessment > Cell hashing > Within", "per lane",
     "call each lane or channel on its own (lanes counted against different tag panels); a call the checks find "
     "unsafe (tags with no counts in whole lanes, a tag with no ambient counts) is refused with the reason, and "
     "Save anyway saves it with its warnings"),
    ("Data Assessment > QC Distributions", "doublets, sample QC, sex check",
     "per-cell QC plots, per-sample QC, doublet calls, pseudobulk sample QC, sex check"),
    ("Data Assessment > QC Distributions > QC per sample", "",
     "each sample's cells and median genes, UMIs and % mito with flags; a sample sequenced as two or more "
     "libraries is flagged 'Libraries differ' when its libraries' cell-type mix (or depth) differs more than "
     "different samples alike in condition, tissue and sex do (one library may hold another sample's cells); "
     "in MAD mode each sample's thresholds, its genes cut never below a floor (200 genes for cells, 100 for "
     "spots; the Filtering step's 'Min genes/cell floor'), marked where the floor set it"),
    ("Data Assessment > QC states: RP share vs mito", "", "damaged or stressed cells per type"),
    ("Data Assessment > Cluster QC > Evidence per cluster", "",
     "why each cluster was labelled or flagged, in words, never a probability: its markers against random genes "
     "(best database p vs what 1 in 10 random lists reach), genes and % mito vs the typical cluster with the flags "
     "that fired, a doublet-like cluster's two parents and RNA, database types as shares of the evidence (named "
     "groupings only), and the label scView gave itself ('Low-quality cells') or the note the AI got; any grouping"),
    ("Data Assessment > PCA", "", "elbow, variance explained, loadings"),
    ("Data Assessment > Cell-type annotation", "",
     "name clusters by an LLM from their markers, a CellTypist model or marker-database scores (types of other "
     "tissues left out when the tissue is given or in the data), into an obs column; with Cell Ontology ids; <column>_qualified adds the genes that set a cluster apart where clusters "
     "share a label or a label is a mixture or no cell type, e.g. 'Monocytes (CCL8+ RSAD2+)'; with the LLM, "
     "scView itself labels clusters cluster QC flags as damaged 'Low-quality cells', and the LLM names the "
     "rest, told when a cluster's markers do not beat 9 of 10 random gene lists of the same size"),
    ("Data Assessment > Label from a reference dataset", "reference mapping",
     "label cells from another annotated library dataset; flags cells not in the reference"),
    ("Data Assessment > Annotation confidence", "", "label-fit flags per cell"),
    ("Data Assessment > Annotation cross-check", "", "agreement between two annotations"),
    ("Data Assessment > Annotation cross-check > Marker database check", "",
     "each label against PanglaoDB, CellMarker and ACT: supported, weak, no database support or not in "
     "database, and the 3 database types its markers fit best (of the dataset's tissue and of every tissue, when "
     "the tissue is known or given); downloads the databases on first use"),
    ("Unified View", "", "linked scatter (UMAP/tSNE/PCA) with analysis tabs beside it"),
    ("Unified View > Embedding > Tissue", "spatial",
     "spots, bins or cells on the tissue image (Visium, Visium HD); Section, Image on/off; colours, lasso as on UMAP"),
    ("Unified View > Embedding > Tissue > Draw", "cell outlines, nuclei, segmentation",
     "points, or the segmented cells or nuclei in view once few enough are in view (outlines come with a Xenium "
     "folder's boundaries files); colours, opacity, hover and lasso as on the points"),
    ("Unified View > Transcripts", "Xenium transcripts",
     "attach a run's transcripts.parquet, then coloured dots per gene over the tissue, up to 8 genes; q >= 20, no "
     "control features; checked against the count matrix"),
    ("Unified View > Markers", "", "marker table per group"),
    ("Unified View > Expression", "", "gene overlay, several genes, gene vs gene"),
    ("Unified View > Protein", "CITE-seq, CLR or DSB", "CITE-seq proteins; CLR or DSB normalisation"),
    ("Unified View > Protein > Joint RNA + protein (WNN)", "",
     "Seurat's weighted nearest neighbours: per-cell RNA / protein weights, a joint graph, WNN and "
     "protein UMAPs (in the embedding menu), WNN clusters"),
    ("Unified View > Accessibility", "ATAC",
     "only with 10x Multiome peaks (a 10x .h5 or folder, .h5ad with Peaks features, .h5mu): QC, LSI with its "
     "depth check, ATAC UMAP and clusters. Gene activity, nearest genes, motifs, peak-gene links and TSS "
     "enrichment need the genome: GRCh38 (human), mm10 or mm39 (mouse, e.g. 10x GRCm39-2024-A); hg19, mm9 "
     "and other species get a note saying so, and RNA work is unaffected"),
    ("Unified View > Accessibility > Joint RNA + ATAC (WNN)", "",
     "Seurat's weighted nearest neighbours on RNA PCs and LSI 2:50: per-cell RNA weight by group, WNN UMAP, clusters"),
    ("Unified View > Accessibility > Gene activity", "fragments (Signac GeneActivity) or peaks",
     "gene body + 2 kb upstream as an overlay and against RNA: fragments there (Signac's GeneActivity) when a fragments "
     "file is attached, else peak counts (a gene with no called peak then has none); the card says which"),
    ("Unified View > Accessibility > Differential peaks", "LR with depth, or Wilcoxon",
     "peaks per group vs the rest or one group against another (A vs B, as Signac's FindMarkers ident.1 / "
     "ident.2), chrM and unplaced scaffolds left out by default, logistic regression with a depth covariate (Seurat's LR, the default) or Wilcoxon on TF-IDF, with "
     "nearest gene, distance to TSS, class; CSV. log2FC is Signac's FoldChange (FindMarkers' avg_log2FC) by default; "
     "the older mean TF-IDF + 1 formula, about 4x smaller, is a choice and names results made before v0.79"),
    ("Unified View > Gene Sets", "AUCell, ssGSEA, GSVA",
     "browse MSigDB sets; score them with AUCell, ssGSEA or GSVA"),
    ("Unified View > Compare", "saved scores",
     "the plot coloured by 2-12 columns side by side: groupings, and under Scores every gene-set score, "
     "TF / pathway activity, gene program usage (Programs) and chromVAR motif z (Accessibility) the user has "
     "drawn (each is saved as a column, score_<set>, activity_<source>, program_k<k>_<n> or "
     "chromvar_<motif>_<id>; drawing the same one again replaces it; a program from an earlier NMF run "
     "says so) plus cell-cycle and stress scores; saved scores also show in Colour by and the histogram, "
     "and Remove beside one drops it"),
    ("Unified View > Enrichment", "ORA, GSEA", "pathway enrichment (ORA, preranked GSEA)"),
    ("Unified View > Enrichment > Diagram", "Reactome",
     "a Reactome row's pathway diagram coloured by a DE result's fold changes (also on the DE tab's "
     "enrichment); KEGG and WikiPathways rows link out"),
    ("Unified View > DE", "", "selection vs rest or between groups; cell-level and pseudobulk tests"),
    ("Unified View > DE > Audit this result", "", "donor-shuffle check of a DE result"),
    ("Unified View > Activity > TFs (CollecTRI)", "", "transcription factor activity"),
    ("Unified View > Activity > Pathways (PROGENy)", "", "pathway activity"),
    ("Unified View > Activity > Cytokine response", "CytoSig", "CytoSig cytokine response per group"),
    ("Unified View > Activity > Upstream TF candidates", "motifs",
     "TFs whose JASPAR motifs are enriched in each group's marker promoters (candidates)"),
    ("Unified View > Gene lists", "", "saved gene lists"),
    ("Unified View > Communication > One run", "ligand-receptor",
     "ligand-receptor scores between groups (LIANA consensus)"),
    ("Unified View > Communication > Compare conditions", "",
     "interactions gained, lost or changed between two conditions"),
    ("Unified View > Report", "", "one page of evidence per cluster or gene"),
    ("Unified View > Programs", "NMF", "consensus NMF gene programs"),
    ("Unified View > Copy number", "CNV", "inferred copy-number score per cell"),
    ("Unified View > Clonotypes", "TCR/BCR", "TCR/BCR repertoire"),
    ("Unified View > Perturbation", "CRISPR, Mixscape", "CRISPR guide calls, Mixscape"),
    ("Unified View > Perturbation > Within (replicate / batch)", "",
     "Mixscape compares each cell with controls of its own replicate, batch or lane"),
    ("Unified View > Perturbation > DE genes from", "",
     "Mixscape's DE genes per target from all genes (Seurat RunMixscape's rule, the default) or the variable genes; "
     "the result has each target's KO share per guide (Per guide) and per replicate"),
    ("Unified View > Perturbation > Protein by target", "",
     "with a protein panel (CITE / ECCITE-seq): each protein's shift per target in all, KO and NP cells against the "
     "controls, judged on the replicates when 'Within' names them, and targets that move a protein with no RNA KO "
     "class (which Mixscape alone calls all NP)"),
    ("Unified View > Spatial genes", "Moran's I",
     "spatially variable genes (Moran's I) on Visium spots, Visium HD bins or imaging cells, one section or all (then "
     "within each section, with each section's own I); click a gene to colour the tissue"),
    ("Unified View > Domains", "niches",
     "spatial domains (clustering that sees each spot's, bin's or cell's neighbours; k chosen or suggested by stability), "
     "neighbourhood enrichment between labels, composition niches (from labels, or on Visium spots from Spot "
     "composition's cell-type shares); colours the tissue"),
    ("Unified View > Domains > Co-occurrence", "squidpy",
     "how much more often one label is found within each distance of another than among all pairs at that range (score "
     "above 1), with a score-by-distance plot per label and a labels-by-labels heatmap, several sections each scored "
     "against its own label shares (one section's curves on request); grey heatmap cells are scores from fewer than "
     "100 pairs at that distance, too few to read; nearby is not interacting"),
    ("Unified View > Traits", "GWAS",
     "GWAS Catalog traits linked to each group's markers (leads, not causal genes); one gene's traits"),
    ("Unified View > Velocity", "",
     "RNA velocity from spliced and unspliced counts (upload them or attach a velocyto .loom, an .h5ad or a STARsolo "
     "folder): steady-state or stochastic model as scVelo; arrows on the plot, per-group coherence, phase portraits; "
     "not the dynamical model or latent time"),
    ("Unified View > Deconvolution", "bulk RNA-seq",
     "cell-type fractions of uploaded bulk RNA-seq samples, this dataset as the reference; self-check, "
     "reference download for MuSiC / CIBERSORTx"),
    ("Unified View > Spot composition", "Visium, RCTD",
     "each spot's or Visium HD bin's cell-type shares from another library dataset as the reference; tissue overlay "
     "per type, dominant type, self-check"),
    ("Unified View > Spot composition > Reference labels", "catch-all label, too many types",
     "leave reference labels out of the fit (catch-alls such as Unknown are marked, never applied) or merge them under "
     "a broader name to get under 50 types"),
    ("Unified View > Accessibility > Motifs in peaks", "JASPAR, FindMotifs, chromVAR",
     "scan every peak for the 879 JASPAR 2024 motifs (same hits as Signac's AddMotifs); motifs enriched in a group's "
     "differential peaks against GC-matched peaks, with each TF's own RNA in the group and its logo; chromVAR activity "
     "per cell: motifs by variability, groups x motifs, group against the rest, a motif on the plot"),
    ("Unified View > Accessibility > Fragment QC", "",
     "attach atac_fragments.tsv.gz + .tbi: per-cell TSS enrichment and nucleosome signal (Signac's definitions), "
     "fragment length ladder, TSS profile"),
    ("Unified View > Accessibility > Coverage tracks", "",
     "from the fragments file: depth per group for a gene or a region up to 2 Mb, with peaks and gene arrows"),
    ("Unified View > Accessibility > Peak-gene links", "",
     "Signac's LinkPeaks: which peaks within 500 kb of a gene's TSS correlate with its expression, tested against 200 matched "
     "peaks from other chromosomes; variable genes, all genes or a list, all cells or chosen groups; arcs at the locus. A "
     "candidate regulatory element, a correlation and never proof of regulation"),
    ("Unified View > Resolutions", "cluster tree, merge check",
     "clustering resolution sweep, cluster tree, merge check"),
    ("Unified View > Variable genes", "", "re-cluster under other HVG / neighbour settings"),
    ("Unified View > Plot reliability", "", "colour cells by how well the plot keeps their neighbours"),
    ("Unified View > Layouts", "UMAP stability", "alternative UMAPs: which structure is stable"),
    ("Unified View > Filter", "", "select cells by gene and metadata conditions"),
    ("Unified View > Gates > Learned gates", "",
     "Learn gate... trains a small model (L1 logistic regression) on the selected cells against the other cells "
     "of their samples; the library exports, imports and applies it to any dataset (in/out column + gate score)"),
    ("Observations", "", "cell metadata and group composition"),
    ("Gene Expression", "", "gene expression plots"),
    ("Gene Sets & Enrichment", "", "gene sets and enrichment on a full page"),
    ("Marker Genes", "", "marker table, dot plot and heatmap of the top markers per group"),
    ("Trajectory", "", "PAGA, pseudotime and the genes that change along it"),
    ("Compare > Joint embedding", "Harmony, Scanorama, BBKNN",
     "merge 2 to 32 library datasets, corrected with Harmony, Scanorama or BBKNN; datasets scView never "
     "filtered are filtered first with a single run's QC (the plan lists what each loses; it can be switched off)"),
    ("Compare > Composition > Between conditions", "",
     "cell-type proportions between conditions, donors as replicates"),
    ("Compare > Composition > Neighbourhoods (Milo)", "", "neighbourhood differential abundance"),
    ("Compare > Composition > Neighbourhood DE (miloDE)", "", "DE within neighbourhoods"),
    ("Compare > Composition > Two datasets", "", "two datasets' proportions side by side"),
    ("Compare > Plan a study", "", "donors per group needed, from this pilot"),
    ("History", "replay script", "what was done, edit and re-run, replay script"),
    ("History > Stored results", "", "every stored result, newest first; download or delete"),
    ("Download", "CELLxGENE, Vitessce, .rds",
     "h5ad, for CELLxGENE, for Vitessce, Seurat or SingleCellExperiment .rds, metadata, replay script"),
    ("AI Co-pilot", "", "this assistant; checks each claim against the fact it cites (underlined in the "
     "answer, hover for the verdict); Write methods drafts a methods paragraph and checks its numbers, "
     "tools and versions against the recorded recipe"),
)

CLOSING = "These are all of scView's tabs and tools; name a feature only by its path here."

# In the full guide only: the compact block is at its cap (tests/test_feature_guide.py). The tab stays in it.
FULL_ONLY = ("Unified View > Accessibility >", "Unified View > Embedding > Tissue > Draw", "Unified View > Transcripts",
             "Unified View > Domains > Co-occurrence", "Unified View > Spot composition > Reference labels",
             "Data Assessment > Cell-type annotation", "Data Assessment > Cluster QC", "Unified View > Compare",
             "Data Assessment > Annotation cross-check > Marker database check",
             "Data Assessment > QC Distributions > QC per sample", "Data Assessment > Cell/Gene Filtering",
             "Data Assessment > Cell hashing >", "Unified View > Perturbation >")


def compact_guide() -> str:
    """Every feature's path, grouped by tab, with a few words where the label
    alone does not say it (the data route: kept small)."""
    heads: dict[str, str] = {}
    items: dict[str, list[str]] = {}
    for path, hint, _what in FEATURES:
        if path.startswith(FULL_ONLY):
            continue
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
