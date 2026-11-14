/** Help for the Unified View tabs on other data and cross-cell questions. Keys: `tab.<id>`, `<tab>.<thing>`. */
export const MODALITIES_HELP: Record<string, string> = {
  "tab.protein":
    "Antibody (CITE-seq) levels per cell: colour by a protein, compare groups, and set a protein against its gene's RNA.",
  "tab.communication":
    "Ligand-receptor pairs that may carry signals between groups of cells, ranked across several scoring methods, or compared between two conditions.",
  "tab.cnv":
    "Copy-number gains and losses inferred from expression along each chromosome, to tell tumour cells from normal ones.",
  "tab.clonotypes":
    "T and B cell receptor clonotypes: expanded clones, repertoire diversity, overlap between groups and gene usage.",
  "tab.perturbation":
    "Pooled CRISPR screens: each cell's guide and target, then Mixscape to tell which targeted cells were really perturbed.",
  "tab.traits": "Whether each group's marker genes sit near genetic risk loci for a trait or disease (GWAS Catalog).",
  "tab.deconvolution": "Estimate cell-type proportions in bulk RNA-seq samples, with this dataset as the reference.",
  "tab.velocity": "RNA velocity: arrows from spliced and unspliced counts that show where cells are heading.",

  // Protein (CITE-seq)
  "protein.chips":
    "Each chip is one antibody. Click it to colour the cells by its level and to draw its violin by group under the plot. Dashed chips are isotype controls: antibodies with no target, which show background binding.",
  "protein.clr":
    "CLR: each protein's count relative to the cell's other proteins (centred log-ratio). Needs no empty droplets; the antibody background is not removed.",
  "protein.dsb":
    "DSB: each protein relative to its background in empty droplets (how many background SDs above them a cell is), with each cell's technical noise taken out using the isotype controls. Around 0 is background; clearly positive is staining (Mulè et al. 2022).",
  "protein.dsb_window":
    "The empty droplets used as background: droplets that are not cells, with RNA and protein UMI totals inside these log10 ranges. The defaults are the dsb vignette's (protein 1.5 to 3, RNA up to 2.5).",
  "protein.dsb_no_empty":
    "Without empty droplets the background is estimated from the cells (dsb's ModelNegativeADTnorm): the lower of two groups of each protein's values. A protein every cell expresses can then sit near 0.",
  "protein.isotypes":
    "Isotype controls are antibodies that bind nothing specific. The ticked ones, with each cell's background level, estimate that cell's technical noise, which the step below removes from every protein.",
  "protein.group_by":
    "The groups used for the protein violin under the plot and for the heatmap below. Pick cell types or clusters to see which groups carry each protein.",
  "protein.heatmap":
    "Each column is a protein and each row a group. The colour is the protein's mean level in the group; scaled per protein, its lowest group is at the bottom of the scale and its highest at the top.",
  "protein.vs_rna":
    "Each dot is a cell: the RNA of the gene that encodes the protein (x axis) against the protein (y axis). Spearman's rho says how well they agree. Cells with protein but no RNA are common, because transcripts drop out; that is what the antibodies add.",
  "protein.wnn":
    "Weighted nearest neighbours (Hao et al. 2021): for each cell, how well its RNA neighbours and its protein neighbours predict it decides how much each counts, and one neighbour graph is built from both. Useful when the antibodies separate cells that the RNA alone blurs.",
  "protein.rna_weight":
    "Each cell's RNA weight from WNN, from 0 to 1; its protein weight is 1 minus this. High means the RNA described the cell's neighbours better, low means the proteins did. Drawn on a fixed 0 to 1 scale.",
  "protein.weight_by_group":
    "Each box is one group's RNA weights. Groups to the right lean on the RNA, groups to the left on the proteins, often because the panel has good markers for them.",

  // Protein network (STRING)
  "ppi.top_genes":
    "How many genes from the top of the chosen list go into the network. A longer list can find more interactions but makes a busier picture.",
  "ppi.score":
    "STRING's confidence that an interaction is real, from 0 to 1000. A higher minimum keeps fewer, better-supported links.",
  "ppi.colour":
    "Colour the proteins by community (groups more linked to each other than to the rest) or by log2 fold change in the source list. Circle size is the number of partners.",

  // Communication
  "communication.mode":
    "One run scores the interactions between the groups of one column. Compare conditions scores two conditions separately, on the same groups and pairs, and lists the interactions gained, lost or changed between them.",
  "communication.groupby":
    "The column whose groups act as senders (expressing the ligand) and receivers (expressing the receptor), usually cell types. Every pair of groups is scored both ways, a group with itself too. Groups under 5 cells are left out.",
  "communication.cells":
    "Score only the cells of one condition, for example the untreated ones. All cells pools every condition, which can mix signals that occur in only one of them.",
  "communication.expr_prop":
    "The ligand and the receptor must each be detected in at least this share of the sender's and the receiver's cells. Lower finds more, weaker interactions; higher keeps only broadly expressed pairs.",
  "communication.dot":
    "Columns are sender to receiver pairs and rows ligand-receptor pairs. Larger dots are more specific to that pair of groups, brighter colours more strongly expressed. Both ranks are shown as minus log10, so bigger means better.",
  "communication.circle":
    "Each node is a group and each line runs from a sender to a receiver, thicker for more (or stronger) specific interactions. Weak links and a group's signals to itself are not drawn. Click a group to filter the table to it as sender.",
  "communication.heatmap":
    "Rows are senders and columns receivers. The colour is the number of specific interactions between them (Count) or their summed ligand-receptor mean (Strength); bright cells mark the pairs of groups worth reading in the table.",
  "communication.roles":
    "For each group, how much it sends (ligand side, outgoing) and receives (receptor side, incoming) over the specific interactions. A group with long outgoing bars mainly signals to others.",
  "communication.weight":
    "Count: the number of specific interactions. Strength: their summed ligand-receptor mean, which weighs highly expressed pairs more. Specific means a specificity rank at or below the threshold given under the plot.",
  "communication.compare_conditions":
    "The column that holds the conditions, the baseline (A) and the condition to test (B). Every change reads as B against A: gained means found in B only.",
  "communication.diff_heatmap":
    "Rows are senders and columns receivers. Each square is B minus A: the change in the number (Count) or summed ligand-receptor mean (Strength) of specific interactions. Colours are centred on 0, so the two ends are gains and losses. Click a square to filter the table.",
  "communication.call":
    "Filter the table by its call: gained (specific in B only and higher there), lost (the reverse), changed (specific in both, means clearly apart) or unchanged.",
  "communication.padj_filter":
    "Keep only interactions whose donor-level test gives an adjusted p below 0.05. The test needs donors in both conditions; cells are not replicates, so without donors there is no test.",

  // Copy number
  "cnv.reference_column": "The column that names the cell types, used to pick the normal reference groups below.",
  "cnv.reference_groups":
    "Tick the groups you are confident are normal, such as immune or stromal cells from the same samples. Every cell is compared with them, so a tumour group ticked by mistake makes real changes look smaller. Leave the groups you want to test unticked.",
  "cnv.window":
    "How many neighbouring genes on a chromosome are averaged together. A larger window gives a smoother, less noisy profile but misses changes shorter than it.",
  "cnv.step":
    "Keep every n-th window along the genome. A larger step gives a smaller result that is faster to compute, with coarser detail.",
  "cnv.exclude_x":
    "Leave chromosome X out. It differs between the sexes, so when the tumour and the reference cells come from donors of different sex, X can look gained or lost when it is not.",
  "cnv.colour_by":
    "Colour the main plot by each cell's CNV score (how far its profile is from the reference overall) or by its CNV cluster (cells with similar profiles). Malignant cells usually stand out with high scores.",
  "cnv.rows_by":
    "Which column groups the heatmap rows and the score boxes. The reference column is the default; choose clusters or samples to see whether a profile belongs to one of them.",
  "cnv.heatmap":
    "Look for blocks: cells that share a red or blue stretch over the same chromosomes carry the same gain or loss, as cells of one tumour clone do. Reference rows should stay pale, near 0.",
  "cnv.score_box":
    "Each box is one group's CNV scores. Grey boxes are the reference groups and set the baseline; a group whose box sits well above them is a candidate for malignant cells.",

  // Clonotypes
  "clonotypes.contig_table":
    "The per-cell receptor chains from 10x V(D)J: Cell Ranger's filtered_contig_annotations.csv (or all_contig_annotations.csv), or an AIRR rearrangement TSV. Its barcodes are matched to this dataset's cells.",
  "clonotypes.sample":
    "When the dataset holds several 10x libraries, the same barcode can occur in each. Name the sample this contig table came from, so its barcodes are matched only within that sample's cells.",
  "clonotypes.colour_by":
    "Expansion colours cells by their clone's size band, Clone size by the number of cells in their clonotype, Chain status by whether a cell has paired chains, a single chain or more. A large clone is many cells descended from one T or B cell.",
  "clonotypes.largest":
    "The clonotypes with the most cells. 'by' picks a column to report which of its groups holds most of each clone's cells; 'in' limits the list to one group. Click a row to select its cells on the plot.",
  "clonotypes.repertoire_by":
    "The column whose groups the diversity, overlap and gene usage views compare, usually samples or cell types.",
  "clonotypes.rarefy":
    "Diversity grows with the number of cells sampled, so larger groups look more diverse. Rarefy subsamples every group to the size of the smallest, 20 seeded draws, and shows their mean and SD, for a fair comparison. Groups below the minimum cells are left out.",
  "clonotypes.usage":
    "Which V or J gene segments each group's receptors use, as a share of its clonotypes (each clone counted once) or of its cells. Rows are groups and columns genes; compare rows to see segments one group favours.",

  // Perturbation
  "perturbation.guides":
    "Step 1 gives each cell its guide and the gene that guide targets. Cells with guides for different genes become 'multiple' and cells without one 'none'. The counts below update as you change the settings; nothing is saved until you press Save.",
  "perturbation.source":
    "Where the guides come from: Cell Ranger's own guide calls, the guide UMI counts measured with the cells (called here with the thresholds beside it), or a column that already names each cell's guide or target.",
  "perturbation.min_umi":
    "A guide counts in a cell only from this many UMIs. Raise it when many cells get a guide from a few stray reads.",
  "perturbation.min_ratio":
    "With two or more guides in a cell, keep only the top one when it has at least this many times the UMIs of the second; otherwise the cell has several guides. 0 turns this off, so every guide counts.",
  "perturbation.control":
    "The target label of the non-targeting guides. Mixscape compares every target's cells with these control cells, so pick the label that marks guides with no target.",
  "perturbation.split_by":
    "Find each cell's control neighbours only within its own replicate or batch. Use it when replicates or batches differ, so a batch difference is not read as a perturbation.",
  "perturbation.colour_by":
    "class: each cell as its target's KO or NP, or NT for the controls. KO / NP / NT: the same without the target names. P(KO): each cell's probability of being perturbed, from 0 to 1.",
  "perturbation.result":
    "Per target: its cells, the share called KO and the genes that differ from the controls; a target with too few such genes is all NP. Click a row for its top genes and P(KO) histogram; DE opens its KO cells against the controls.",
  "perturbation.posterior":
    "How the probability of being perturbed spreads over this target's cells. Two clear peaks, near 0 and near 1, mean KO and NP cells separate well; many cells in the middle mean the split is uncertain.",
  "perturbation.lda":
    "A 2D view that best separates the targets' KO cells from the controls on their perturbation signatures. Targets whose cells sit together have a similar effect; the grey cloud is the controls.",

  // Traits (GWAS)
  "traits.about":
    "For each group, its marker genes are compared with the genes the GWAS Catalog maps near variants associated with each trait. A trait is listed when the overlap covers at least 3 separate loci. The empirical p compares with random gene sets that match the markers on how many traits each gene is linked to.",
  "traits.group_by":
    "The column whose groups are tested, usually cell types or clusters. Each group's top markers come from the Markers tab for this column.",
  "traits.measurements":
    "Traits under 'Other measurement' are mostly protein and metabolite levels, where a gene is often found next to its own product. They are left out unless you include them.",
  "traits.group":
    "Pick a group to list its traits. Bold traits are leads: adjusted p and empirical p both below 0.05. Click a trait for its overlap genes, then a gene to colour the plot by it and list its own traits. These are leads, not causal genes.",
  "traits.gene_traits":
    "Every trait the GWAS Catalog maps to this gene at genome-wide significance, with the studies behind it. A gene near a variant is not proven to be the gene the variant acts on.",

  // Deconvolution
  "deconvolution.about":
    "A bulk RNA-seq sample is a mixture of cells. With this dataset's cell types as the reference, each bulk sample's counts are split into the share of cells of each type that best explains them. These are estimates: a cell type missing from this dataset cannot be found, and the reference should match the bulk samples' tissue and condition.",
  "deconvolution.celltypes":
    "The column whose labels are the cell types to estimate. Only these types can be found in the bulk samples, so use the most detailed annotation you trust.",
  "deconvolution.donors":
    "The column naming each cell's donor. The weighted method uses differences between donors to weigh genes down, and the self-check holds each donor out in turn. Auto picks a donor column it finds.",
  "deconvolution.reference_cells":
    "Use only the cells of one condition as the reference. A reference that mixes conditions can misread bulk samples taken from just one of them, so pick the condition that matches your bulk samples.",
  "deconvolution.fractions":
    "Cell fractions: the share of cells. RNA fractions: the share of the sample's RNA, which is larger for types with more RNA per cell (cell size from this dataset).",
  "deconvolution.result":
    "Each bar is one bulk sample, split into the estimated share of each cell type. In the table, Fit R² is how much of the sample the fit explains; '!' marks a share far above that type's share of the reference cells (doubtful) and '*' a fit that did not settle.",
  "deconvolution.check":
    "Each donor is held out in turn. The reference built from the other donors estimates the held-out donor's own pseudobulk and mixtures of its cells with known fractions (2,000 to 10,000 cells each). The mixtures come from this dataset, so this is a best case: real bulk samples add platform and protocol differences.",
  "deconvolution.check_read":
    "Each point is one cell type in one test mixture or held-out donor: true share (x axis) against estimate (y axis). Points on the dotted line are exact, points above it over-estimates. A type with a low r or a large bias deserves less trust in your bulk results.",

  // Velocity
  "velocity.attach":
    "Velocity needs counts of spliced and unspliced reads made from the same sequencing files as this dataset (velocyto, STARsolo or kb). Several samples: pick every sample's file together. Cells are matched by barcode within their own sample, genes by id or name; cells without counts get no arrow.",
  "velocity.model":
    "Steady state fits each gene's ratio of unspliced to spliced RNA from their smoothed levels. Stochastic, the default here, also uses how they vary across neighbouring cells. Both follow scVelo.",
  "velocity.table_by":
    "The column whose groups the summary table reports, such as clusters or cell types. It does not change the velocities.",
  "velocity.n_top":
    "How many of the most variable genes (by their spliced counts) the model is fitted on. Only genes with enough spliced and unspliced counts are kept.",
  "velocity.neighbors":
    "How many neighbouring cells each cell's counts are averaged over before fitting. The dataset's own neighbour graph is used when it has at least this many; the note after a run says which.",
  "velocity.arrows":
    "Arrows show where cells are heading on the current 2D plot; Scale sets their length and Density how many are drawn. They show direction on a flat projection, not a proven fate.",
  "velocity.confidence":
    "Colour the cells by velocity confidence: how well each cell's velocity agrees with its neighbours', from 0 to 1. Low values mark regions where the arrows mean little.",
  "velocity.summary":
    "Coherence is a group's mean confidence; groups with low coherence or few velocity genes give arrows to trust less. Unspliced is the share of reads not yet spliced; very low values leave little to fit.",
  "velocity.phase":
    "Each dot is a cell's spliced and unspliced levels for this gene, averaged over the cell and its neighbours. The red line is the fitted steady-state ratio (gamma). Cells above it are read as inducing the gene, cells below as repressing it. A gene with no clear almond shape gives little to fit.",
};
