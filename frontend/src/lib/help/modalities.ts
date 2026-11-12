/** Help for the Unified View tabs on other data and cross-cell questions. Keys: `tab.<id>`, `<tab>.<thing>`. */
export const MODALITIES_HELP: Record<string, string> = {
  "tab.protein":
    "Antibody (CITE-seq) levels per cell: colour by a protein, compare groups, and set a protein against its gene's RNA.",
  "tab.communication":
    "Ligand-receptor pairs that may carry signals between groups of cells, ranked across several scoring methods.",
  "tab.cnv":
    "Copy-number gains and losses inferred from expression along each chromosome, to tell tumour cells from normal ones.",
  "tab.clonotypes":
    "T and B cell receptor clonotypes: expanded clones, repertoire diversity, overlap between groups and gene usage.",
  "tab.perturbation": "Perturbation screens: what each guide or perturbation does to the cells that received it.",
  "tab.traits": "Whether each group's marker genes sit near genetic risk loci for a trait or disease (GWAS Catalog).",
  "tab.deconvolution": "Estimate cell-type proportions in bulk RNA-seq samples, with this dataset as the reference.",
  "tab.velocity": "RNA velocity: arrows from spliced and unspliced counts that show where cells are heading.",
};
