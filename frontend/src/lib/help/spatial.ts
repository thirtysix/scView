/** Help for the spatial and ATAC Unified View tabs. Keys: `tab.<id>`, `<tab>.<thing>`. */
export const SPATIAL_HELP: Record<string, string> = {
  "tab.spatial": "Genes whose expression forms a pattern across the tissue rather than being scattered at random.",
  "tab.transcripts": "Single transcripts drawn as dots on the tissue, gene by gene (Xenium).",
  "tab.domains":
    "Regions of the tissue with a similar make-up, which cell types sit next to each other, and recurring niches.",
  "tab.spotcomp": "Estimate which cell types make up each spot or bin, with a single-cell dataset as the reference.",
  "tab.accessibility":
    "Chromatin accessibility (10x Multiome ATAC): peaks, motifs, gene activity and links between peaks and genes.",
};
