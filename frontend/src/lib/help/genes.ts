/** Help for the gene-centred Unified View tabs. Keys: `tab.<id>` for the tab itself, `<tab>.<thing>` inside it. */
export const GENES_HELP: Record<string, string> = {
  "tab.markers":
    "Genes that set each group apart from the rest. Use it to see what makes a cluster distinct and to name cell types.",
  "tab.expression":
    "Colour the cells by a gene's expression and compare groups in the violin. Use it to check where a known marker is on.",
  "tab.genesets":
    "Browse gene sets (MSigDB and your own) and score each cell for one, to see where a pathway or programme is active.",
  "tab.enrichment":
    "Pathways over-represented among a group's marker genes, so a cluster reads as biology rather than a gene list.",
  "tab.de":
    "Differential expression: a selection against the rest, or one group against another, with a volcano plot.",
  "tab.activity":
    "Transcription factor and pathway activity per cell, inferred from each one's known target genes.",
  "tab.genelists": "Save and name your own gene lists, and reuse them in the other tabs and in later sessions.",
  "tab.report": "A written summary of one cluster or one gene, built from scView's own results.",
  "tab.programs":
    "Gene programmes: sets of genes that rise and fall together across cells, and how much each cell uses each one.",
};
