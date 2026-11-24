/** Help for the Compare panel: Milo's neighbourhood graph. */
export const COMPARE_HELP: Record<string, string> = {
  "joint.combine":
    "Make one column of the joint dataset from one column per dataset, so every cell has a label: a cell type called Celltype in one dataset and cell_line in the other. Each cell keeps its own column's value. The source columns stay as they are. You can import labels into it later, as into a column you made.",
  "joint.combine-prefix":
    "Write the dataset's label before each value (B: parental). Use it when two datasets use one name for different things; leave it off when the same name is the same label. A suggested name that is one of the combined columns becomes name_combined, so that column stays.",
  "milo.graphK":
    "Neighbours per cell of the graph Milo builds its neighbourhoods on. Dataset's uses the dataset's own graph. A number builds a graph just for this run, so the dataset is not changed. A larger k makes bigger neighbourhoods: more cells per sample, more power, less detail. Milo's papers use 20 to 30.",
  "milo.graphPcs":
    "How many principal components that graph is built on. All uses every component the dataset's graph used. Fewer keeps the main structure and drops noise; set it to match a paper (for example 30).",
};
