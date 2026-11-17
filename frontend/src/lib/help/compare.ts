/** Help for the Compare panel: Milo's neighbourhood graph. */
export const COMPARE_HELP: Record<string, string> = {
  "milo.graphK":
    "Neighbours per cell of the graph Milo builds its neighbourhoods on. Dataset's uses the dataset's own graph. A number builds a graph just for this run, so the dataset is not changed. A larger k makes bigger neighbourhoods: more cells per sample, more power, less detail. Milo's papers use 20 to 30.",
  "milo.graphPcs":
    "How many principal components that graph is built on. All uses every component the dataset's graph used. Fewer keeps the main structure and drops noise; set it to match a paper (for example 30).",
};
