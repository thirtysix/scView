// History chips of the batch-correction and neighbours steps (backend core/batch_methods.py).

const METHOD_TEXT: Record<string, string> = {
  harmony: "batch correction: Harmony (corrected PCs)",
  scanorama: "batch correction: Scanorama (corrected PCs)",
  bbknn: "batch correction: BBKNN (batch-balanced neighbour graph)",
};

/** The chip of a batch-correction / neighbours param, or null when it is not one of theirs. */
export function batchParamText(key: string, value: unknown): string | null {
  if (key === "batch_method" && typeof value === "string") return METHOD_TEXT[value] ?? `batch correction: ${value}`;
  if (key === "kept" && value === true) return "graph: BBKNN's, from batch correction (kept)";
  if (key === "graph" && value === "bbknn") return "graph: BBKNN";
  if (key === "neighbors_within_batch" && typeof value === "number")
    return `${value} neighbour${value === 1 ? "" : "s"} from each batch`;
  if (key === "n_neighbors_total" && typeof value === "number") return `${value} neighbours per cell`;
  if (key === "representation" && typeof value === "string") return `corrected PCs: ${value}`;
  if (key === "approx" && value === false) return "exact neighbours";
  return null;
}
