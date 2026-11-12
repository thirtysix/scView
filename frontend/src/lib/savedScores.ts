/**
 * Per-cell scores kept as obs columns (backend core/saved_scores.py): a scored gene set
 * (score_<set>) or one TF / pathway's activity (activity_<source>), saved when the user
 * draws it, plus the scores the pipeline keeps (cell cycle, dissociation stress).
 * Compare colourings offers them under "Scores"; they colour with a continuous bar.
 */
import type { ObsColumnInfo } from "@/api/types";
import { isCategoricalDtype } from "@/lib/obsColumns";
import { prettyObsLabel } from "@/lib/formatting";

/** Scores the pipeline already keeps per cell: offered with the saved ones, never removed. */
export const BUILTIN_SCORE_COLUMNS: Record<string, string> = {
  S_score: "Cell-cycle S score",
  G2M_score: "Cell-cycle G2/M score",
  dissociation_stress_score: "Dissociation-stress score",
};

const METHOD_LABELS: Record<string, string> = {
  score_genes: "score_genes",
  aucell: "AUCell",
  ssgsea: "ssGSEA",
  gsva: "GSVA",
};
const NETWORK_LABELS: Record<string, string> = { collectri: "TF activity", progeny: "Pathway activity" };

/** A per-cell score column: one the app saved, or a pipeline score (numeric). */
export function isScoreColumn(col: ObsColumnInfo): boolean {
  if (col.saved_score) return true;
  return col.name in BUILTIN_SCORE_COLUMNS && !isCategoricalDtype(col.dtype);
}

/** The panel title: the set or source name, else the column's own label. */
export function scoreTitle(col: ObsColumnInfo): string {
  return col.saved_score?.name ?? prettyObsLabel(col.name);
}

/** What the score is, in a few words: "Gene-set score (AUCell)", "TF activity". */
export function scoreDetail(col: ObsColumnInfo): string {
  const s = col.saved_score;
  if (!s) return BUILTIN_SCORE_COLUMNS[col.name] ?? "Score";
  if (s.kind === "activity") return NETWORK_LABELS[s.network ?? ""] ?? "Activity";
  return `Gene-set score (${METHOD_LABELS[s.method ?? ""] ?? s.method ?? "score_genes"})`;
}

// ---------------------------------------------------------------------------
// A score saved or removed: whoever shows the dataset's columns refreshes them
// ---------------------------------------------------------------------------

type Listener = (datasetId: string, column: string | null) => void;
const listeners = new Set<Listener>();

/** Say a score column was saved, replaced or removed (`column` null: not known). */
export function announceSavedScore(datasetId: string, column: string | null): void {
  for (const l of listeners) l(datasetId, column);
}

/** Listen for saved scores; returns the unsubscribe. */
export function onSavedScore(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
