/**
 * Per-cell scores kept as obs columns (backend core/saved_scores.py): a scored gene set
 * (score_<set>), one TF / pathway's activity (activity_<source>), one gene program's usage
 * (program_k<k>_<n>) or one motif's chromVAR z (chromvar_<motif>_<id>), saved when the user
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

/** What the score is, in a few words: "Gene-set score (AUCell)", "TF activity",
 *  "Program usage: GZMB, PRF1, NKG7", "chromVAR z (MA0080.6)"; ", earlier run" when the
 *  dataset's gene programs or chromVAR run is a later one than the score came from. */
export function scoreDetail(col: ObsColumnInfo): string {
  const s = col.saved_score;
  if (!s) return BUILTIN_SCORE_COLUMNS[col.name] ?? "Score";
  const earlier = s.earlier_run ? ", earlier run" : "";
  if (s.kind === "activity") return NETWORK_LABELS[s.network ?? ""] ?? "Activity";
  if (s.kind === "program") {
    const what = s.method === "NMF usage" ? "Program usage (unscaled)" : "Program usage";
    const genes = (s.top_genes ?? []).slice(0, 3).join(", ");
    return `${what}${genes ? `: ${genes}` : ""}${earlier}`;
  }
  if (s.kind === "chromvar") return `chromVAR z${s.motif_id ? ` (${s.motif_id})` : ""}${earlier}`;
  return `Gene-set score (${METHOD_LABELS[s.method ?? ""] ?? s.method ?? "score_genes"})`;
}

/** The hover text of a score in Compare: what it is, where it came from, its column. */
export function scoreTooltip(col: ObsColumnInfo): string {
  const s = col.saved_score;
  const base = `${scoreTitle(col)}, ${scoreDetail(col)} (column ${col.name})`;
  if (!s?.earlier_run) return base;
  const run = s.kind === "program" ? "an earlier gene programs (NMF) run" : "an earlier chromVAR run";
  return `${base}. From ${run}: the dataset's ${s.kind === "program" ? "programs" : "chromVAR scores"} now come from a later run.`;
}

/** A saved record entry as far as matching needs it (api/savedScores.ts SavedScoreRecord). */
interface RecordLike {
  kind: string;
  name: string;
  method?: string | null;
  genes?: string[];
}

/**
 * Whether the dataset already holds this gene-set score as drawn: a saved entry for the
 * set with the same method and the same genes scored. A set has one column whatever the
 * method, so re-picking a score drawn earlier (AUCell) after another method saved
 * (score_genes) must save it again, or the plot and the column disagree.
 */
export function savedGeneSetMatches(
  saved: Record<string, RecordLike>,
  name: string,
  method: string,
  genesFound: string[],
): boolean {
  const entry = Object.values(saved).find((e) => e.kind === "geneset" && e.name === name);
  if (!entry) return false;
  if ((entry.method ?? "score_genes") !== method) return false;
  const a = new Set(entry.genes ?? []);
  return a.size === new Set(genesFound).size && genesFound.every((g) => a.has(g));
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
