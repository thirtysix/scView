import { apiFetch } from "./client";
import { announceSavedScore } from "@/lib/savedScores";

/** One saved score's record entry (backend core/saved_scores.py), as GET /scores lists it. */
export interface SavedScoreRecord {
  kind: "geneset" | "activity";
  name: string;
  method?: string | null;
  /** A gene set: the genes scored (found in the dataset) */
  genes?: string[];
  network?: string | null;
}

/** The dataset's saved scores: column -> what it holds. */
export async function listSavedScores(datasetId: string): Promise<Record<string, SavedScoreRecord>> {
  const res = await apiFetch<{ scores: Record<string, SavedScoreRecord> }>(`/datasets/${datasetId}/scores`);
  return res.scores ?? {};
}

/** Remove a saved per-cell score (backend api/v1/saved_scores.py): its column and its record.
 *  Only a column the app saved as a score; any other answers 404. */
export async function removeSavedScore(
  datasetId: string,
  column: string,
): Promise<{ removed: string; kind: string; name: string }> {
  const res = await apiFetch<{ removed: string; kind: string; name: string }>(
    `/datasets/${datasetId}/scores/${encodeURIComponent(column)}`,
    { method: "DELETE" },
  );
  announceSavedScore(datasetId, column);
  return res;
}
