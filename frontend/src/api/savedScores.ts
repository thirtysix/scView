import { apiFetch } from "./client";
import { announceSavedScore } from "@/lib/savedScores";

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
