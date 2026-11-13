import { useEffect } from "react";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { onSavedScore } from "@/lib/savedScores";

/**
 * Re-read the dataset's columns when a score is saved or removed (lib/savedScores.ts), so
 * Compare, Colour by and the histogram list it at once. Mounted once, in the app shell.
 */
export function useSavedScoreRefresh(): void {
  const refreshDataset = useDatasetRefresh();
  useEffect(
    () =>
      onSavedScore((datasetId, column) => {
        void refreshDataset(datasetId, column ?? undefined).catch((e) =>
          console.error("Could not refresh the dataset after a saved score:", e),
        );
      }),
    [refreshDataset],
  );
}
