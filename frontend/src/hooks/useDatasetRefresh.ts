import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getDataset } from "@/api/datasets";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useObsRevisionStore } from "@/stores/obsRevisionStore";

/**
 * After the backend changed an obs column: reload the dataset's info (so colour-by
 * and group-by pickers list the column straight away), refetch the scatter's
 * colours and mark the column changed for views holding its values.
 */
export function useDatasetRefresh(): (datasetId: string, column?: string) => Promise<DatasetInfo> {
  const queryClient = useQueryClient();
  return useCallback(
    async (datasetId: string, column?: string) => {
      const fresh = await getDataset(datasetId);
      // Only if it is still the dataset on screen: the user may have switched.
      if (useDatasetStore.getState().currentDatasetId === datasetId) {
        useDatasetStore.getState().setCurrentDataset(fresh);
      }
      if (column) useObsRevisionStore.getState().bump(datasetId, column);
      await queryClient.invalidateQueries({ queryKey: ["embedding", datasetId] });
      return fresh;
    },
    [queryClient],
  );
}
