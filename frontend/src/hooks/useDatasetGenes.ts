import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import { genesByUpper } from "@/lib/geneLists";

export interface DatasetGenes {
  /** Every gene the dataset's expression has, in the dataset's order */
  list: string[];
  set: ReadonlySet<string>;
  byUpper: ReadonlyMap<string, string>;
}

/**
 * The names the dataset's expression is looked up by (`GET /datasets/{id}/genes`,
 * the same list the gene search searches), fetched once per dataset, to tell
 * which genes of a list it has. Null while loading or without a dataset.
 */
export function useDatasetGenes(datasetId: string | null | undefined): DatasetGenes | null {
  const { data } = useQuery({
    queryKey: ["dataset-genes", datasetId],
    queryFn: () => apiFetch<{ genes: string[] }>(`/datasets/${datasetId}/genes`),
    enabled: !!datasetId,
    staleTime: 10 * 60 * 1000,
  });
  return useMemo(() => {
    if (!data) return null;
    const list = data.genes ?? [];
    return { list, set: new Set(list), byUpper: genesByUpper(list) };
  }, [data]);
}
