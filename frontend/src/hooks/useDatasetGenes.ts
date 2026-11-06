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
 * the same list the gene search searches), fetched once per dataset and layer, to
 * tell which genes of a list it has. `layer` ("" or absent: the default layer) as
 * `GET /expression` takes it. `genes` is null while loading or without a dataset;
 * `failed` when the list could not be read (an unknown layer: 400).
 */
export function useDatasetGenesState(
  datasetId: string | null | undefined,
  layer = "",
): { genes: DatasetGenes | null; failed: boolean } {
  const { data, isError } = useQuery({
    queryKey: ["dataset-genes", datasetId, layer],
    queryFn: () =>
      apiFetch<{ genes: string[] }>(
        `/datasets/${datasetId}/genes${layer ? `?layer=${encodeURIComponent(layer)}` : ""}`,
      ),
    enabled: !!datasetId,
    staleTime: 10 * 60 * 1000,
  });
  const genes = useMemo(() => {
    if (!data) return null;
    const list = data.genes ?? [];
    return { list, set: new Set(list), byUpper: genesByUpper(list) };
  }, [data]);
  return { genes, failed: isError };
}

/** The dataset's genes (default layer), or null while loading or without a dataset. */
export function useDatasetGenes(datasetId: string | null | undefined): DatasetGenes | null {
  return useDatasetGenesState(datasetId).genes;
}
