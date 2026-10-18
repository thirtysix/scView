import { useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchGeneExpression } from "@/api/expression";

/** A gene the layer does not have (cached, so it is not asked for again). */
const ABSENT = null;

export interface GeneExpression {
  /** Each found gene's per-cell values, in the order asked; null while loading */
  values: Record<string, Float32Array> | null;
  /** Genes asked for that the layer does not have */
  missing: string[];
  isLoading: boolean;
  error: string | null;
}

const geneKey = (datasetId: string, layer: string, gene: string) => ["gene-expression", datasetId, layer, gene];

/**
 * Per-cell expression of several genes for the feature grid, the stacked
 * violin and the gene-vs-gene view. Each gene is cached on its own, and the
 * genes not cached yet come in ONE request (`GET /expression` takes a list),
 * so adding a gene to a set of eight fetches one column, not nine.
 */
export function useGeneExpression(
  datasetId: string | null,
  genes: string[],
  layer: string,
  enabled = true,
): GeneExpression {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["gene-expression-set", datasetId, layer, ...genes],
    enabled: enabled && !!datasetId && genes.length > 0,
    staleTime: 5 * 60 * 1000,
    gcTime: 60 * 1000,
    queryFn: async () => {
      const ds = datasetId!;
      const cached = (g: string) =>
        queryClient.getQueryData<Float32Array | typeof ABSENT>(geneKey(ds, layer, g));
      const wanted = genes.filter((g) => cached(g) === undefined);
      if (wanted.length > 0) {
        let fetched: Record<string, Float32Array> = {};
        try {
          fetched = await fetchGeneExpression(ds, wanted, layer || undefined);
        } catch (err) {
          // 404: none of them is in the layer; anything else is a real failure.
          if (!(err instanceof Error && err.message.startsWith("API error 404"))) throw err;
        }
        for (const g of wanted) queryClient.setQueryData(geneKey(ds, layer, g), fetched[g] ?? ABSENT);
      }
      const values: Record<string, Float32Array> = {};
      const missing: string[] = [];
      for (const g of genes) {
        const v = cached(g);
        if (v) values[g] = v;
        else missing.push(g);
      }
      return { values, missing };
    },
  });
  return {
    values: query.data?.values ?? null,
    missing: query.data?.missing ?? [],
    isLoading: query.isLoading && query.fetchStatus !== "idle",
    error: query.error ? String(query.error.message ?? query.error) : null,
  };
}
