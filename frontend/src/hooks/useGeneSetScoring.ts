import { useCallback, useState } from "react";
import { apiFetch } from "@/api/client";

/**
 * Score a gene set on the scatter (`/genesets/score`), one at a time. Shared by
 * the enrichment views so a term click behaves the same everywhere; on failure
 * the first gene is shown instead (`onFallbackGene`).
 */
export function useGeneSetScoring(
  datasetId: string | null | undefined,
  onScoreGeneSet: (scores: Float32Array, name: string) => void,
  onFallbackGene: (gene: string) => void,
): { scoreGenes: (genes: string[], name: string) => Promise<void>; scoringName: string | null } {
  const [scoringName, setScoringName] = useState<string | null>(null);

  const scoreGenes = useCallback(
    async (genes: string[], name: string) => {
      if (!datasetId || scoringName || genes.length === 0) return;
      setScoringName(name);
      try {
        const res = await apiFetch<{ scores: number[] }>(`/datasets/${datasetId}/genesets/score`, {
          method: "POST",
          body: JSON.stringify({ gene_set: genes, score_name: name }),
        });
        onScoreGeneSet(new Float32Array(res.scores), name);
      } catch {
        onFallbackGene(genes[0]!);
      } finally {
        setScoringName(null);
      }
    },
    [datasetId, scoringName, onScoreGeneSet, onFallbackGene],
  );

  return { scoreGenes, scoringName };
}
