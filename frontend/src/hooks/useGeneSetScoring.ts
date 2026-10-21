import { useCallback, useState } from "react";
import type { ActivityRef } from "@/api/activity";
import { scoreGeneSet, type ScoreMethod } from "@/api/genesetScore";
import { useScoringMethodStore } from "@/stores/scoringMethodStore";

/** What a score is handed to (UnifiedViewPanel's handleScoreComplete): the scores, the
 *  set's name and genes, and the method; ``activity`` is for TF / pathway activity only. */
export type OnScoreGeneSet = (
  scores: Float32Array,
  name: string,
  genes?: string[],
  activity?: ActivityRef,
  method?: ScoreMethod,
) => void;

/**
 * Score a gene set on the scatter (`/genesets/score`), one at a time, with the method
 * chosen in ScoringMethodChoice. Shared by the enrichment views so a term click
 * behaves the same everywhere; on failure the first gene is shown instead
 * (`onFallbackGene`).
 */
export function useGeneSetScoring(
  datasetId: string | null | undefined,
  onScoreGeneSet: OnScoreGeneSet,
  onFallbackGene: (gene: string) => void,
): { scoreGenes: (genes: string[], name: string) => Promise<void>; scoringName: string | null } {
  const [scoringName, setScoringName] = useState<string | null>(null);
  const method = useScoringMethodStore((s) => s.method);

  const scoreGenes = useCallback(
    async (genes: string[], name: string) => {
      if (!datasetId || scoringName || genes.length === 0) return;
      setScoringName(name);
      try {
        const res = await scoreGeneSet(datasetId, genes, name, method);
        onScoreGeneSet(new Float32Array(res.scores), name, genes, undefined, method);
      } catch {
        onFallbackGene(genes[0]!);
      } finally {
        setScoringName(null);
      }
    },
    [datasetId, scoringName, onScoreGeneSet, onFallbackGene, method],
  );

  return { scoreGenes, scoringName };
}
