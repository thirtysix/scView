import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { decodeArrowBuffer } from "@/lib/arrow";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * Per-cell category codes of one obs column, from the embedding buffer that
 * carries it: the same query the scatter, the split grid and the cluster map
 * use, so a column already on screen is not fetched again.
 */
export function useObsCodes(
  datasetId: string | null,
  embedding: string,
  column: string | null | undefined,
): { codes: Int32Array | null; error: boolean } {
  const pipelineRunning = useSettingsStore((s) => s.pipelineRunning);
  const { data: buffer, error } = useQuery({
    queryKey: ["embedding", datasetId, embedding, column],
    queryFn: () => fetchEmbeddingBinary(datasetId!, embedding, column!),
    enabled: !!datasetId && !!embedding && !!column && !pipelineRunning,
    staleTime: 5 * 60 * 1000,
    gcTime: 60 * 1000,
  });
  const codes = useMemo(() => {
    if (!buffer || !column) return null;
    const v = decodeArrowBuffer(buffer)[column];
    return v ? Int32Array.from(v as ArrayLike<number>) : null;
  }, [buffer, column]);
  return { codes, error: !!error };
}
