import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { decodeArrowBuffer } from "@/lib/arrow";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * Per-cell values of one numeric obs column (total_counts, % mito, a score),
 * from the embedding buffer that carries it: the query useObsCodes and the
 * scatter use, so a column already coloured by is not fetched again.
 */
export function useObsValues(
  datasetId: string | null,
  embedding: string,
  column: string | null | undefined,
): { values: Float32Array | null; isLoading: boolean; error: boolean } {
  const pipelineRunning = useSettingsStore((s) => s.pipelineRunning);
  const { data: buffer, error, isLoading } = useQuery({
    queryKey: ["embedding", datasetId, embedding, column],
    queryFn: () => fetchEmbeddingBinary(datasetId!, embedding, column!),
    enabled: !!datasetId && !!embedding && !!column && !pipelineRunning,
    staleTime: 5 * 60 * 1000,
    gcTime: 60 * 1000,
  });
  const values = useMemo(() => {
    if (!buffer || !column) return null;
    const v = decodeArrowBuffer(buffer)[column];
    if (!v) return null;
    return v instanceof Float32Array ? v : Float32Array.from(v as ArrayLike<number>);
  }, [buffer, column]);
  return { values, isLoading: !!column && isLoading, error: !!error };
}
