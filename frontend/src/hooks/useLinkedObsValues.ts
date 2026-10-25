import { useContext, useEffect, useState } from "react";
import { QueryClientContext } from "@tanstack/react-query";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { decodeArrowBuffer } from "@/lib/arrow";

/**
 * Per-cell values of a numeric obs column (a QC metric), for a panel that
 * shows the selection on data it otherwise gets binned or sampled from the
 * server (C035). Fetched only while `column` is set (the caller passes null
 * without a selection); through the app's query cache when there is one (the
 * same key as useObsValues), so a column already loaded is not fetched again.
 * Works without a QueryClientProvider (panels rendered on their own).
 */
export function useLinkedObsValues(
  datasetId: string | null,
  embedding: string,
  column: string | null,
): Float32Array | null {
  const client = useContext(QueryClientContext);
  const [state, setState] = useState<{ key: string; values: Float32Array | null } | null>(null);
  const key = `${datasetId}\u0000${embedding}\u0000${column}`;
  useEffect(() => {
    if (!datasetId || !embedding || !column) return;
    let cancelled = false;
    const fetchIt = () => fetchEmbeddingBinary(datasetId, embedding, column);
    const buffer = client
      ? client.fetchQuery({ queryKey: ["embedding", datasetId, embedding, column], queryFn: fetchIt, staleTime: 5 * 60 * 1000 })
      : fetchIt();
    buffer
      .then((b) => {
        const v = decodeArrowBuffer(b)[column];
        const values = v ? (v instanceof Float32Array ? v : Float32Array.from(v as ArrayLike<number>)) : null;
        if (!cancelled) setState({ key, values });
      })
      .catch(() => { if (!cancelled) setState({ key, values: null }); });
    return () => { cancelled = true; };
  }, [client, datasetId, embedding, column, key]);
  return column && state?.key === key ? state.values : null;
}
