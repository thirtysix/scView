import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";

/**
 * True when `dataset` is `datasetId` and lists `embedding`. The stored
 * embedding name (default X_umap) outlives the dataset it was picked on, so
 * without this check a dataset without it is asked for it once before
 * UnifiedViewPanel switches to one it has (a 404 on every open).
 */
export function embeddingReady(
  dataset: DatasetInfo | null,
  datasetId: string | null | undefined,
  embedding: string,
): boolean {
  return (
    !!datasetId &&
    !!embedding &&
    dataset?.id === datasetId &&
    dataset.available_embeddings.includes(embedding)
  );
}

export function useEmbeddingReady(datasetId: string | null | undefined, embedding: string): boolean {
  const dataset = useDatasetStore((s) => s.currentDataset);
  return embeddingReady(dataset, datasetId, embedding);
}
