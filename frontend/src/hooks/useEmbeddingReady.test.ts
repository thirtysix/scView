import { describe, it, expect } from "vitest";
import type { DatasetInfo } from "@/api/types";
import { embeddingReady } from "./useEmbeddingReady";

const ds = { id: "a", available_embeddings: ["X_pca", "X_tsne"] } as unknown as DatasetInfo;

describe("embeddingReady", () => {
  it("waits for the open dataset to list the embedding", () => {
    expect(embeddingReady(ds, "a", "X_tsne")).toBe(true);
    // Persisted default the dataset does not have: no request.
    expect(embeddingReady(ds, "a", "X_umap")).toBe(false);
    // Dataset id switched before its info arrived.
    expect(embeddingReady(ds, "b", "X_pca")).toBe(false);
    expect(embeddingReady(null, "a", "X_pca")).toBe(false);
    expect(embeddingReady(ds, "a", "")).toBe(false);
  });
});
