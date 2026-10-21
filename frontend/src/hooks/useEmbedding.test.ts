import { describe, it, expect } from "vitest";
import { embeddingDims, isPcaEmbedding, sameValues } from "./useEmbedding";

describe("sameValues", () => {
  it("matches equal coordinates, NaN included, and nothing else", () => {
    const a = new Float32Array([1, 2, NaN, 4]);
    expect(sameValues(a, new Float32Array([1, 2, NaN, 4]))).toBe(true);
    expect(sameValues(a, new Float32Array([1, 2, NaN, 5]))).toBe(false);
    expect(sameValues(a, new Float32Array([1, 2, NaN]))).toBe(false);
    expect(sameValues(null, a)).toBe(false);
  });
});

describe("embeddingDims", () => {
  it("asks for a PC pair only for a PCA, off the default, and within the embedding", () => {
    expect(embeddingDims("X_pca", [2, 3], 50)).toEqual([2, 3]);
    expect(embeddingDims("X_pca_harmony", [2, 3], 50)).toEqual([2, 3]);
    expect(embeddingDims("X_pca", [0, 1], 50)).toBeUndefined(); // the default: no param
    expect(embeddingDims("X_umap", [2, 3], 2)).toBeUndefined();
    expect(embeddingDims("X_pca", [2, 3], undefined)).toBeUndefined();
    expect(embeddingDims("X_pca", [48, 50], 50)).toBeUndefined(); // beyond this dataset's PCs
    expect(embeddingDims("X_pca", [3, 3], 50)).toBeUndefined();
    expect(isPcaEmbedding("X_PCA")).toBe(true);
    expect(isPcaEmbedding("X_tsne")).toBe(false);
  });
});
