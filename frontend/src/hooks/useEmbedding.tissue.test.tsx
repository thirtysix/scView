import { describe, it, expect, vi, beforeEach } from "vitest";
import type { ReactNode } from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

// The decoder worker answers with x = [1, 2], y = [10, 20], whatever was asked.
const { fetchEmbeddingBinary } = vi.hoisted(() => ({ fetchEmbeddingBinary: vi.fn() }));
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary }));
vi.mock("@/workers/arrowDecoder.worker?worker", () => ({
  default: class {
    private listeners: ((e: MessageEvent) => void)[] = [];
    addEventListener(_: string, fn: (e: MessageEvent) => void) {
      this.listeners.push(fn);
    }
    removeEventListener(_: string, fn: (e: MessageEvent) => void) {
      this.listeners = this.listeners.filter((l) => l !== fn);
    }
    postMessage() {
      const col = (v: number[]) => ({ data: Float32Array.from(v).buffer, type: "float32", length: v.length });
      const data = { type: "success", numRows: 2, result: { x: col([1, 2]), y: col([10, 20]) } };
      queueMicrotask(() => this.listeners.forEach((l) => l({ data } as MessageEvent)));
    }
    terminate() {}
  },
}));

import { useEmbedding } from "./useEmbedding";

const DATASET = {
  id: "vis",
  available_embeddings: ["X_umap"],
  obs_columns: [],
  spatial: { technology: "visium", unit: "spot", library_column: "library_id", libraries: [] },
} as unknown as DatasetInfo;

const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "vis" });
  useSettingsStore.setState({ colorBy: "", pipelineRunning: false });
  fetchEmbeddingBinary.mockReset();
  fetchEmbeddingBinary.mockResolvedValue(new ArrayBuffer(8));
});

describe("useEmbedding on the tissue", () => {
  it("asks for obsm spatial by name and turns y (down) into the plot's y (up)", async () => {
    useSettingsStore.setState({ embedding: "spatial" });
    const { result } = renderHook(() => useEmbedding(), { wrapper });
    await waitFor(() => expect(result.current.positions).not.toBeNull());
    expect(fetchEmbeddingBinary).toHaveBeenCalledWith("vis", "spatial", undefined, undefined);
    expect(Array.from(result.current.positions!)).toEqual([1, -10, 2, -20]);
    expect(result.current.embedding).toBe("spatial"); // what the positions are of
  });

  it("keeps a UMAP's y as it is", async () => {
    useSettingsStore.setState({ embedding: "X_umap" });
    const { result } = renderHook(() => useEmbedding(), { wrapper });
    await waitFor(() => expect(result.current.positions).not.toBeNull());
    expect(Array.from(result.current.positions!)).toEqual([1, 10, 2, 20]);
  });
});
