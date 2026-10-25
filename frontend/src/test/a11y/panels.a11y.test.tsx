/**
 * axe-core over every Unified View tab and the other panels, in their empty or
 * loading state (requests never answer here): no serious or critical violations.
 */
import { describe, it, expect, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { seriousViolations } from "@/test/axe";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

vi.mock("@/hooks/useEmbedding", async () => ({
  ...(await vi.importActual<object>("@/hooks/useEmbedding")),
  useEmbedding: () => ({
    positions: Float32Array.from([0, 0, 1, 1, 2, 2]), colors: Int32Array.from([0, 1, 1]), numCells: 3,
    colorColumnName: "cell_type", dimensions: 2, isLoading: false, error: null,
  }),
}));
vi.mock("@/components/plots/EmbeddingScatter", () => ({ EmbeddingScatter: () => <div /> }));
vi.mock("@/components/plots/Plot", () => ({ default: () => null, Plot: () => null }));
vi.mock("@/api/client", async () => ({
  ...(await vi.importActual<object>("@/api/client")),
  apiFetch: vi.fn(() => new Promise(() => {})),
  apiFetchBinary: vi.fn(() => new Promise(() => {})),
}));
globalThis.fetch = vi.fn(() => new Promise(() => {})) as never;

import { UnifiedViewPanel } from "@/components/panels/UnifiedViewPanel";
import { ObservationsPanel } from "@/components/panels/ObservationsPanel";
import { GeneSetPanel } from "@/components/panels/GeneSetPanel";
import { MarkerGenesPanel } from "@/components/panels/MarkerGenesPanel";
import { TrajectoryPanel } from "@/components/panels/TrajectoryPanel";
import { ProvenancePanel } from "@/components/panels/ProvenancePanel";
import { DataPanel } from "@/components/panels/DataPanel";
import { GeneExpressionPanel } from "@/components/panels/GeneExpressionPanel";

const DATASET: DatasetInfo = {
  id: "kang", name: "Kang", filename: "kang.h5ad", n_cells: 3, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap", "X_pca"],
  obs_columns: [
    { name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "NK"] },
    { name: "n_genes", dtype: "int64", n_unique: 3 },
  ],
  created_at: "",
};
const check = async (name: string, root: Element) => {
  expect({ name, violations: await seriousViolations(root) }).toEqual({ name, violations: [] });
};
Element.prototype.scrollTo ??= () => {};
globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;
const settle = () => act(() => new Promise((r) => setTimeout(r, 30)));

describe("panels without a serious accessibility violation", () => {
  it("every Unified View tab and the other panels", async () => {
    useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "kang" });
    useSettingsStore.setState({ colorBy: "cell_type", embedding: "X_umap" });
    const u = render(<QueryClientProvider client={new QueryClient()}><UnifiedViewPanel /></QueryClientProvider>);
    await settle();
    for (const t of screen.getAllByRole("tab").map((x) => x.textContent!)) {
      fireEvent.click(screen.getByRole("tab", { name: t }));
      await settle();
      await check("unified tab " + t, u.container);
    }
    u.unmount();
    for (const [n, P] of [["obs", ObservationsPanel], ["genesets", GeneSetPanel], ["markers", MarkerGenesPanel], ["trajectory", TrajectoryPanel], ["provenance", ProvenancePanel], ["data", DataPanel], ["expression", GeneExpressionPanel]] as const) {
      const r = render(<QueryClientProvider client={new QueryClient()}><P /></QueryClientProvider>);
      await settle();
      await check(n, r.container);
      r.unmount();
    }
  }, 60000);
});
