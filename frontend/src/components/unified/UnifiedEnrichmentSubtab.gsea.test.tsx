import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import type { GseaResponse } from "@/api/enrichment";
import { useDatasetStore } from "@/stores/datasetStore";

const { apiFetch, runGsea } = vi.hoisted(() => ({ apiFetch: vi.fn(), runGsea: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/enrichment", () => ({ runGsea }));
vi.mock("@/components/panels/MSigDBCollectionTree", () => ({
  MSigDBCollectionTree: () => null,
  DEFAULT_MSIGDB_COLLECTIONS: ["h.all"],
}));

import { UnifiedEnrichmentSubtab } from "./UnifiedEnrichmentSubtab";

const DATASET: DatasetInfo = {
  id: "kang", name: "Kang", filename: "kang.h5ad", n_cells: 100, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], created_at: "",
  obs_columns: [{ name: "cluster", dtype: "category", n_unique: 2, values: ["CD14 Mono", "B"] }],
};

const GSEA: GseaResponse = {
  rank_metric: "marker_score",
  rank_metric_label: "marker score (Wilcoxon z-score vs the rest)",
  n_genes_ranked: 13890, n_excluded_ribo_mito: 163, n_sets_tested: 50, permutations: 1000,
  min_size: 15, max_size: 500, collections: ["h.all"], exclude_ribo_mito: true,
  up: { n_terms: 1, n_significant: 1, results: [{
    term: "HALLMARK_COMPLEMENT", es: 0.6, nes: 1.8, pvalue: 0, fdr: 0, set_size: 165,
    leading_edge: ["ANXA5", "LGALS3"], collection: "h.all",
  }] },
  down: { n_terms: 1, n_significant: 1, results: [{
    term: "HALLMARK_MYC_TARGETS_V1", es: -0.5, nes: -2.05, pvalue: 0, fdr: 0, set_size: 190,
    leading_edge: ["NPM1"], collection: "h.all",
  }] },
  seconds: 1.57, cached: false, column: "cluster", group: "CD14 Mono",
};

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "kang" });
  apiFetch.mockReset();
  apiFetch.mockResolvedValue({ groups: ["CD14 Mono", "B"] });
  runGsea.mockReset();
  runGsea.mockResolvedValue(GSEA);
});

const renderTab = () =>
  render(
    <UnifiedEnrichmentSubtab
      onTermClick={vi.fn()}
      onScoreGeneSet={vi.fn()}
      groupByColumn="cluster"
      setGroupByColumn={vi.fn()}
    />,
  );

describe("UnifiedEnrichmentSubtab GSEA", () => {
  it("runs GSEA on the selected marker group and shows both directions", async () => {
    renderTab();
    const button = screen.getByRole("button", { name: "GSEA" });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    await waitFor(() => expect(runGsea).toHaveBeenCalledTimes(1));
    expect(runGsea).toHaveBeenCalledWith("kang", {
      column: "cluster", group: "CD14 Mono", collections: ["h.all"], exclude_ribo_mito: true,
    });
    expect(await screen.findByText("HALLMARK_COMPLEMENT")).toBeInTheDocument();
    expect(screen.getByText(/ranked by marker score/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Down in CD14 Mono · 1 set/ }));
    expect(screen.getByText("HALLMARK_MYC_TARGETS_V1")).toBeInTheDocument();
  });

  it("clears the GSEA result when another group is picked", async () => {
    renderTab();
    const button = screen.getByRole("button", { name: "GSEA" });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);
    await screen.findByText("HALLMARK_COMPLEMENT");
    fireEvent.change(screen.getByDisplayValue("CD14 Mono"), { target: { value: "B" } });
    expect(screen.queryByText("HALLMARK_COMPLEMENT")).not.toBeInTheDocument();
  });
});
