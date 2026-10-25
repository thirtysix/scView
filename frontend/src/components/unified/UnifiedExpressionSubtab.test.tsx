import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useGeneListsStore } from "@/stores/geneListsStore";

vi.mock("@/components/unified/MultiGenePanel", () => ({ MultiGenePanel: () => null }));
vi.mock("@/components/unified/CoexpressedGenes", () => ({ CoexpressedGenes: () => null }));

import { UnifiedExpressionSubtab } from "./UnifiedExpressionSubtab";

const dataset = {
  id: "ds1", name: "kang", filename: "kang.h5ad", n_cells: 6, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], obs_columns: [], created_at: "",
} as DatasetInfo;

const renderTab = (activeGene: string | null) =>
  render(
    <UnifiedExpressionSubtab
      onGeneSelect={vi.fn()}
      activeGene={activeGene}
      groupByColumn=""
      setGroupByColumn={vi.fn()}
      expressionScale={null}
    />,
  );

beforeEach(() => {
  localStorage.clear();
  useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
  useGeneListsStore.setState({ lists: [], selectedId: null });
});

describe("UnifiedExpressionSubtab", () => {
  it("adds the searched gene to a gene list", () => {
    useGeneListsStore.getState().create("Mine", ["CD3E"]);
    renderTab("MS4A1");
    fireEvent.click(screen.getByRole("button", { name: "Add to gene list" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /Mine/ }));
    expect(useGeneListsStore.getState().lists[0]!.genes).toEqual(["CD3E", "MS4A1"]);
  });

  it("offers nothing to add before a gene is searched", () => {
    renderTab(null);
    expect(screen.queryByRole("button", { name: "Add to gene list" })).toBeNull();
    expect(screen.queryByRole("checkbox", { name: /Density/ })).toBeNull();
  });

  it("offers the density toggle next to the gene's colour", () => {
    renderTab("MS4A1");
    expect(screen.getByRole("checkbox", { name: /Density/ })).toBeTruthy();
  });
});
