import { beforeEach, describe, expect, it } from "vitest";
import type { DatasetInfo } from "@/api/types";
import type { CellFilter } from "@/lib/cellFilter";
import { describeSource, selectionContext, selectionSource } from "@/lib/copilotSelection";
import { useCopilotSelectionStore } from "@/stores/copilotSelectionStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useGatesStore } from "@/stores/gatesStore";
import { useSelectionStore } from "@/stores/selectionStore";

const ds = (n: number): DatasetInfo => ({
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: n,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [],
  created_at: "",
});
const filter: CellFilter = { op: "and", layer: "", items: [{ kind: "gene", gene: "ISG15", op: ">", value: 2 }] };

beforeEach(() => {
  useDatasetStore.setState({ currentDatasetId: "kang", currentDataset: ds(100) });
  useSelectionStore.setState({ selectedCellIndices: null, selectionFilter: null, selectionMode: "none", highlightedGroup: null });
  useCopilotSelectionStore.setState({ excluded: null, lastSent: null });
  useGatesStore.setState({ shown: null, records: {} });
});

describe("the selection in the co-pilot's view context", () => {
  it("is absent without a selection", () => {
    expect(selectionContext()).toBeUndefined();
    useSelectionStore.setState({ selectedCellIndices: new Set() });
    expect(selectionContext()).toBeUndefined();
  });

  it("sends a short selection as indices and a large one as a bitmask", () => {
    useSelectionStore.setState({ selectedCellIndices: new Set([7, 2, 5]), selectionMode: "lasso" });
    expect(selectionContext()).toEqual({
      n_cells: 3, source: "lasso", fresh: true, cells: { n_cells: 100, indices: [2, 5, 7] },
    });
    useDatasetStore.setState({ currentDataset: ds(10_000) });
    useSelectionStore.setState({ selectedCellIndices: new Set(Array.from({ length: 4000 }, (_, i) => i * 2)) });
    const ctx = selectionContext()!;
    expect(ctx.n_cells).toBe(4000);
    expect("bitmask" in ctx.cells).toBe(true);
    // 10,000 cells -> 1,250 bytes -> 1,668 base64 characters, not ~20 kB of indices
    expect(JSON.stringify(ctx).length).toBeLessThan(2000);
  });

  it("is fresh until a question carried it, and comes back when the cells change", () => {
    const cells = new Set([1, 2, 3]);
    useSelectionStore.setState({ selectedCellIndices: cells });
    useCopilotSelectionStore.getState().markSent(cells);
    expect(selectionContext()!.fresh).toBe(false);
    useCopilotSelectionStore.getState().exclude(cells);
    expect(selectionContext()).toBeUndefined();
    useSelectionStore.setState({ selectedCellIndices: new Set([1, 2]) });
    expect(selectionContext()).toMatchObject({ n_cells: 2, fresh: true });
  });

  it("names a shown gate, then a filter, then the lasso or a group click", () => {
    const cells = new Set([1, 2, 3]);
    useSelectionStore.setState({ selectedCellIndices: cells, selectionFilter: filter, selectionMode: "none" });
    useGatesStore.setState({
      shown: { datasetId: "kang", gateId: "g1", cells },
      records: {
        kang: {
          fingerprint: null, nCells: 100, updatedAt: "",
          gates: [{ id: "g1", name: "T cells", parentId: null, def: { kind: "filter", filter }, createdAt: "" }],
        },
      },
    });
    expect(selectionSource()).toBe("gate:T cells");
    expect(describeSource(selectionSource())).toBe("gate T cells");
    useGatesStore.setState({ shown: { datasetId: "kang", gateId: "g1", cells: new Set([9]) } });
    expect(selectionSource()).toBe("filter");
    useSelectionStore.setState({ selectionFilter: null, selectionMode: "lasso" });
    expect(selectionSource()).toBe("lasso");
    useSelectionStore.setState({ selectionMode: "none", highlightedGroup: { column: "cluster", value: "B" } });
    expect(selectionSource()).toBe("group");
  });
});
