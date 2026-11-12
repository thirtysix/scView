import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo, ObsColumnInfo } from "@/api/types";
import { oneGroupReason } from "@/api/markers";
import { defaultGroupBy } from "@/lib/obsColumns";
import { resetViewForDataset, useDatasetStore } from "@/stores/datasetStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const { apiFetch } = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch }));

import { UnifiedMarkersSubtab } from "./UnifiedMarkersSubtab";

// A one-library Xenium import before clustering: library_id has one value.
const LIBRARY: ObsColumnInfo = { name: "library_id", dtype: "category", n_unique: 1, values: ["lib1"] };
const CLUSTERS: ObsColumnInfo = { name: "leiden", dtype: "category", n_unique: 3, values: ["0", "1", "2"] };

const dataset = (obs_columns: ObsColumnInfo[], active_clustering?: string): DatasetInfo => ({
  id: "xen", name: "Xenium", filename: "xen", n_cells: 100, n_genes: 10, status: "ready",
  available_embeddings: ["spatial"], created_at: "", obs_columns, active_clustering,
});

beforeEach(() => {
  apiFetch.mockReset();
  apiFetch.mockResolvedValue({ groups: ["0", "1"], markers: { "0": [], "1": [] } });
});

describe("defaultGroupBy", () => {
  it("skips a column with one group", () => {
    expect(defaultGroupBy([LIBRARY, CLUSTERS])).toBe("leiden");
    expect(defaultGroupBy([LIBRARY, { ...CLUSTERS, name: "region" }])).toBe("region");
  });

  it("is empty when no column has two groups", () => {
    expect(defaultGroupBy([LIBRARY])).toBe("");
    expect(defaultGroupBy([])).toBe("");
  });

  it("keeps the active clustering first, then clustering names", () => {
    const sample = { ...CLUSTERS, name: "sample" };
    expect(defaultGroupBy([sample, CLUSTERS], "leiden")).toBe("leiden");
    expect(defaultGroupBy([sample, CLUSTERS])).toBe("leiden");
    expect(defaultGroupBy([sample, LIBRARY], "library_id")).toBe("sample");
  });
});

describe("oneGroupReason", () => {
  it("names a one-group column and passes others", () => {
    expect(oneGroupReason(LIBRARY)).toMatch(/^library_id has only one group/);
    expect(oneGroupReason(CLUSTERS)).toBeNull();
    expect(oneGroupReason(undefined)).toBeNull();
  });
});

describe("a group-by kept across datasets", () => {
  it("is dropped when the new dataset has one group in it", () => {
    useUnifiedViewStore.getState().setGroupByColumn("library_id");
    resetViewForDataset(dataset([LIBRARY, CLUSTERS]));
    expect(useUnifiedViewStore.getState().groupByColumn).toBe("");
    useUnifiedViewStore.getState().setGroupByColumn("leiden");
    resetViewForDataset(dataset([LIBRARY, CLUSTERS]));
    expect(useUnifiedViewStore.getState().groupByColumn).toBe("leiden");
  });
});

describe("UnifiedMarkersSubtab on a one-group column", () => {
  it("does not ask for markers and says why, with a picker to move on", () => {
    useDatasetStore.setState({ currentDataset: dataset([LIBRARY, CLUSTERS]), currentDatasetId: "xen" });
    const setGroupByColumn = vi.fn();
    render(<UnifiedMarkersSubtab onGeneClick={vi.fn()} groupByColumn="library_id" setGroupByColumn={setGroupByColumn} />);
    expect(apiFetch).not.toHaveBeenCalled();
    expect(screen.getByTestId("one-group")).toHaveTextContent("library_id has only one group");
    fireEvent.change(screen.getByLabelText("Group markers by"), { target: { value: "leiden" } });
    expect(setGroupByColumn).toHaveBeenCalledWith("leiden");
  });

  it("asks for a column with two groups", async () => {
    useDatasetStore.setState({ currentDataset: dataset([LIBRARY, CLUSTERS]), currentDatasetId: "xen" });
    render(<UnifiedMarkersSubtab onGeneClick={vi.fn()} groupByColumn="leiden" setGroupByColumn={vi.fn()} />);
    await waitFor(() => expect(apiFetch).toHaveBeenCalledWith(expect.stringContaining("groupby_column=leiden")));
    expect(screen.queryByTestId("one-group")).toBeNull();
  });
});
