/** An unprocessed dataset: GET /markers answers 200 with no groups and a reason (it
 *  used to 404, a failed request in the console on every open). Both marker views show
 *  the reason and log nothing. */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import { noMarkersReason, NO_MARKERS_REASON, type MarkersResponse } from "@/api/markers";
import { useDatasetStore } from "@/stores/datasetStore";

const { apiFetch } = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/components/plots/MarkerPlot", () => ({ MarkerPlot: () => null }));
vi.mock("@/components/plots/CellHeatmap", () => ({ CellHeatmap: () => null }));
vi.mock("@/components/plots/MarkerSpecificityPlot", () => ({ MarkerSpecificityPlot: () => null }));

import { UnifiedMarkersSubtab } from "./UnifiedMarkersSubtab";
import { MarkerGenesPanel } from "@/components/panels/MarkerGenesPanel";

const REASON = "No marker genes yet: run the pipeline or pick a column.";
const EMPTY: MarkersResponse = { groups: [], markers: {}, reason: REASON };
const MARKERS: MarkersResponse = {
  groups: ["B", "T"],
  markers: {
    B: [{ gene: "MS4A1", group: "B", logfoldchange: 4, pval: 1e-9, pval_adj: 1e-7 }],
    T: [{ gene: "CD3E", group: "T", logfoldchange: 3, pval: 1e-9, pval_adj: 1e-7 }],
  },
};

function dataset(columns: DatasetInfo["obs_columns"]): DatasetInfo {
  return {
    id: "raw", name: "raw", filename: "raw.h5", n_cells: 1222, n_genes: 33538, status: "ready",
    available_embeddings: [], created_at: "", obs_columns: columns,
  } as DatasetInfo;
}

let consoleError: ReturnType<typeof vi.spyOn>;
beforeEach(() => {
  apiFetch.mockReset();
  consoleError = vi.spyOn(console, "error");
});
afterEach(() => {
  expect(consoleError).not.toHaveBeenCalled();
  consoleError.mockRestore();
});

describe("noMarkersReason", () => {
  it("is the backend's reason for an empty table, null when there are markers", () => {
    expect(noMarkersReason(EMPTY)).toBe(REASON);
    expect(noMarkersReason({ groups: [], markers: {} })).toBe(NO_MARKERS_REASON);
    expect(noMarkersReason(MARKERS)).toBeNull();
  });
});

describe("Unified View markers on an unprocessed dataset", () => {
  it("shows the reason, not an error", async () => {
    useDatasetStore.setState({ currentDatasetId: "raw", currentDataset: dataset([]) });
    apiFetch.mockResolvedValue(EMPTY);
    render(<UnifiedMarkersSubtab onGeneClick={vi.fn()} groupByColumn="" setGroupByColumn={vi.fn()} />);
    expect(await screen.findByTestId("no-markers")).toHaveTextContent(REASON);
    expect(apiFetch).toHaveBeenCalledWith("/datasets/raw/markers?format=json");
  });
});

describe("Marker Genes panel on an unprocessed dataset", () => {
  it("without a column to group by: shows the reason", async () => {
    useDatasetStore.setState({ currentDatasetId: "raw", currentDataset: dataset([]) });
    apiFetch.mockResolvedValue(EMPTY);
    render(<MarkerGenesPanel />);
    expect(await screen.findByTestId("no-markers-reason")).toHaveTextContent(REASON);
    expect(screen.getByText("No marker genes available")).toBeInTheDocument();
  });

  it("with an author column: the empty default table falls through to that column's markers", async () => {
    useDatasetStore.setState({
      currentDatasetId: "raw",
      currentDataset: dataset([{ name: "cell_type", dtype: "category", n_unique: 2 }]),
    });
    apiFetch.mockImplementation((url: string) =>
      Promise.resolve(url.includes("groupby_column=cell_type") ? MARKERS : EMPTY),
    );
    render(<MarkerGenesPanel />);
    await waitFor(() => expect(screen.getByText("MS4A1")).toBeInTheDocument());
    expect(screen.queryByTestId("no-markers-reason")).toBeNull();
    expect(apiFetch).toHaveBeenCalledWith("/datasets/raw/markers?format=json&groupby_column=cell_type");
  });
});
