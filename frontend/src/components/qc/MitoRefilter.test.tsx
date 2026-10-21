import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { RefilterPreview } from "@/api/mitoFilter";

const { getMitoRefilter, runMitoRefilter, getDataset } = vi.hoisted(() => ({
  getMitoRefilter: vi.fn(),
  runMitoRefilter: vi.fn(),
  getDataset: vi.fn(),
}));
vi.mock("@/api/mitoFilter", () => ({ getMitoRefilter, runMitoRefilter }));
vi.mock("@/api/datasets", () => ({ getDataset }));

import { MitoRefilter, typedText } from "./MitoRefilter";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

function preview(extra: Partial<RefilterPreview> = {}): RefilterPreview {
  return {
    column: "scview_leiden_r0.5", type_column: "scview_mito_type", mito_mode: "per_type", source_file: "pbmc.h5ad",
    reset_to_counts: false, n_original: 1222, n_current: 1050, n_after: 1150, removed_before: 172, removed_after: 72,
    brought_back: 110, newly_removed: 10, removed_other: 30,
    typed: { kept: 1050, column: 0, expression: 172, none: 0 },
    assignment: { method: "nearest_centroid", metric: "pearson", genes: 1980, n: 172, n_assigned: 172, median_r: 0.71, n_low: 3 },
    types: [
      { name: "0", n_all: 400, median: 4.1, bound: 7.0, rule: "own", removed_before: 20, removed_after: 25, brought_back: 2, newly_removed: 7 },
      { name: "3", n_all: 300, median: 11.8, bound: 16.2, rule: "own", removed_before: 150, removed_after: 40, brought_back: 108, newly_removed: 3 },
      { name: "8", n_all: 57, median: 96.5, bound: null, rule: "global", note: "Median 96.5% mito: damaged", removed_before: 57, removed_after: 57, brought_back: 0, newly_removed: 0 },
    ],
    rule: null,
    steps: ["qc_metrics", "filtering", "normalization", "clustering"],
    skipped_steps: [],
    can: "Brings back 110 of the 172 cells the last filtering removed: those within their type's own % mito bound.",
    cannot: "62 stay removed: 30 by the counts, genes or doublet rules (applied again unchanged) and 32 for % mito. Labels, gates and columns made in scView since are not carried over.",
    notes: [],
    warnings: [],
    ...extra,
  };
}

beforeEach(() => {
  getMitoRefilter.mockReset();
  runMitoRefilter.mockReset();
  getDataset.mockReset().mockResolvedValue({ id: "d1", n_cells: 1150 });
  useSettingsStore.setState({ pipelineRunning: false });
});

describe("MitoRefilter", () => {
  it("previews per type what changes, how removed cells were typed and what it can't bring back", async () => {
    getMitoRefilter.mockResolvedValue(preview());
    render(<MitoRefilter datasetId="d1" column="scview_leiden_r0.5" perCluster />);
    fireEvent.click(screen.getByRole("button", { name: /Re-filter with per-cluster bounds/ }));
    await screen.findByTestId("refilter-preview");
    expect(getMitoRefilter).toHaveBeenCalledWith("d1", "scview_leiden_r0.5");
    expect(screen.getByText(/1,150 of 1,222 cells kept/)).toBeInTheDocument();
    const three = screen.getByTestId("refilter-row-3");
    expect(three.textContent).toContain("11.8%");
    expect(three.textContent).toContain("16.2%");
    expect(three.textContent).toContain("108");
    expect(screen.getByTestId("refilter-row-8").textContent).toContain("global");
    expect(screen.getByText(/172 by expression/)).toBeInTheDocument();
    expect(screen.getByText(/Brings back 110 of the 172/)).toBeInTheDocument();
    expect(screen.getByText(/62 stay removed/)).toBeInTheDocument();
    expect(screen.getByText(/re-runs 4 steps: QC metrics, filtering, normalization, clustering/)).toBeInTheDocument();
    expect(screen.getByText("Back").getAttribute("title")).toMatch(/Removed now, kept after/);
  });

  it("runs, shows progress, refreshes the dataset and tells the panel", async () => {
    getMitoRefilter.mockResolvedValue(preview());
    let release: (v: Record<string, unknown>) => void = () => {};
    runMitoRefilter.mockImplementation((_id: string, _col: string, onEvent: (e: unknown) => void) => {
      onEvent({ event: "refilter", data: { stage: "load", message: "Reading the original upload" } });
      onEvent({ event: "step_start", data: { step: "filtering", index: 1, total: 4 } });
      return new Promise((r) => (release = r));
    });
    const onDone = vi.fn();
    render(<MitoRefilter datasetId="d1" column="scview_leiden_r0.5" perCluster onDone={onDone} />);
    fireEvent.click(screen.getByRole("button", { name: /Re-filter with per-cluster bounds/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Re-filter (1,150 cells)" }));
    expect(await screen.findByText("Step 2/4: filtering")).toBeInTheDocument();
    expect(useSettingsStore.getState().pipelineRunning).toBe(true);
    release({ steps_run: ["filtering"] });
    expect(await screen.findByText("Re-filtered: 1,150 cells kept (was 1,050).")).toBeInTheDocument();
    expect(runMitoRefilter.mock.calls[0]![0]).toBe("d1");
    expect(runMitoRefilter.mock.calls[0]![1]).toBe("scview_leiden_r0.5");
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(getDataset).toHaveBeenCalledWith("d1");
    expect(useDatasetStore.getState().currentDataset).toEqual({ id: "d1", n_cells: 1150 });
    expect(useSettingsStore.getState().pipelineRunning).toBe(false);
  });

  it("shows a refusal or a failed run without losing the panel", async () => {
    getMitoRefilter.mockRejectedValueOnce(new Error("No original upload is stored for this dataset."));
    render(<MitoRefilter datasetId="d1" column="cell_type" perCluster={false} />);
    fireEvent.click(screen.getByRole("button", { name: /Re-filter with per-type bounds/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No original upload is stored");
    getMitoRefilter.mockResolvedValue(preview());
    runMitoRefilter.mockRejectedValue(new Error("The re-filter stopped before it finished; nothing was saved."));
    fireEvent.click(screen.getByRole("button", { name: /Re-filter with per-type bounds/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Re-filter (1,150 cells)" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("nothing was saved");
  });

  it("says how the removed cells were typed", () => {
    expect(typedText(preview())).toBe(
      "The 172 cells the last filtering removed were typed: 172 by expression (the 'scview_leiden_r0.5' group " +
        "whose mean profile is most correlated with the cell's own, over 1,980 variable genes without MT- genes; " +
        "median r 0.71; 3 with r under 0.2).",
    );
    expect(
      typedText(preview({ column: "Cell type", typed: { kept: 1050, column: 172, expression: 0, none: 4 }, assignment: null })),
    ).toBe(
      "The 172 cells the last filtering removed were typed: 172 from the original upload's 'Cell type'. " +
        "4 cells have no type and keep the global rule.",
    );
  });
});
