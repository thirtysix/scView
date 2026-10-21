import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { MitoByType as Report, TypeMito } from "@/api/mitoByType";
import type { DatasetInfo } from "@/api/types";

const { getMitoByType, getMitoRefilter, runMitoRefilter } = vi.hoisted(() => ({
  getMitoByType: vi.fn(),
  getMitoRefilter: vi.fn(),
  runMitoRefilter: vi.fn(),
}));
vi.mock("@/api/mitoByType", () => ({ getMitoByType }));
vi.mock("@/api/mitoFilter", () => ({ getMitoRefilter, runMitoRefilter }));
vi.mock("@/api/datasets", () => ({ getDataset: vi.fn().mockResolvedValue({ id: "d1" }) }));

import { MitoByType } from "./MitoByType";
import { useDatasetStore } from "@/stores/datasetStore";

function type(group: string, extra: Partial<TypeMito> = {}): TypeMito {
  return {
    group, n_cells: 300, median: 3.0, mad: 0.7, quantiles: { q05: 1.8, q25: 2.5, q75: 3.6, q95: 4.4 },
    bound: 5.1, removed_global: 0, removed_global_frac: 0, removed_bound: 6, removed_bound_frac: 0.02,
    flagged: false, damaged: false, reasons: [], note: "", ...extra,
  };
}

const LNCAP = type("LNCaP", {
  n_cells: 400, median: 12.1, mad: 1.0, bound: 15.1, removed_global: 260, removed_global_frac: 0.65,
  removed_bound: 16, removed_bound_frac: 0.04, flagged: true,
  reasons: ["The 10.0% ceiling sits below its median (12.1%): it removes 65% of its 400 cells; its own bound, 15.1%, would remove 4%."],
});

function report(extra: Partial<Report> = {}): Report {
  return {
    column: "cell_type", reason: "cell type", measurable: true, n_cells: 720, n_labelled: 700,
    ceiling: 10, ceiling_source: "recorded", filter_mode: "fixed", already_filtered: true,
    mad_k: 3, loss_flag: 0.3, min_cells: 30, types: [LNCAP, type("CAF")], n_flagged: 1,
    removed_global_total: 260, removed_bound_total: 22,
    summary: "A 10.0% ceiling removes 65% of LNCaP (median 12.1%); its own bound (15.1%) would remove 4%.",
    note: "", rule: "Each type's own bound is its median % mito + 3 MADs.",
    citations: "Osorio & Cai 2021, Bioinformatics, doi:10.1093/bioinformatics/btaa751.", ...extra,
  };
}

beforeEach(() => {
  getMitoByType.mockReset();
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: {
      id: "d1",
      obs_columns: [
        { name: "cell_type", dtype: "category", n_unique: 2 },
        { name: "leiden", dtype: "category", n_unique: 8 },
        { name: "n_genes", dtype: "float64", n_unique: 500 },
      ],
    } as unknown as DatasetInfo,
  });
});

describe("MitoByType", () => {
  it("opens on a flagged type: summary, highlighted row, both removal counts, caption with n", async () => {
    getMitoByType.mockResolvedValue(report());
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByText("1 of 2 types cut hard by the 10.0% ceiling")).toBeInTheDocument();
    expect(screen.getByText(/A 10.0% ceiling removes 65% of LNCaP/)).toBeInTheDocument();
    const row = screen.getByTestId("mito-row-LNCaP");
    expect(row.className).toMatch(/bg-amber/);
    expect(row.textContent).toContain("260 (65%)");
    expect(row.textContent).toContain("16 (4%)");
    expect(row.textContent).toContain("15.1%");
    expect(screen.getByTestId("mito-row-CAF").className).not.toMatch(/bg-amber/);
    expect(screen.getByText(/sits below its median/)).toBeInTheDocument();
    expect(screen.getByText(/n = 700 of 720 cells; 2 types of 'cell_type'/)).toBeInTheDocument();
    expect(screen.getByText(/the dataset's last filtering/)).toBeInTheDocument();
    expect(screen.getByText("Own bound").getAttribute("title")).toMatch(/median \+ 3 MADs/);
    expect(screen.getByText("Median mito").getAttribute("title")).toMatch(/mitochondrial/);
  });

  it("says when % mito can't be measured (Kang: no MT- genes)", async () => {
    getMitoByType.mockResolvedValue(report({
      measurable: false, types: [], n_flagged: 0, ceiling: null, ceiling_source: null, summary: "",
      note: "No mitochondrial (MT-) genes in this dataset, so % mito can't be measured and no per-type mito bound can be set.",
    }));
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByText("% mito not measurable")).toBeInTheDocument();
    expect(screen.queryByLabelText("Global % mito ceiling")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Mito by cell type/ }));
    expect(screen.getByText(/can't be measured/)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("re-asks with a typed ceiling and a chosen column", async () => {
    getMitoByType.mockResolvedValue(report());
    render(<MitoByType datasetId="d1" />);
    await screen.findByText("1 of 2 types cut hard by the 10.0% ceiling");
    expect(getMitoByType).toHaveBeenLastCalledWith("d1", undefined, undefined);
    const input = screen.getByLabelText("Global % mito ceiling");
    fireEvent.change(input, { target: { value: "20" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(getMitoByType).toHaveBeenLastCalledWith("d1", undefined, 20));
    fireEvent.change(screen.getByLabelText("Label column for mito by cell type"), { target: { value: "leiden" } });
    await waitFor(() => expect(getMitoByType).toHaveBeenLastCalledWith("d1", "leiden", 20));
    // Only categorical columns are offered.
    expect(screen.queryByRole("option", { name: "n_genes" })).not.toBeInTheDocument();
  });

  it("marks a damaged type in red and stays collapsed when nothing needs a look", async () => {
    getMitoByType.mockResolvedValue(report({
      types: [type("CAF"), type("8", { median: 96.5, bound: null, removed_bound: null, removed_bound_frac: null,
        removed_global: 56, removed_global_frac: 0.98, damaged: true,
        note: "Median 96.5% mito: most of its reads are mitochondrial, likely damaged or dying cells." })],
      n_flagged: 0, summary: "A 20.0% ceiling removes at most 0% of any type. 8 (median 96.5%) looks damaged.",
    }));
    render(<MitoByType datasetId="d1" />);
    const row = await screen.findByTestId("mito-row-8");
    expect(row.className).toMatch(/bg-red/);
    expect(row.textContent).toContain("—");
    expect(screen.getByText(/likely damaged or dying cells/)).toBeInTheDocument();

    getMitoByType.mockResolvedValue(report({ types: [type("CAF")], n_flagged: 0, summary: "no type is flagged" }));
    render(<MitoByType datasetId="d2" />);
    expect(await screen.findByText("The 10.0% ceiling suits every type (1 types)")).toBeInTheDocument();
  });

  it("offers to re-filter a dataset whose filtering cut a type hard (the actual loss, from its record)", async () => {
    const lost = type("3", {
      n_cells: 300, median: 11.8, bound: 16.2, removed_global: 10, removed_global_frac: 0.03,
      removed_filter: 210, removed_filter_frac: 0.7, flagged: true,
      reasons: ["The filtering step's % mito rule removed 70% of its 300 cells (median 11.8%)."],
    });
    getMitoByType.mockResolvedValue(report({
      column: "scview_leiden_r0.5", reason: "clustering", filter_mode: "mad", ceiling: 20,
      types: [lost, type("0", { removed_filter: 4, removed_filter_frac: 0.01 })], includes_removed: 214,
      assigned_removed: 214,
      summary: "The filtering step's % mito rule removed 70% of 3 (median 11.8%).",
    }));
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByText("1 of 2 types cut hard by filtering")).toBeInTheDocument();
    expect(screen.getByText("Removed by filtering").getAttribute("title")).toMatch(/MAD mode/);
    expect(screen.getByTestId("mito-row-3").textContent).toContain("210 (70%)");
    expect(screen.getByRole("button", { name: /Re-filter with per-cluster bounds/ })).toBeInTheDocument();
    expect(screen.getByText(/Filtered with one global rule/)).toBeInTheDocument();
  });

  it("keeps saying what the re-filter did after the card stops offering it", async () => {
    getMitoByType.mockResolvedValue(report({ column: "scview_leiden_r0.5", reason: "clustering" }));
    getMitoRefilter.mockResolvedValue({
      column: "scview_leiden_r0.5", type_column: "scview_mito_type", n_original: 1222, n_current: 553, n_after: 1062,
      removed_before: 669, removed_after: 160, brought_back: 509, newly_removed: 0, removed_other: 1,
      typed: { kept: 553, column: 0, expression: 667, none: 2 }, assignment: null, types: [], rule: null,
      steps: ["filtering"], skipped_steps: [], can: "Brings back 509 of the 669 cells.", cannot: "160 stay removed.",
      notes: [], warnings: [], source_file: "pbmc.h5ad", mito_mode: "per_type", reset_to_counts: false,
    });
    runMitoRefilter.mockResolvedValue({ steps_run: ["filtering"] });
    const onRefiltered = vi.fn();
    render(<MitoByType datasetId="d1" onRefiltered={onRefiltered} />);
    fireEvent.click(await screen.findByRole("button", { name: /Re-filter with per-cluster bounds/ }));
    // after the run the panel refetches: nothing flagged, so the action is gone
    getMitoByType.mockResolvedValue(report({ column: "scview_leiden_r0.5", reason: "clustering", n_flagged: 0,
      types: [type("0")], filter_mito_mode: "per_type" }));
    fireEvent.click(await screen.findByRole("button", { name: "Re-filter (1,062 cells)" }));
    await waitFor(() => expect(onRefiltered).toHaveBeenCalled());
    expect(screen.getByTestId("mito-refiltered")).toHaveTextContent(
      "Re-filtered from the original upload: 1,062 cells kept (was 553; 509 brought back).",
    );
  });

  it("reads a dataset filtered with per-type bounds: the bound each type had, no re-filter", async () => {
    getMitoByType.mockResolvedValue(report({
      filter_mito_mode: "per_type", filter_type_column: "cell_type", n_flagged: 0,
      types: [
        type("LNCaP", { median: 12.1, bound: 15.1, applied_rule: "own", applied_bound: 15.1,
          removed_global: 260, removed_global_frac: 0.65, note: "Filtered with its own bound, 15.1%." }),
        type("CAF", { applied_rule: "own", applied_bound: 5.1 }),
      ],
      summary: "Filtered with per-type bounds on 'cell_type': 2 of 2 types used their own bound, the rest the global rule.",
    }));
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByText("Filtered with per-type bounds (2 of 2 types)")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Mito by cell type/ }));
    expect(screen.getByText("Filtered at")).toBeInTheDocument();
    expect(screen.getByTestId("mito-row-LNCaP").textContent).toContain("15.1%");
    expect(screen.getByText(/Filtered with per-type bounds on 'cell_type'\./)).toBeInTheDocument();
    expect(screen.queryByTestId("mito-refilter")).not.toBeInTheDocument();
  });

  it("does not offer to re-filter a per-type filtering whose flagged type kept the global rule", async () => {
    getMitoByType.mockResolvedValue(report({
      filter_mito_mode: "per_type", filter_type_column: "cell_type", n_flagged: 1,
      types: [
        type("rare", { n_cells: 40, median: 14, bound: null, applied_rule: "global", removed_global: 20,
          removed_global_frac: 0.5, flagged: true, reasons: ["The 10.0% ceiling sits below its median (14.0%)."] }),
        type("CAF", { applied_rule: "own", applied_bound: 5.1 }),
      ],
    }));
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByText(/1 cut hard by the global rule/)).toBeInTheDocument();
    expect(screen.queryByTestId("mito-refilter")).not.toBeInTheDocument();
  });

  it("points an unfiltered dataset to the filtering step's per-type choice", async () => {
    getMitoByType.mockResolvedValue(report({ already_filtered: false, ceiling_source: "default", filter_mode: null }));
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByTestId("mito-per-type-hint")).toHaveTextContent(/Per cell type/);
    expect(screen.queryByTestId("mito-refilter")).not.toBeInTheDocument();
    expect(screen.getByText(/Not filtered by scView yet/)).toBeInTheDocument();
  });

  it("shows an error without breaking the panel", async () => {
    getMitoByType.mockRejectedValue(new Error("API error 500: boom"));
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByText(/Mito by cell type unavailable: API error 500: boom/)).toBeInTheDocument();
  });
});
