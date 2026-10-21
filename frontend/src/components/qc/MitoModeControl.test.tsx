import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { FilterThresholds } from "@/api/qc";
import type { MitoTypesTable } from "@/api/mitoFilter";
import type { DatasetInfo } from "@/api/types";

const { getMitoByType } = vi.hoisted(() => ({ getMitoByType: vi.fn() }));
vi.mock("@/api/mitoByType", () => ({ getMitoByType }));

import { MitoModeControl, MITO_NOT_MEASURABLE, recordedMitoParams } from "./MitoModeControl";
import { useDatasetStore } from "@/stores/datasetStore";

const T: FilterThresholds = {
  min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false,
  qc_threshold_mode: "mad", qc_mad_k_mt: 3, qc_mito_mode: "global", qc_mito_type_column: "",
};

const TABLE: MitoTypesTable = {
  column: "cell_type", reason: "cell type", k: 3, cap: 50, min_cells: 30, damaged_median: 50, ceiling_pct_mt: 10,
  types: [
    { name: "A", n_cells: 100, median: 3, mad: 1, bound: 6, rule: "own", why: null, capped: false, note: "",
      warning: null, removed_global: 0, removed: 4 },
    { name: "B", n_cells: 100, median: 12, mad: 2, bound: 18, rule: "own", why: null, capped: false, note: "",
      warning: null, removed_global: 100, removed: 4 },
    { name: "C", n_cells: 20, median: 15, mad: 0, bound: null, rule: "global", why: "too_few",
      capped: false, note: "Only 20 cells: too few for a bound of its own, so it keeps the global rule.",
      warning: null, removed_global: 20, removed: 20 },
    { name: "H", n_cells: 60, median: 40, mad: 4, bound: 50, rule: "own", why: null, capped: true,
      note: "Median + 3 MADs is 52.0%: capped at 50%.", warning: "H: its own bound, 50.0%, keeps cells ...",
      removed_global: 60, removed: 1 },
  ],
  n_own: 3, n_global: 1, n_unlabelled: 10, reused_from: null, warnings: ["H: its own bound, 50.0%, keeps cells ..."],
  applied: "per_type",
};

beforeEach(() => {
  getMitoByType.mockReset();
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: {
      id: "d1",
      obs_columns: [
        { name: "cell_type", dtype: "category", n_unique: 4 },
        { name: "scview_leiden_r0.5", dtype: "category", n_unique: 9 },
        { name: "n_genes", dtype: "float64", n_unique: 500 },
      ],
    } as unknown as DatasetInfo,
  });
});

describe("MitoModeControl", () => {
  it("switches between one rule and per cell type", () => {
    const onChange = vi.fn();
    render(<MitoModeControl datasetId="d1" values={T} onChange={onChange} hasMt />);
    expect(screen.getByRole("button", { name: "One rule" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByLabelText("Cell types for per-type % mito bounds")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Per cell type" }));
    expect(onChange).toHaveBeenCalledWith({ qc_mito_mode: "per_type" });
    expect(getMitoByType).not.toHaveBeenCalled(); // the preview said whether mito is measurable
  });

  it("per type: picks the column and shows each type's rule and loss under both rules", () => {
    const onChange = vi.fn();
    render(<MitoModeControl datasetId="d1" values={{ ...T, qc_mito_mode: "per_type" }} onChange={onChange} hasMt table={TABLE} />);
    const select = screen.getByLabelText("Cell types for per-type % mito bounds");
    expect(screen.getByRole("option", { name: "Auto: cell_type" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "n_genes" })).not.toBeInTheDocument(); // categorical only
    fireEvent.change(select, { target: { value: "cell_type" } });
    expect(onChange).toHaveBeenCalledWith({ qc_mito_type_column: "cell_type" });
    const b = screen.getByTestId("mito-type-B");
    expect(b.textContent).toContain("12.0%");
    expect(b.textContent).toContain("18.0%");
    expect(b.textContent).toContain("100");
    expect(screen.getByTestId("mito-type-C").textContent).toContain("global (too few cells)");
    expect(screen.getByTestId("mito-type-H").textContent).toContain("50.0% (cap)");
    expect(screen.getByTestId("mito-type-A").lastElementChild?.className).toMatch(/amber/); // cut more per type
    expect(screen.getByText("H: its own bound, 50.0%, keeps cells ...")).toBeInTheDocument();
    expect(screen.getByText(/Removed for % mito: 180 by the global rule, 29 per type/)).toBeInTheDocument();
    expect(screen.getByText(/n = 280 cells; 4 types of 'cell_type'/)).toBeInTheDocument();
    expect(screen.getByText("Bound used").getAttribute("title")).toMatch(/median \+ 3 MADs/);
  });

  it("is disabled with the reason when % mito can't be measured, and never says 0%", () => {
    render(<MitoModeControl datasetId="d1" values={{ ...T, qc_mito_mode: "per_type" }} onChange={vi.fn()} hasMt={false} table={null} />);
    expect(screen.getByRole("button", { name: "Per cell type" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "One rule" })).toBeDisabled();
    expect(screen.getByText(MITO_NOT_MEASURABLE)).toBeInTheDocument();
    expect(screen.queryByLabelText("Cell types for per-type % mito bounds")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/\b0(\.0)?%/);
  });

  it("asks the backend whether mito is measurable when no preview says so", async () => {
    getMitoByType.mockResolvedValue({ measurable: false, column: "cell_type" });
    render(<MitoModeControl datasetId="d1" values={T} onChange={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Per cell type" })).toBeDisabled());
    expect(getMitoByType).toHaveBeenCalledWith("d1");
    expect(screen.getByText(MITO_NOT_MEASURABLE)).toBeInTheDocument();
  });

  it("warns that Reprocess from Scratch drops a scView clustering chosen as the types", () => {
    render(
      <MitoModeControl
        datasetId="d1"
        values={{ ...T, qc_mito_mode: "per_type", qc_mito_type_column: "scview_leiden_r0.5" }}
        onChange={vi.fn()}
        hasMt
      />,
    );
    expect(screen.getByText(/Reprocess from Scratch removes 'scview_leiden_r0.5'/)).toBeInTheDocument();
  });

  it("keeps a per-type filtering's settings for the next run", () => {
    expect(recordedMitoParams({ qc_mito_mode: "per_type", qc_mito_type_column: "scview_mito_type" })).toEqual({
      qc_mito_mode: "per_type", qc_mito_type_column: "scview_mito_type",
    });
    expect(recordedMitoParams({ qc_mito_mode: "global" })).toEqual({});
    expect(recordedMitoParams({})).toEqual({});
  });
});
