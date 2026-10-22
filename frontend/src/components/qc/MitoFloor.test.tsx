/** The per-type % mito floor ("never stricter than the global rule") in the filter
 *  control's table, the re-filter and the Mito by cell type card. Numbers follow the
 *  backend's hand-worked fixture (tests/test_qc_mito_floor.py): L (low baseline, own
 *  2.5 %) loses 4 per type and none with the floor; H (high, own 18 %) loses 30 to the
 *  13 % global rule and 4 per type either way. */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { FilterThresholds } from "@/api/qc";
import type { MitoTypesTable, RefilterPreview } from "@/api/mitoFilter";
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

import { MitoModeControl, recordedMitoParams } from "./MitoModeControl";
import { MitoRefilter } from "./MitoRefilter";
import { MitoByType } from "./MitoByType";
import { useDatasetStore } from "@/stores/datasetStore";

const T: FilterThresholds = {
  min_genes: 0, max_genes: 0, min_counts: 0, max_pct_mt: 13, drop_doublets: false,
  qc_threshold_mode: "fixed", qc_mad_k_mt: 3, qc_mito_mode: "per_type", qc_mito_type_column: "",
};

function table(floor: boolean): MitoTypesTable {
  return {
    column: "cell_type", reason: "cell type", k: 3, cap: 50, min_cells: 30, damaged_median: 50, ceiling_pct_mt: 13,
    types: [
      { name: "L", n_cells: 100, median: 1, mad: 0.5, bound: 2.5, rule: "own", why: null, capped: false, note: "",
        warning: null, removed_global: 0, removed: floor ? 0 : 4, removed_type: 4, removed_floor: 0,
        floor_uses: "global", n_floored: 100, global_bound: [13, 13] },
      { name: "H", n_cells: 100, median: 12, mad: 2, bound: 18, rule: "own", why: null, capped: false, note: "",
        warning: null, removed_global: 30, removed: 4, removed_type: 4, removed_floor: 4,
        floor_uses: "type", n_floored: 0, global_bound: [13, 13] },
      { name: "M", n_cells: 80, median: 6, mad: 1, bound: 9, rule: "own", why: null, capped: false, note: "",
        warning: null, removed_global: 2, removed: floor ? 1 : 3, removed_type: 3, removed_floor: 1,
        floor_uses: "mixed", n_floored: 40, global_bound: [4.5, 13] },
      { name: "Q", n_cells: 60, median: 8, mad: 0, bound: null, rule: "global", why: "no_spread", capped: false,
        note: "No spread", warning: null, removed_global: 0, removed: 0, removed_type: 0, removed_floor: 0,
        floor_uses: null, n_floored: 0, global_bound: [13, 13] },
    ],
    n_own: 3, n_global: 1, n_unlabelled: 0, reused_from: null, warnings: [], applied: "per_type", floor,
  };
}

beforeEach(() => {
  getMitoByType.mockReset();
  getMitoRefilter.mockReset();
  runMitoRefilter.mockReset();
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: {
      id: "d1",
      obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 4 }],
    } as unknown as DatasetInfo,
  });
});

describe("the floor in the filtering step's % mito control", () => {
  it("is a checkbox under per type, off by default, that sets qc_mito_type_floor", () => {
    const onChange = vi.fn();
    render(<MitoModeControl datasetId="d1" values={T} onChange={onChange} hasMt table={table(false)} />);
    const box = screen.getByRole("checkbox", { name: "Never stricter than the global rule" });
    expect(box).not.toBeChecked();
    fireEvent.click(box);
    expect(onChange).toHaveBeenCalledWith({ qc_mito_type_floor: true });
    expect(screen.getByLabelText("About never stricter than the global rule")).toBeInTheDocument();
  });

  it("is not offered with one global rule", () => {
    render(<MitoModeControl datasetId="d1" values={{ ...T, qc_mito_mode: "global" }} onChange={vi.fn()} hasMt />);
    expect(screen.queryByRole("checkbox", { name: "Never stricter than the global rule" })).toBeNull();
  });

  it("floor off: the table shows the loss with the floor off and on, own bounds used", () => {
    render(<MitoModeControl datasetId="d1" values={T} onChange={vi.fn()} hasMt table={table(false)} />);
    expect(screen.getByText("Never stricter").getAttribute("title")).toMatch(/never more than the global rule/);
    const l = screen.getByTestId("mito-type-L");
    const cells = Array.from(l.querySelectorAll("td")).map((td) => td.textContent);
    expect(cells).toEqual(["L", "100", "1.0%", "2.5%", "0", "4", "0"]);
    expect(screen.getByText("Per type").className).toMatch(/font-semibold/); // applied
    expect(screen.getByText("Never stricter").className).not.toMatch(/font-semibold/);
    expect(
      screen.getByText(/Removed for % mito: 32 by the global rule, 11 per type, 5 per type never stricter than the global rule \(not applied\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/A bound works both ways/)).toBeInTheDocument();
  });

  it("floor on: says which bound each type used (own / global / both) and the floor's loss", () => {
    render(
      <MitoModeControl datasetId="d1" values={{ ...T, qc_mito_type_floor: true }} onChange={vi.fn()} hasMt table={table(true)} />,
    );
    expect(screen.getByRole("checkbox", { name: "Never stricter than the global rule" })).toBeChecked();
    const bound = (name: string) => screen.getByTestId(`mito-type-${name}`).querySelectorAll("td")[3]!;
    expect(bound("L").textContent).toBe("global");
    expect(bound("L").getAttribute("title")).toMatch(/Its own bound, 2\.5%, is below the global rule's \(13\.0%\)/);
    expect(bound("H").textContent).toBe("18.0%");
    expect(bound("M").textContent).toBe("9.0% / global");
    expect(bound("M").getAttribute("title")).toMatch(/for 40 cells in samples whose bound is above it/);
    expect(bound("Q").textContent).toBe("global (no spread)");
    expect(screen.getByText("Never stricter").className).toMatch(/font-semibold/);
    expect(screen.getByText(/5 per type never stricter than the global rule \(applied\)/)).toBeInTheDocument();
    expect(screen.getByText(/a type's bound is the higher of its own and the global rule's \(the ceiling\)/)).toBeInTheDocument();
  });

  it("in MAD mode the caption says the global bound is each cell's own sample's", () => {
    render(
      <MitoModeControl
        datasetId="d1"
        values={{ ...T, qc_threshold_mode: "mad", qc_mito_type_floor: true }}
        onChange={vi.fn()}
        hasMt
        table={table(true)}
      />,
    );
    expect(screen.getAllByText(/the bound of the cell's own sample, under the ceiling/).length).toBeGreaterThan(0);
  });

  it("a dataset filtered with the floor keeps it for the next run; old records mean off", () => {
    expect(recordedMitoParams({ qc_mito_mode: "per_type", qc_mito_type_column: "ct", qc_mito_type_floor: true })).toEqual({
      qc_mito_mode: "per_type", qc_mito_type_column: "ct", qc_mito_type_floor: true,
    });
    expect(recordedMitoParams({ qc_mito_mode: "per_type", qc_mito_type_column: "ct" })).toEqual({
      qc_mito_mode: "per_type", qc_mito_type_column: "ct",
    });
  });
});

function refilterPreview(floor: boolean): RefilterPreview {
  return {
    column: "cell_type", type_column: "cell_type", mito_mode: "per_type", mito_type_floor: floor,
    source_file: "x.h5ad", reset_to_counts: false, n_original: 300, n_current: 266, n_after: floor ? 296 : 292,
    removed_before: 34, removed_after: floor ? 4 : 8, brought_back: 30, newly_removed: floor ? 0 : 4, removed_other: 0,
    typed: { kept: 266, column: 34, expression: 0, none: 0 }, assignment: null,
    types: [
      { name: "L", n_all: 100, median: 1, bound: 2.5, rule: "own", floor_uses: "global", n_floored: 100,
        removed_before: 0, removed_after: floor ? 0 : 4, brought_back: 0, newly_removed: floor ? 0 : 4 },
      { name: "H", n_all: 100, median: 12, bound: 18, rule: "own", floor_uses: "type", n_floored: 0,
        removed_before: 34, removed_after: 4, brought_back: 30, newly_removed: 0 },
    ],
    rule: null, steps: ["filtering"], skipped_steps: [], can: "Brings back 30.", cannot: "4 stay removed.",
    notes: [], warnings: [],
  };
}

describe("the floor in the re-filter", () => {
  it("asks again with the floor when it is ticked, shows the bound used and runs with it", async () => {
    getMitoRefilter.mockImplementation((_id: string, _col: string, floor: boolean) => Promise.resolve(refilterPreview(floor)));
    runMitoRefilter.mockResolvedValue({ steps_run: ["filtering"] });
    render(<MitoRefilter datasetId="d1" column="cell_type" perCluster={false} />);
    const box = () => screen.getByRole("checkbox", { name: "Never stricter than the global rule" });
    expect(box()).not.toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: /Re-filter with per-type bounds/ }));
    await screen.findByTestId("refilter-preview");
    expect(getMitoRefilter).toHaveBeenLastCalledWith("d1", "cell_type", false);
    expect(screen.getByTestId("refilter-row-L").querySelectorAll("td")[3]!.textContent).toBe("2.5%");
    fireEvent.click(box());
    await waitFor(() => expect(getMitoRefilter).toHaveBeenLastCalledWith("d1", "cell_type", true));
    await waitFor(() => expect(screen.getByTestId("refilter-row-L").querySelectorAll("td")[3]!.textContent).toBe("global"));
    expect(screen.getByText(/never stricter than the global rule \(that same processing's\)/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Re-filter (296 cells)" }));
    await waitFor(() => expect(runMitoRefilter).toHaveBeenCalled());
    expect(runMitoRefilter.mock.calls[0]![3]).toBe(true);
  });
});

function cardType(group: string, extra: Partial<TypeMito>): TypeMito {
  return {
    group, n_cells: 100, median: 1, mad: 0.5, quantiles: {}, bound: 2.5, removed_global: 0, removed_global_frac: 0,
    removed_bound: 4, removed_bound_frac: 0.04, flagged: false, damaged: false, reasons: [], note: "", ...extra,
  };
}

function card(extra: Partial<Report> = {}): Report {
  return {
    column: "cell_type", reason: "cell type", measurable: true, n_cells: 200, n_labelled: 200, ceiling: 13,
    ceiling_source: "recorded", filter_mode: "fixed", already_filtered: true, mad_k: 3, loss_flag: 0.3, min_cells: 30,
    types: [
      cardType("L", { floor_bound: 13, removed_floor: 0, removed_floor_frac: 0, applied_rule: "own", applied_bound: 2.5,
                      applied_floor: "global" }),
      cardType("H", { median: 12, bound: 18, removed_global: 30, removed_global_frac: 0.3, floor_bound: 18,
                      removed_floor: 4, removed_floor_frac: 0.04, applied_rule: "own", applied_bound: 18,
                      applied_floor: "type" }),
    ],
    n_flagged: 0, removed_global_total: 30, removed_bound_total: 8, removed_floor_total: 4,
    filter_mito_mode: "per_type", filter_type_column: "cell_type", filter_type_floor: true,
    summary: "Filtered with per-type bounds on 'cell_type', never stricter than the global rule: 1 of 2 types used their own bound.",
    note: "", rule: "rule", citations: "", ...extra,
  };
}

describe("the floor on the Mito by cell type card", () => {
  it("shows what each type's own bound removes when never stricter, and which bound filtering used", async () => {
    getMitoByType.mockResolvedValue(card());
    render(<MitoByType datasetId="d1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Mito by cell type/ }));
    const head = screen.getByText("Own, never stricter");
    expect(head.getAttribute("title")).toMatch(/Never more than the ceiling removes/);
    const l = Array.from(screen.getByTestId("mito-row-L").querySelectorAll("td")).map((td) => td.textContent);
    expect(l.slice(5)).toEqual(["0 (0%)", "4 (4%)", "0 (0%)", "global"]);
    const h = Array.from(screen.getByTestId("mito-row-H").querySelectorAll("td")).map((td) => td.textContent);
    expect(h.slice(5)).toEqual(["30 (30%)", "4 (4%)", "4 (4%)", "18.0%"]);
    expect(screen.getByText(/Filtered with per-type bounds on 'cell_type', never stricter than the global rule\./)).toBeInTheDocument();
  });

  it("an older backend without the floor's numbers shows no floor column", async () => {
    const plain = card({ filter_type_floor: undefined, removed_floor_total: undefined });
    plain.types = plain.types.map((t) => {
      const c = { ...t };
      delete c.removed_floor;
      delete c.removed_floor_frac;
      delete c.floor_bound;
      delete c.applied_floor;
      return c;
    });
    getMitoByType.mockResolvedValue(plain);
    render(<MitoByType datasetId="d1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Mito by cell type/ }));
    expect(screen.queryByText("Own, never stricter")).toBeNull();
    const l = Array.from(screen.getByTestId("mito-row-L").querySelectorAll("td")).map((td) => td.textContent);
    expect(l[l.length - 1]).toBe("2.5%");
  });
});
