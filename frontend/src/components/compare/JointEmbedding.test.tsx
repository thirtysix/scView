import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import type { IntegrationEvent, IntegrationPlan } from "@/api/integration";
import type { PreflightVerdict } from "@/api/preflight";

const { listDatasets, planIntegration, runIntegration, cancelPipelineRun } = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  planIntegration: vi.fn(),
  runIntegration: vi.fn(),
  cancelPipelineRun: vi.fn(),
}));
vi.mock("@/api/datasets", () => ({ listDatasets, getDataset: vi.fn() }));
vi.mock("@/api/integration", () => ({
  planIntegration,
  runIntegration,
  getJointSummary: vi.fn().mockResolvedValue({ dataset_id: "", joint: false }),
}));
vi.mock("@/api/pipelineRun", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/pipelineRun")>()),
  cancelPipelineRun,
}));

import {
  CombineLabels,
  DroppedColumns,
  JointEmbedding,
  UnitFilteringNote,
  combineSpec,
  defaultJointName,
  missingText,
  prefixPatch,
  unitFilteringText,
} from "./JointEmbedding";
import type { UnitFiltering } from "@/api/integration";

const ds = (id: string): DatasetInfo => ({
  id,
  name: id,
  filename: `${id}.h5ad`,
  n_cells: 4992,
  n_genes: 36601,
  status: "ready",
  available_embeddings: [],
  obs_columns: [],
  created_at: "",
});

const VERDICT: PreflightVerdict = {
  level: "tight",
  sentence: "About 5.1 GB at its peak (variable genes), roughly 2 to 6 min. scView has 6.2 GB free of its 10 GB limit.",
  need_bytes: 5.1 * 1024 ** 3,
  peak_bytes: 5.1 * 1024 ** 3,
  peak_stage: "variable genes",
  available_bytes: 6.2 * 1024 ** 3,
  limit_bytes: 10 * 1024 ** 3,
  used_bytes: 3.8 * 1024 ** 3,
  limited: true,
  source: "container limit",
  seconds: [120, 360],
  cached: { count: 0, bytes: 0, frees_enough: false },
  suggestions: [],
  error_band: 0.2,
  breakdown: [],
  notes: [],
};

const PLAN: IntegrationPlan = {
  datasets: ["s1", "s2"].map((id) => ({
    id, name: id, label: id, n_cells: 4279, n_genes: 36601, counts_source: "X", columns: [], error: null,
  })),
  merge: { intersection: 35477, union: 36601, suspicious_low_overlap: false, reconciliation: null, est_genes: 35477, warnings: [] },
  shared_columns: [],
  est_cells: 8558,
  blocking: [],
  preflight: VERDICT,
  ok: true,
};

describe("JointEmbedding run: memory estimate, watchdog, Cancel", () => {
  it("shows the plan's memory estimate, the run's memory warning, and cancels the new dataset's run", async () => {
    listDatasets.mockResolvedValue([ds("s1"), ds("s2")]);
    planIntegration.mockResolvedValue(PLAN);
    cancelPipelineRun.mockResolvedValue({ cancelling: true, step: "pca" });
    let emit!: (e: IntegrationEvent) => void;
    runIntegration.mockImplementation((_body, onEvent: (e: IntegrationEvent) => void) => {
      emit = onEvent;
      return new Promise(() => {});
    });
    render(<JointEmbedding />);
    for (const box of await screen.findAllByRole("checkbox")) fireEvent.click(box);
    expect(await screen.findByText(VERDICT.sentence)).toBeInTheDocument();
    expect(screen.getByText("Memory:")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Build joint embedding/ }));
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull(); // no id yet
    emit({ event: "stage", data: { stage: "merge", message: "Merging 2 datasets", dataset_id: "j1" } });
    emit({ event: "step_start", data: { step: "pca", index: 5, total: 12 } });
    emit({
      event: "watchdog",
      data: { type: "watchdog", kind: "memory", threshold_pct: 80, used_gb: 8.1, limit_gb: 10, pct: 81, limited: true, step: "pca" },
    });
    expect(await screen.findByText(/Memory at 81% of the 10(\.0)? GB limit/)).toBeInTheDocument();
    const cancel = await screen.findByRole("button", { name: "Cancel" });
    expect(cancel.getAttribute("title")).toMatch(/joint dataset is removed/);
    fireEvent.click(cancel);
    expect(cancelPipelineRun).toHaveBeenCalledWith("j1");
    expect(within(cancel).getByText("Cancelling...")).toBeInTheDocument();
    expect(screen.getByText(/Cancelling: PCA finishes first/)).toBeInTheDocument();
  });
});

describe("JointEmbedding combined label column", () => {
  const SUGGESTED: IntegrationPlan = {
    ...PLAN,
    label_columns: { s1: ["Celltype", "orig.ident"], s2: ["cell_line", "orig.ident"] },
    label_suggestion: { name: "Celltype", columns: { s1: "Celltype", s2: "cell_line" }, shared_values: [] },
    combined: [],
  };

  it("sets up the suggested combination, re-checks it with the plan and sends it with the run", async () => {
    listDatasets.mockResolvedValue([ds("s1"), ds("s2")]);
    planIntegration.mockImplementation(async (_ids: string[], _labels: string[], combine?: unknown[]) =>
      combine
        ? {
            ...SUGGESTED,
            combined: [{ name: "Celltype", columns: { s1: "Celltype", s2: "cell_line" }, labels: { s1: "s1", s2: "s2" },
              prefix: false, n_values: 8, shared_values: ["parental"], n_cells: { s1: 1919, s2: 968 },
              n_missing: { s1: 0, s2: 150 } }],
          }
        : SUGGESTED,
    );
    runIntegration.mockReturnValue(new Promise(() => {}));
    render(<JointEmbedding />);
    const picker = await screen.findByTestId("joint-picker");
    for (const box of within(picker).getAllByRole("checkbox")) fireEvent.click(box);
    const box = await screen.findByTestId("combine-labels", {}, { timeout: 3000 });
    expect(box.textContent).toContain("s1: Celltype, s2: cell_line");
    expect((within(box).getByLabelText("Combined column name") as HTMLInputElement).value).toBe("Celltype");
    expect((within(box).getByLabelText("Column of s2") as HTMLSelectElement).value).toBe("cell_line");
    const summary = await screen.findByTestId("combined-summary", {}, { timeout: 3000 });
    expect(summary.textContent).toContain("'Celltype' names the cells of 2 datasets with 8 values");
    expect(summary.textContent).toContain("1 value is in more than one dataset (parental)");
    const missing = within(box).getAllByTestId("combine-missing");
    expect(missing.map((m) => m.textContent)).toEqual(["all 1,919 cells have a value", "150 of 968 cells have no value"]);
    expect(missing[1]!.className).toContain("text-amber-700");
    const asked = planIntegration.mock.calls.at(-1)!;
    expect(asked[2]).toEqual([{ name: "Celltype", columns: { s1: "Celltype", s2: "cell_line" }, prefix: false }]);

    fireEvent.click(screen.getByRole("button", { name: /Build joint embedding/ }));
    expect(runIntegration.mock.calls.at(-1)![0].combine).toEqual(asked[2]);
  });
});

describe("prefixPatch: the dataset label and a name equal to a combined column", () => {
  const d = { on: true, name: "Celltype", columns: { a: "Celltype", b: "cell_line" }, prefix: false };
  it("renames the suggested name, keeps a typed one (the plan refuses it then)", () => {
    expect(prefixPatch(d, "Celltype", true)).toEqual({ prefix: true, name: "Celltype_combined" });
    expect(prefixPatch({ ...d, name: "cell_line" }, "Celltype", true)).toEqual({ prefix: true }); // typed
    expect(prefixPatch({ ...d, name: "line" }, "Celltype", true)).toEqual({ prefix: true }); // no clash
    expect(prefixPatch(d, "Celltype", false)).toEqual({ prefix: false });
    expect(prefixPatch(d, null, true)).toEqual({ prefix: true }); // no suggestion
  });

  it("in the box: ticking the label renames the default name, not a typed one", () => {
    const plan = {
      ...PLAN,
      label_columns: { s1: ["Celltype"], s2: ["cell_line"] },
      label_suggestion: { name: "Celltype", columns: { s1: "Celltype", s2: "cell_line" }, shared_values: [] },
      combined: [],
    } as IntegrationPlan;
    const onChange = vi.fn();
    const draft = { on: true, name: "Celltype", columns: { s1: "Celltype", s2: "cell_line" }, prefix: false };
    const { rerender } = render(<CombineLabels plan={plan} draft={draft} onChange={onChange} />);
    fireEvent.click(screen.getByRole("checkbox", { name: /Put the dataset label before each value/ }));
    expect(onChange).toHaveBeenLastCalledWith({ ...draft, prefix: true, name: "Celltype_combined" });
    rerender(<CombineLabels plan={plan} draft={{ ...draft, name: "cell_line" }} onChange={onChange} />);
    fireEvent.click(screen.getByRole("checkbox", { name: /Put the dataset label before each value/ }));
    expect(onChange).toHaveBeenLastCalledWith({ ...draft, name: "cell_line", prefix: true });
  });
});

describe("missingText", () => {
  const checked = [{ columns: { a: "Celltype", b: "Cell_line" }, n_cells: { a: 10, b: 968 }, n_missing: { a: 0, b: 150 } }];
  it("says per dataset how many cells the chosen column leaves without a value", () => {
    expect(missingText(checked, "b", "Cell_line")).toEqual({ text: "150 of 968 cells have no value", warn: true });
    expect(missingText(checked, "a", "Celltype")).toEqual({ text: "all 10 cells have a value", warn: false });
    expect(missingText(checked, "b", "cell_line")).toBeNull(); // another column: not checked yet
    expect(missingText(checked, "b", "")).toBeNull();
    expect(missingText([{ columns: { a: "x" } }], "a", "x")).toBeNull(); // an older backend: no counts
  });
});

describe("combineSpec", () => {
  it("sends a combination only when it is on, named and has two datasets' columns", () => {
    const d = { on: true, name: " line ", columns: { a: "Celltype", b: "", c: "cell_line" }, prefix: true };
    expect(combineSpec(d, ["a", "b", "c"])).toEqual({ name: "line", columns: { a: "Celltype", c: "cell_line" }, prefix: true });
    expect(combineSpec({ ...d, on: false }, ["a", "b", "c"])).toBeNull();
    expect(combineSpec({ ...d, name: "" }, ["a", "c"])).toBeNull();
    expect(combineSpec(d, ["a", "b"])).toBeNull(); // c is no longer selected
    expect(combineSpec(null, ["a"])).toBeNull();
  });
});

describe("JointEmbedding dataset picker", () => {
  it("says it is loading, not that the library is empty, until the list arrives", async () => {
    let resolve!: (ds: DatasetInfo[]) => void;
    listDatasets.mockReturnValue(new Promise<DatasetInfo[]>((r) => (resolve = r)));
    render(<JointEmbedding />);
    expect(screen.getByText(/Loading datasets/)).toBeInTheDocument();
    expect(screen.queryByText(/No ready datasets/)).toBeNull();

    resolve([]);
    expect(await screen.findByText(/No ready datasets/)).toBeInTheDocument();
    expect(screen.queryByText(/Loading datasets/)).toBeNull();
  });

  it("stays usable with 28 datasets: it scrolls, filters, selects what is shown, and counts", async () => {
    const many = Array.from({ length: 30 }, (_, i) => ds(i < 28 ? `slide_${i}` : `other_${i}`));
    listDatasets.mockResolvedValue(many);
    planIntegration.mockResolvedValue({ ...PLAN, datasets: [] });
    render(<JointEmbedding />);
    const picker = await screen.findByTestId("joint-picker");
    expect(picker.className).toContain("overflow-y-auto");
    expect(within(picker).getAllByRole("checkbox")).toHaveLength(30);
    fireEvent.change(screen.getByLabelText("Filter datasets"), { target: { value: "SLIDE" } });
    expect(within(picker).getAllByRole("checkbox")).toHaveLength(28);
    fireEvent.click(screen.getByText("Select shown"));
    expect(screen.getByTestId("joint-picked")).toHaveTextContent("28 of 30 selected");
    expect(within(picker).getAllByRole("checkbox").every((b) => (b as HTMLInputElement).checked)).toBe(true);
    fireEvent.change(screen.getByLabelText("Filter datasets"), { target: { value: "other" } });
    expect(within(picker).getAllByRole("checkbox")).toHaveLength(30); // selected ones stay in view
    fireEvent.click(screen.getByText("Clear"));
    expect(screen.getByTestId("joint-picked")).toHaveTextContent("0 of 30 selected");
  });
});

describe("defaultJointName", () => {
  it("names a few datasets in full and many by the first three and a count", () => {
    expect(defaultJointName(["a"])).toBe("");
    expect(defaultJointName(["a", "b"])).toBe("joint: a + b");
    expect(defaultJointName(Array.from({ length: 28 }, (_, i) => `s${i}`))).toBe("joint: s0 + s1 + s2 + 25 more");
  });
});

describe("DroppedColumns: what the joint dataset leaves out", () => {
  const some = { column: "Fibroblast", reason: "a number in only some of the datasets", datasets: ["ctrl_P1"] };
  const qc = { column: "total_counts", reason: "recomputed by the joint run", datasets: ["ctrl_P1", "IZ_P3"] };

  it("lists a short list open, grouped by why, naming the datasets that have a partial column", () => {
    render(<DroppedColumns columns={[some, qc]} nDatasets={2} />);
    expect(screen.getByText("2 annotation columns are not carried into the joint dataset:")).toBeInTheDocument();
    expect(screen.getByText("A number in only some of the datasets:")).toBeInTheDocument();
    expect(screen.getByText("Recomputed by the joint run:")).toBeInTheDocument();
    expect(screen.getByText("(ctrl_P1)", { exact: false })).toBeInTheDocument();
    expect(screen.getByTestId("dropped-columns").tagName).toBe("DIV");
  });

  it("folds a long list under one line", () => {
    const many = Array.from({ length: 9 }, (_, i) => ({ ...qc, column: `qc${i}` }));
    render(<DroppedColumns columns={many} nDatasets={2} />);
    const box = screen.getByTestId("dropped-columns");
    expect(box.tagName).toBe("DETAILS");
    expect(within(box).getByText(/9 annotation columns are not carried .* \(show which and why\)/)).toBeInTheDocument();
  });
});

describe("the joint run's filtering of datasets scView never filtered", () => {
  const rows: UnitFiltering[] = [
    { label: "P16", n_cells: 4400, filtered_before: false, n_removed: 708, by_criterion: { min_genes: 708, min_counts: 300 },
      mode: "mad", genes_floor: 100, note: null },
    { label: "P1", n_cells: 4000, filtered_before: true, n_removed: 0, by_criterion: {}, mode: null, genes_floor: null,
      note: "filtered by scView before: left as it is" },
    { label: "P7", n_cells: 3000, filtered_before: false, n_removed: 0, by_criterion: {}, mode: "mad", genes_floor: 100, note: null },
  ];

  it("says per dataset what it removes, and that filtered ones are left alone", () => {
    expect(unitFilteringText(rows[0]!)).toBe("P16: removes 708 of 4,400 cells (16.1%), mostly Min genes");
    expect(unitFilteringText(rows[1]!)).toBe("P1: filtered by scView before, left as it is");
    expect(unitFilteringText(rows[2]!)).toBe("P7: removes no cells");
  });

  it("is on by default and can be switched off", () => {
    const onChange = vi.fn();
    const { rerender } = render(<UnitFilteringNote rows={rows} on onChange={onChange} />);
    const box = screen.getByRole("checkbox", { name: "Filter the datasets scView has not filtered" });
    expect(box).toBeChecked();
    expect(screen.getByTestId("unit-filtering")).toHaveTextContent("at least 100 genes per cell");
    expect(screen.getByTestId("unit-filtering")).toHaveTextContent("708 cells are removed in all.");
    fireEvent.click(box);
    expect(onChange).toHaveBeenCalledWith(false);
    rerender(<UnitFilteringNote rows={rows} on={false} onChange={onChange} />);
    expect(screen.getByTestId("unit-filtering")).toHaveTextContent("Off: every cell of every dataset joins.");
  });

  it("cannot be switched when every dataset was filtered before", () => {
    render(<UnitFilteringNote rows={[rows[1]!]} on onChange={vi.fn()} />);
    expect(screen.getByRole("checkbox")).toBeDisabled();
    expect(screen.getByTestId("unit-filtering")).toHaveTextContent("none is filtered again");
  });

  it("sends the switch with the run", async () => {
    listDatasets.mockResolvedValue([ds("s1"), ds("s2")]);
    planIntegration.mockResolvedValue({ ...PLAN, filtering: rows });
    runIntegration.mockImplementation(() => new Promise(() => {}));
    render(<JointEmbedding />);
    const picker = await screen.findByTestId("joint-picker");
    for (const box of within(picker).getAllByRole("checkbox")) fireEvent.click(box);
    fireEvent.click(await screen.findByRole("checkbox", { name: "Filter the datasets scView has not filtered" }));
    fireEvent.click(screen.getByText("Build joint embedding"));
    expect(runIntegration.mock.calls.at(-1)![0].filter_units).toBe(false);
  });
});
