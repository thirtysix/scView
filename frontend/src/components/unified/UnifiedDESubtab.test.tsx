import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import type { DEResponse } from "@/api/de";
import type { DatasetInfo } from "@/api/types";
import { DE_DEFAULTS, useDESettingsStore } from "@/stores/deSettingsStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useGeneListsStore } from "@/stores/geneListsStore";

const { computeDE, computeDECompare, computeDEMethods, computeDECompareMethods, fetchDEComposition } = vi.hoisted(() => ({
  computeDE: vi.fn(),
  computeDECompare: vi.fn(),
  computeDEMethods: vi.fn(),
  computeDECompareMethods: vi.fn(),
  // Between groups, the composition note fetches as soon as two groups are picked.
  fetchDEComposition: vi.fn<(id: string, params: object) => Promise<unknown>>(() => new Promise(() => {})),
}));
vi.mock("@/api/de", () => ({ computeDE, computeDECompare, computeDEMethods, computeDECompareMethods, fetchDEComposition }));
// The follow-up cards fetch on their own; not what is under test here.
vi.mock("@/components/unified/DEEnrichment", () => ({ DEEnrichment: () => null }));
vi.mock("@/components/unified/DEActivity", () => ({ DEActivity: () => null }));
type PlotProps = {
  data: { x: number[]; y: number[]; marker?: { color?: string[] } }[];
  layout: { shapes?: { x0: number; line: { dash: string } }[]; xaxis?: { ticktext?: string[] } };
};
/** The latest render of the detection histogram (the volcano is a Plot too). */
const histogram = () => plots.props.filter((p) => p.layout.xaxis?.ticktext?.includes("<0.1%")).at(-1);
const plots = vi.hoisted(() => ({ props: [] as PlotProps[] }));
vi.mock("@/components/plots/Plot", () => ({
  default: (props: PlotProps) => {
    plots.props.push(props);
    return <div data-testid="plot" />;
  },
}));

import { UnifiedDESubtab } from "./UnifiedDESubtab";

const ds = (id = "ds"): DatasetInfo => ({
  id,
  name: id,
  filename: `${id}.h5ad`,
  n_cells: 100,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] },
    { name: "ind", dtype: "category", n_unique: 3, values: ["d1", "d2", "d3"] },
  ],
  created_at: "",
});

const noop = () => {};
const renderDE = () => render(<UnifiedDESubtab onGeneClick={noop} onScoreGeneSet={noop} />);
const selects = () => screen.getAllByRole("combobox") as HTMLSelectElement[];

beforeEach(() => {
  useDESettingsStore.setState({ ...DE_DEFAULTS, scopedHighlight: null });
  useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  useDatasetStore.getState().setCurrentDataset(ds());
});

describe("UnifiedDESubtab settings", () => {
  it("keep their values across a tab switch (a remount)", () => {
    const first = renderDE();
    fireEvent.click(screen.getByRole("button", { name: "Between groups" }));
    // Compare stim, CTRL vs STIM, within cell_type = T
    fireEvent.change(selects()[1]!, { target: { value: "CTRL" } });
    fireEvent.change(selects()[2]!, { target: { value: "STIM" } });
    fireEvent.change(selects()[3]!, { target: { value: "cell_type" } });
    fireEvent.change(selects()[4]!, { target: { value: "T" } });
    first.unmount();

    renderDE();
    expect(screen.getByRole("button", { name: "Between groups" })).toHaveClass("bg-primary");
    expect(selects().map((s) => s.value).slice(0, 5)).toEqual(["stim", "CTRL", "STIM", "cell_type", "T"]);
  });

  it("take a highlighted group as the scope once, not over a scope picked since", () => {
    useDESettingsStore.getState().update({ mode: "groups" });
    const first = renderDE();
    act(() => useSelectionStore.getState().setHighlight({ column: "cell_type", value: "NK" }));
    expect(selects()[4]!.value).toBe("NK");
    fireEvent.change(selects()[4]!, { target: { value: "B" } });
    first.unmount();
    renderDE();
    expect(selects()[4]!.value).toBe("B");
  });

  it("start from the defaults on another dataset", () => {
    useDESettingsStore.getState().update({ mode: "groups", splitBy: "ind", fcThreshold: 3 });
    act(() => useDatasetStore.getState().setCurrentDataset(ds("other")));
    renderDE();
    expect(screen.getByRole("button", { name: "Selection vs rest" })).toHaveClass("bg-primary");
    expect(useDESettingsStore.getState().fcThreshold).toBe(1);
  });
});

/** A 0.01 % grid detection histogram: 5,930 genes under 10 %, 8,123 at or over. */
function detection(): number[] {
  const h = new Array<number>(10_001).fill(0);
  h[0] = 2000;
  h[300] = 2930; // 3 %
  h[700] = 1000; // 7 %
  h[2000] = 8123; // 20 %
  return h;
}

const RESULT: DEResponse = {
  n_selected: 40,
  n_rest: 60,
  label: "selection",
  genes: [{ gene: "CD14", logfoldchange: 3, pval: 1e-10, pval_adj: 1e-8, stat: 7 }],
  min_pct: 0.1,
  n_genes_tested: 8123,
  n_genes_skipped: 5930,
  detection: detection(),
};

describe("UnifiedDESubtab gene prefilter", () => {
  beforeEach(() => {
    plots.props = [];
    computeDE.mockReset().mockResolvedValue(RESULT);
    computeDECompare.mockReset().mockResolvedValue(RESULT);
  });

  it("sends the detection cut-off picked, 1% by default", async () => {
    act(() => useSelectionStore.getState().setSelection(new Set([1, 2, 3, 4])));
    renderDE();
    const pick = screen.getByRole("combobox", { name: "Genes tested" }) as HTMLSelectElement;
    expect(pick.value).toBe("0.01");
    fireEvent.change(pick, { target: { value: "0.1" } });
    expect(useDESettingsStore.getState().minPct).toBe(0.1);
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await waitFor(() => expect(computeDE).toHaveBeenCalled());
    expect(computeDE.mock.calls[0]![3]).toBe(0.1);
  });

  it("says how many genes were tested and skipped, and marks the cut-off on the histogram", async () => {
    useDESettingsStore.getState().update({ minPct: 0.1 }); // what RESULT was computed with
    act(() => useSelectionStore.getState().setSelection(new Set([1, 2, 3, 4])));
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await screen.findByText(/8,123 genes tested; 5,930 detected in under 10%/);
    const h = histogram()!;
    expect(h.layout.shapes!.map((s) => [s.x0, s.line.dash])).toEqual([[-1, "dash"]]);
    // The under-0.1 %, 3 % and 7 % bars are grey (left out), the 20 % bar blue.
    const bars = h.data[0]!;
    // The 0.1-decade bar holding a fraction (bar 0 is the one under 0.1 %).
    const colour = (frac: number) =>
      bars.marker!.color![bars.x.findIndex((v, i) => i > 0 && Math.abs(v - Math.log10(frac)) <= 0.05)];
    expect([bars.marker!.color![0], colour(0.03), colour(0.07), colour(0.2)]).toEqual([
      "#cbd5e1", "#cbd5e1", "#cbd5e1", "#60a5fa",
    ]);

    // Another cut-off is previewed from the histogram before computing again.
    const pick = () => screen.getByRole("combobox", { name: "Genes tested" });
    fireEvent.change(pick(), { target: { value: "0.05" } });
    expect(screen.getByText(/At 5% \(dotted line\), 9,123 genes would be tested/)).toBeTruthy();
    expect(histogram()!.layout.shapes!.map((s) => s.line.dash)).toEqual(["dash", "dot"]);
    fireEvent.change(pick(), { target: { value: "0.01" } });
    expect(screen.getByText(/At 1% \(dotted line\), 12,053 genes would be tested/)).toBeTruthy();
    fireEvent.change(pick(), { target: { value: "0" } });
    expect(screen.getByText(/With no cut-off, 14,053 genes would be tested/)).toBeTruthy();
  });

  it("applies to the cell-level test between groups, not to pseudobulk", async () => {
    useDESettingsStore.getState().update({ mode: "groups", minPct: 0.25 });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await waitFor(() => expect(computeDECompare).toHaveBeenCalled());
    expect(computeDECompare.mock.calls[0]![1]).toMatchObject({ min_pct: 0.25 });
    expect(computeDECompare.mock.calls[0]![1]).not.toHaveProperty("method");

    fireEvent.click(screen.getByRole("button", { name: "Pseudobulk (donor-aware)" }));
    expect(screen.queryByRole("combobox", { name: "Genes tested" })).toBeNull();
  });
});

const SIZE = {
  source: "obs['n_counts']",
  median_a: 3873,
  median_b: 1403,
  ratio: 2.761,
  caution: true,
  threshold: 1.5,
  values_a: [3500, 3900, 4200],
  values_b: [1300, 1400, 1500],
};

describe("UnifiedDESubtab transcriptome-size caution", () => {
  beforeEach(() => {
    plots.props = [];
    computeDECompare.mockReset();
    useDESettingsStore.getState().update({ mode: "groups", splitBy: "cell_type", groupA: "NK", groupB: "T" });
  });

  it("warns and draws each side's total counts as a violin", async () => {
    computeDECompare.mockResolvedValue({ ...RESULT, label: "NK", transcriptome_size: SIZE });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    const alert = await screen.findByText(/Transcriptome size differs/);
    expect(alert.parentElement!.textContent).toContain("NK cells hold 2.76x the RNA of T cells");
    expect(alert.parentElement!.textContent).toContain("reads as log2FC -1.46 here");
    const violin = plots.props.find((p) => (p.data as { type?: string }[])[0]?.type === "violin")!;
    const traces = violin.data as unknown as { name: string; y: number[] }[];
    expect(traces.map((t) => [t.name, t.y])).toEqual([
      ["NK", [3500, 3900, 4200]],
      ["T", [1300, 1400, 1500]],
    ]);
  });

  it("only notes the medians when they are close", async () => {
    computeDECompare.mockResolvedValue({
      ...RESULT, label: "NK", transcriptome_size: { ...SIZE, median_a: 1500, ratio: 1.069, caution: false },
    });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await screen.findByText(/Median total counts per cell: NK 1,500, T 1,403 \(1.07x/);
    expect(screen.queryByText(/Transcriptome size differs/)).toBeNull();
    expect(plots.props.some((p) => (p.data as { type?: string }[])[0]?.type === "violin")).toBe(false);
  });
});

const LOGREG: DEResponse = {
  n_selected: 40,
  n_rest: 60,
  label: "selection",
  method: "logreg",
  genes: [
    { gene: "CD14", logfoldchange: 3, pval: null, pval_adj: null, stat: 0.42 },
    { gene: "LYZ", logfoldchange: 2, pval: null, pval_adj: null, stat: 0.9 },
  ],
  min_pct: 0.01,
  n_genes_tested: 2,
  n_genes_skipped: 0,
};

const AGREEMENT = {
  n_selected: 4,
  n_rest: 96,
  label: "selection",
  top_n: 50,
  methods: [
    { method: "wilcoxon", label: "Wilcoxon rank-sum", stat_label: "z-score", has_pvalues: true, n_significant: 3242, seconds: 11.7 },
    { method: "t-test", label: "Welch t-test", stat_label: "t statistic", has_pvalues: true, n_significant: 6938, seconds: 0.5 },
    { method: "logreg", label: "Logistic regression", stat_label: "coefficient", has_pvalues: false, n_significant: null, seconds: 1 },
  ],
  pairs: [
    { a: "wilcoxon", b: "t-test", jaccard_up: 0.9608, shared_up: 49, jaccard_down: 0.6667, shared_down: 40, spearman: 0.9765 },
  ],
  unique: [{ gene: "PPBP", method: "logreg", side: "down", ranks: { wilcoxon: 2464, "t-test": 2702, logreg: 1 } }],
  unavailable: { MAST: "MAST is an R package." },
  min_pct: 0.01,
  n_genes_tested: 8086,
  n_genes_skipped: 5967,
};

describe("UnifiedDESubtab cell-level method", () => {
  beforeEach(() => {
    plots.props = [];
    computeDE.mockReset().mockResolvedValue(RESULT);
    computeDECompare.mockReset().mockResolvedValue(RESULT);
    computeDEMethods.mockReset().mockResolvedValue(AGREEMENT);
    computeDECompareMethods.mockReset().mockResolvedValue({ ...AGREEMENT, label: "STIM" });
  });

  it("sends the picked test, and no method field for the Wilcoxon default", async () => {
    act(() => useSelectionStore.getState().setSelection(new Set([1, 2, 3, 4])));
    renderDE();
    const pick = screen.getByRole("combobox", { name: "Cell-level test" }) as HTMLSelectElement;
    expect(pick.value).toBe("wilcoxon");
    // MAST is listed, but cannot be picked.
    const mast = Array.from(pick.options).find((o) => o.value === "mast")!;
    expect(mast.disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await waitFor(() => expect(computeDE).toHaveBeenCalledTimes(1));
    expect(computeDE.mock.calls[0]![4]).toBe("wilcoxon");
    fireEvent.change(pick, { target: { value: "t-test" } });
    expect(useDESettingsStore.getState().cellMethod).toBe("t-test");
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await waitFor(() => expect(computeDE).toHaveBeenCalledTimes(2));
    expect(computeDE.mock.calls[1]![4]).toBe("t-test");
  });

  it("passes the test between groups, not for pseudobulk", async () => {
    useDESettingsStore.getState().update({ mode: "groups", cellMethod: "logreg" });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await waitFor(() => expect(computeDECompare).toHaveBeenCalled());
    expect(computeDECompare.mock.calls[0]![1]).toMatchObject({ method: "logreg" });
    fireEvent.click(screen.getByRole("button", { name: "Pseudobulk (donor-aware)" }));
    expect(screen.queryByRole("combobox", { name: "Cell-level test" })).toBeNull();
  });

  it("shows a logistic regression by coefficient, without p-values", async () => {
    computeDE.mockResolvedValue(LOGREG);
    useDESettingsStore.getState().update({ cellMethod: "logreg" });
    act(() => useSelectionStore.getState().setSelection(new Set([1, 2, 3, 4])));
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await screen.findByText(/gives a coefficient per gene and no p-values/);
    expect(screen.getByText("Coef.")).toBeTruthy();
    expect(screen.queryByText("Adj.P")).toBeNull();
    expect(screen.getByText(/2 listed of 2 genes/)).toBeTruthy();
    // Largest coefficient first.
    const rows = screen.getAllByRole("row").slice(1).map((r) => r.textContent);
    expect(rows[0]).toContain("LYZ");
    expect(rows[0]).toContain("0.900");
    const volcano = plots.props.at(-1)!;
    expect((volcano.layout as { yaxis?: { title?: { text?: string } } }).yaxis?.title?.text).toBe(
      "logistic regression coefficient",
    );
  });

  it("compares every test on the selection and shows the agreement", async () => {
    act(() => useSelectionStore.getState().setSelection(new Set([1, 2, 3, 4])));
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: "Compare methods" }));
    await screen.findByText("Method agreement");
    expect(computeDEMethods.mock.calls[0]![1]).toMatchObject({ indices: [1, 2, 3, 4], min_pct: 0.01 });
    const pairs = screen.getByRole("table", { name: "Agreement between tests" });
    expect(pairs.textContent).toContain("Wilcoxon rank-sum vs Welch t-test");
    expect(pairs.textContent).toContain("0.96 (49)");
    const only = screen.getByRole("table", { name: "Genes only one test ranks top" });
    expect(only.textContent).toContain("PPBP");
    expect(only.textContent).toContain("2,464");
    expect(screen.getByText(/MAST not run/)).toBeTruthy();
    expect(screen.getByText("no p-values")).toBeTruthy();
  });

  it("compares every test between groups with the scope", async () => {
    useDESettingsStore.getState().update({
      mode: "groups", splitBy: "stim", groupA: "STIM", groupB: "CTRL", withinCol: "cell_type", withinVal: "NK",
    });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: "Compare methods" }));
    await screen.findByText("Method agreement");
    expect(computeDECompareMethods.mock.calls[0]![1]).toEqual({
      split_by: "stim", group_a: "STIM", group_b: "CTRL", within_column: "cell_type", within_value: "NK", min_pct: 0.01,
    });
  });
});

describe("UnifiedDESubtab gene lists", () => {
  it("adds the significant up or down genes, in table order, to a gene list", async () => {
    useGeneListsStore.setState({ lists: [], selectedId: null });
    computeDECompare.mockReset().mockResolvedValue({
      ...RESULT,
      label: "STIM",
      genes: [
        { gene: "ISG15", logfoldchange: 4, pval: 1e-30, pval_adj: 1e-28, stat: 20 },
        { gene: "RPL3", logfoldchange: -1, pval: 1e-12, pval_adj: 1e-10, stat: -9 },
        { gene: "IFI6", logfoldchange: 3, pval: 1e-20, pval_adj: 1e-18, stat: 15 },
        { gene: "NOTSIG", logfoldchange: 2, pval: 0.5, pval_adj: 0.9, stat: 0.1 },
      ],
    });
    useDESettingsStore.getState().update({ mode: "groups", splitBy: "stim", groupA: "STIM", groupB: "CTRL" });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    fireEvent.click(await screen.findByRole("button", { name: "Up" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "New list…" }));
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    fireEvent.click(screen.getByRole("button", { name: "Down" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "New list…" }));
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(useGeneListsStore.getState().lists.map((l) => [l.name, l.genes])).toEqual([
      ["DE: higher in STIM", ["ISG15", "IFI6"]],
      ["DE: higher in CTRL", ["RPL3"]],
    ]);
  });
});

describe("UnifiedDESubtab table cap", () => {
  it("names the ordering and the total when the table is cut at 200 rows", async () => {
    const genes = Array.from({ length: 250 }, (_, i) => ({
      gene: `G${i}`, logfoldchange: 2 + (i % 5), pval: 1e-9, pval_adj: 1e-8 * (i + 1), stat: 5,
    }));
    computeDECompare.mockReset().mockResolvedValue({ ...RESULT, label: "STIM", genes });
    useDESettingsStore.getState().update({ mode: "groups", splitBy: "stim", groupA: "STIM", groupB: "CTRL" });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    expect((await screen.findByTestId("cap-note")).textContent).toBe(
      "Top 200 of 250 by adjusted p, then |log2FC|; the CSV has all.",
    );
    expect(screen.getAllByRole("row")).toHaveLength(201); // header + 200
  });

  it("shows no cap line when every significant gene fits", async () => {
    computeDECompare.mockReset().mockResolvedValue({ ...RESULT, label: "STIM" });
    useDESettingsStore.getState().update({ mode: "groups", splitBy: "stim", groupA: "STIM", groupB: "CTRL" });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await screen.findByText("CD14");
    expect(screen.queryByTestId("cap-note")).toBeNull();
  });
});

describe("UnifiedDESubtab table headers", () => {
  it("say what log2FC and the adjusted p are and how they were computed", async () => {
    computeDECompare.mockReset().mockResolvedValue({ ...RESULT, label: "STIM", method: "wilcoxon" });
    useDESettingsStore.getState().update({ mode: "groups", splitBy: "stim", groupA: "STIM", groupB: "CTRL" });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await screen.findByText("CD14");
    expect(screen.getByRole("columnheader", { name: "log2FC" })).toHaveAttribute(
      "title",
      "log2 of the mean expression in STIM over CTRL (expm1 of the mean log-normalised values on each side, as scanpy's rank_genes_groups takes it)",
    );
    expect(screen.getByRole("columnheader", { name: "Adj.P" })).toHaveAttribute(
      "title",
      "Wilcoxon rank-sum p-value, Benjamini-Hochberg adjusted over the 1 tested genes",
    );
  });
});

describe("UnifiedDESubtab composition between groups", () => {
  const COMPOSITION = {
    column: "cell_type", label_a: "STIM", label_b: "CTRL", auto: true, n_a: 60, n_b: 40,
    groups: [
      { group: "NK", n_a: 30, n_b: 8, share_a: 0.5, share_b: 0.2, diff_points: 30, flagged: true, one_sided: false },
      { group: "T", n_a: 30, n_b: 32, share_a: 0.5, share_b: 0.8, diff_points: -30, flagged: true, one_sided: false },
    ],
    n_groups: 2, warn: true, note: null, threshold_points: 10, min_cells: 10,
    message: "NK is 50% of STIM cells but 20% of CTRL cells (+30 points): genes specific to NK will look like STIM vs CTRL differences.",
  };

  beforeEach(() => {
    fetchDEComposition.mockReset().mockResolvedValue(COMPOSITION);
    computeDECompare.mockReset().mockResolvedValue({ ...RESULT, label: "STIM", composition: COMPOSITION });
    useDESettingsStore.getState().update({ mode: "groups", splitBy: "stim", groupA: "STIM", groupB: "CTRL" });
  });

  it("shows each side's shares before the DE, and a chip scopes the DE to a group", async () => {
    renderDE();
    expect(await screen.findByText("Composition differs between the sides")).toBeInTheDocument();
    expect(fetchDEComposition.mock.calls.at(-1)![1]).toMatchObject({ split_by: "stim", group_a: "STIM", group_b: "CTRL" });
    fireEvent.click(screen.getByRole("button", { name: "Compare within NK" }));
    expect(useDESettingsStore.getState()).toMatchObject({ withinCol: "cell_type", withinVal: "NK" });
    await waitFor(() =>
      expect(fetchDEComposition.mock.calls.at(-1)![1]).toMatchObject({ within_column: "cell_type", within_value: "NK" }),
    );
    expect(computeDECompare).not.toHaveBeenCalled();
  });

  it("sends the picked composition column with the DE, and none on Auto", async () => {
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await waitFor(() => expect(computeDECompare).toHaveBeenCalledTimes(1));
    expect(computeDECompare.mock.calls[0]![1]).not.toHaveProperty("composition_by");
    fireEvent.change(screen.getByRole("combobox", { name: "Composition by" }), { target: { value: "ind" } });
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await waitFor(() => expect(computeDECompare).toHaveBeenCalledTimes(2));
    expect(computeDECompare.mock.calls[1]![1]).toMatchObject({ composition_by: "ind" });
  });

  it("puts the result's composition caution beside the batch caution", async () => {
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    const caution = await screen.findByTestId("de-composition-caution");
    expect(caution).toHaveTextContent("Composition differs (by cell_type): NK is 50% of STIM cells but 20% of CTRL cells");
    expect(caution).toHaveAttribute("title", expect.stringContaining("more than 10 percentage points"));
  });

  it("shows no caution for an older result without a composition check", async () => {
    computeDECompare.mockResolvedValue({ ...RESULT, label: "STIM" });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await screen.findByText("CD14");
    expect(screen.queryByTestId("de-composition-caution")).toBeNull();
  });
});
