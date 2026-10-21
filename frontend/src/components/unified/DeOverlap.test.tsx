import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { DEOverlapResponse, DEResultSummary, DEResultsList } from "@/api/deOverlap";
import { useGeneListsStore } from "@/stores/geneListsStore";

const { listDEResults, computeDEOverlap } = vi.hoisted(() => ({
  listDEResults: vi.fn(),
  computeDEOverlap: vi.fn(),
}));
vi.mock("@/api/deOverlap", () => ({ listDEResults, computeDEOverlap }));

import { DeOverlap } from "./DeOverlap";

const summary = (id: string, label: string, minute: number, extra: Partial<DEResultSummary> = {}): DEResultSummary => ({
  result_id: id,
  label,
  method: "wilcoxon",
  created_at: `2026-10-21T10:${String(minute).padStart(2, "0")}:00+00:00`,
  n_tested: 7000,
  has_pvalues: true,
  rule: "padj",
  n_sig_up: 700,
  n_sig_down: 1000,
  ...extra,
});

const listing = (results: DEResultSummary[]): DEResultsList => ({
  results,
  keep: 8,
  padj_max: 0.05,
  lfc_min: 1,
  logreg_top_n: 50,
});

// Newest first, as the backend lists them.
const THREE = [
  summary("b", "B: STIM vs CTRL (Wilcoxon)", 3),
  summary("t", "CD4 Naive T: STIM vs CTRL (Wilcoxon)", 2, { method: "logreg", has_pvalues: false, rule: "top_n", n_sig_up: 50, n_sig_down: 50 }),
  summary("m", "CD14 Mono: STIM vs CTRL (Wilcoxon)", 1),
];

const OVERLAP: DEOverlapResponse = {
  direction: "both",
  padj_max: 0.05,
  lfc_min: 1,
  restrict_to_common_tested: false,
  logreg_top_n: 50,
  max_listed: 1000,
  sets: [
    { index: 0, name: "CD14 Mono", kind: "de", result_id: "m", method: "wilcoxon", rule: "padj", n: 5, n_up: 4, n_down: 1, n_tested: 7000, n_not_tested: null },
    { index: 1, name: "CD4 Naive T", kind: "de", result_id: "t", method: "logreg", rule: "top_n", n: 3, n_up: 3, n_down: 0, n_tested: 6900, n_not_tested: null },
    { index: 2, name: "B", kind: "de", result_id: "b", method: "wilcoxon", rule: "padj", n: 3, n_up: 3, n_down: 0, n_tested: 7100, n_not_tested: null },
  ],
  result_ids: ["m", "t", "b"],
  n_common_tested: 6500,
  n_union: 6,
  intersections: [
    { sets: [0, 1, 2], degree: 3, n: 2, n_up: 2, n_down: 0, genes: [
      { gene: "ISG15", lfc: [3.1, 2.5, 2.2] },
      { gene: "IFI6", lfc: [2.0, 1.8, 1.1] },
    ] },
    { sets: [0], degree: 1, n: 2, n_up: 1, n_down: 1, genes: [
      { gene: "NUPR1", lfc: [4, 0.1, null] },
      { gene: "CD14", lfc: [-1.5, 0, null] },
    ] },
    { sets: [2], degree: 1, n: 1, n_up: 1, n_down: 0, genes: [{ gene: "CD79B", lfc: [0, 0.2, 1.4] }] },
  ],
  n_intersections: 3,
  hidden_intersections: 0,
  hidden_genes: 0,
  discordant: { n: 1, combinations: [{ sets: [0, 1], n: 1 }], genes: [{ gene: "FLIP", lfc: [1.2, -1.1, 0], sets: [0, 1] }] },
  pairs: [{ a: 0, b: 1, shared: 2, discordant: 1, union: 6, jaccard: 0.33 }],
  notes: ["CD4 Naive T: logistic regression gives no p-values, so its set is the top 50 genes per direction."],
};

const noop = () => {};
const renderOverlap = (props: Partial<Parameters<typeof DeOverlap>[0]> = {}) =>
  render(<DeOverlap datasetId="ds" pThreshold={0.05} fcThreshold={1} onGeneClick={noop} {...props} />);
const openSection = () => fireEvent.click(screen.getByRole("button", { name: /compare de results/i }));

beforeEach(() => {
  listDEResults.mockReset().mockResolvedValue(listing(THREE));
  computeDEOverlap.mockReset().mockResolvedValue(OVERLAP);
  useGeneListsStore.setState({ lists: [], selectedId: null });
});

describe("DeOverlap empty states", () => {
  it("explains how to get results when none are kept", async () => {
    listDEResults.mockResolvedValue(listing([]));
    renderOverlap();
    openSection();
    expect(await screen.findByTestId("de-overlap-empty")).toHaveTextContent(/at least twice/);
    expect(computeDEOverlap).not.toHaveBeenCalled();
  });

  it("asks for another comparison when one is kept", async () => {
    listDEResults.mockResolvedValue(listing([THREE[0]!]));
    renderOverlap();
    openSection();
    expect(await screen.findByTestId("de-overlap-empty")).toHaveTextContent(/Run another comparison/);
  });
});

describe("DeOverlap", () => {
  it("lists the kept results in run order and compares them with the DE table's cut-offs", async () => {
    renderOverlap({ pThreshold: 0.05, fcThreshold: 1 });
    await waitFor(() => expect(listDEResults).toHaveBeenCalledWith("ds", 0.05, 1));
    expect(screen.getByRole("button", { name: /compare de results/i })).toHaveTextContent("3 kept for this dataset");
    expect(computeDEOverlap).not.toHaveBeenCalled(); // closed: nothing computed
    openSection();
    const boxes = (await screen.findAllByRole("checkbox")).slice(0, 3) as HTMLInputElement[];
    expect(boxes.map((b) => b.closest("label")!.textContent)).toEqual([
      expect.stringContaining("CD14 Mono"),
      expect.stringContaining("CD4 Naive T"),
      expect.stringContaining("B: STIM"),
    ]);
    expect(boxes.every((b) => b.checked)).toBe(true); // the newest (up to 4) start ticked
    expect(screen.getByText(/50 up · 50 down \(top by coef\.\)/)).toBeInTheDocument();
    await waitFor(() => expect(computeDEOverlap).toHaveBeenCalledTimes(1));
    expect(computeDEOverlap).toHaveBeenCalledWith("ds", {
      result_ids: ["m", "t", "b"],
      direction: "both",
      padj_max: 0.05,
      lfc_min: 1,
      restrict_to_common_tested: false,
    });
    expect(await screen.findByTestId("upset-plot")).toBeInTheDocument();
    const caption = screen.getByTestId("de-overlap-caption");
    expect(caption).toHaveTextContent("Comparisons run since the backend started (last 8 kept");
    expect(caption).toHaveTextContent("Genes per set: CD14 Mono 5; CD4 Naive T 3; B 3");
    expect(caption).toHaveTextContent("may not have been tested");
    expect(screen.getByTestId("de-overlap-notes")).toHaveTextContent("logistic regression gives no p-values");
  });

  it("recomputes for another direction, cut-off, the common-tested toggle and fewer results", async () => {
    renderOverlap();
    openSection();
    await waitFor(() => expect(computeDEOverlap).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Up" }));
    await waitFor(() => expect(computeDEOverlap).toHaveBeenLastCalledWith("ds", expect.objectContaining({ direction: "up" })));
    fireEvent.change(screen.getByRole("spinbutton", { name: "log2FC cut-off" }), { target: { value: "0.5" } });
    await waitFor(() => expect(computeDEOverlap).toHaveBeenLastCalledWith("ds", expect.objectContaining({ lfc_min: 0.5 })));
    fireEvent.click(screen.getByRole("checkbox", { name: /only genes tested in all/ }));
    await waitFor(() =>
      expect(computeDEOverlap).toHaveBeenLastCalledWith("ds", expect.objectContaining({ restrict_to_common_tested: true })),
    );
    fireEvent.click(screen.getByRole("checkbox", { name: /CD4 Naive T/ }));
    await waitFor(() =>
      expect(computeDEOverlap).toHaveBeenLastCalledWith("ds", expect.objectContaining({ result_ids: ["m", "b"] })),
    );
    const n = computeDEOverlap.mock.calls.length;
    fireEvent.click(screen.getByRole("checkbox", { name: /CD14 Mono/ }));
    expect(await screen.findByText("Tick two or more results to compare them.")).toBeInTheDocument();
    await act(() => new Promise((r) => setTimeout(r, 300)));
    expect(computeDEOverlap).toHaveBeenCalledTimes(n); // one result: nothing to compare
  });

  it("refreshes the list when the DE subtab has a new result, and ticks the new one", async () => {
    const view = renderOverlap({ latestResultId: "b" });
    openSection();
    await waitFor(() => expect(computeDEOverlap).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("checkbox", { name: /CD4 Naive T/ })); // picked by hand from now on
    await waitFor(() =>
      expect(computeDEOverlap).toHaveBeenLastCalledWith("ds", expect.objectContaining({ result_ids: ["m", "b"] })),
    );
    listDEResults.mockResolvedValue(listing([summary("new", "NK: STIM vs CTRL (Wilcoxon)", 4), ...THREE]));
    view.rerender(<DeOverlap datasetId="ds" pThreshold={0.05} fcThreshold={1} onGeneClick={noop} latestResultId="new" />);
    await waitFor(() => expect(listDEResults).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(computeDEOverlap).toHaveBeenLastCalledWith("ds", expect.objectContaining({ result_ids: ["m", "b", "new"] })),
    );
    expect(screen.getByRole("checkbox", { name: /CD4 Naive T/ })).not.toBeChecked(); // unticked stays so
  });

  it("shows an intersection's genes with each result's log2FC, to copy or save as a gene list", async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.assign(navigator, { clipboard: { writeText } });
    const onGeneClick = vi.fn();
    renderOverlap({ onGeneClick });
    openSection();
    expect(await screen.findByText("Click a bar for its genes.")).toBeInTheDocument();
    const cols = screen.getAllByTestId("upset-column");
    expect(cols).toHaveLength(3);
    expect(cols[0]).toHaveAttribute("aria-label", "2 genes in all 3 sets (2 up, 0 down)");
    fireEvent.click(cols[1]!);
    const table = await screen.findByTestId("de-overlap-genes");
    expect(table).toHaveTextContent("2 genes in only CD14 Mono (1 up, 1 down)");
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows.map((r) => r.textContent)).toEqual(["NUPR14.000.10–", "CD14-1.500.00–"]);
    // In the set: bold and coloured by sign; elsewhere grey; untested: a dash that says so.
    const cells = within(rows[0]!).getAllByRole("cell");
    expect(cells[1]).toHaveClass("font-semibold", "text-red-600");
    expect(cells[2]).toHaveClass("text-slate-400");
    expect(cells[3]).toHaveAttribute("title", "Not tested in this result");
    fireEvent.click(rows[1]!);
    expect(onGeneClick).toHaveBeenCalledWith("CD14");

    fireEvent.click(within(table).getByRole("button", { name: /copy/i }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("NUPR1\nCD14"));
    expect(await within(table).findByText("Copied 2 genes")).toBeInTheDocument();

    fireEvent.click(within(table).getByRole("button", { name: /save as gene list/i }));
    fireEvent.click(within(table).getByRole("menuitem", { name: /new list/i }));
    fireEvent.click(within(table).getByRole("button", { name: "Add" }));
    const lists = useGeneListsStore.getState().lists;
    expect(lists).toHaveLength(1);
    expect(lists[0]).toMatchObject({ name: "DE overlap: only CD14 Mono", genes: ["NUPR1", "CD14"] });
  });

  it("lists discordant genes apart, marking the results they are significant in", async () => {
    renderOverlap();
    openSection();
    fireEvent.click(await screen.findByRole("button", { name: /Discordant: 1 gene up in one result, down in another/ }));
    const table = await screen.findByTestId("de-overlap-genes");
    const cells = within(within(table).getAllByRole("row")[1]!).getAllByRole("cell");
    expect(cells.map((c) => c.textContent)).toEqual(["FLIP", "1.20", "-1.10", "0.00"]);
    expect(cells[1]).toHaveClass("text-red-600");
    expect(cells[2]).toHaveClass("text-blue-600");
    expect(cells[3]).toHaveClass("text-slate-400");
  });

  it("adds a gene list as an extra set", async () => {
    useGeneListsStore.getState().create("My ISGs", ["ISG15", "MX1"]);
    renderOverlap();
    openSection();
    await waitFor(() => expect(computeDEOverlap).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("checkbox", { name: "My ISGs" }));
    await waitFor(() =>
      expect(computeDEOverlap).toHaveBeenLastCalledWith(
        "ds",
        expect.objectContaining({ extra_sets: [{ name: "My ISGs", genes: ["ISG15", "MX1"] }] }),
      ),
    );
  });

  it("says when a result is no longer kept, and refreshes the list", async () => {
    computeDEOverlap.mockRejectedValue(
      new Error('API error 404: {"detail":"That DE result is no longer kept; the backend keeps the last 8 since it started. Run the DE again."}'),
    );
    renderOverlap();
    openSection();
    expect(await screen.findByText(/That DE result is no longer kept; the backend keeps the last 8/)).toBeInTheDocument();
    await waitFor(() => expect(listDEResults.mock.calls.length).toBeGreaterThanOrEqual(2));
  });
});
