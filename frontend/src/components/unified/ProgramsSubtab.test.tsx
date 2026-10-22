import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { ProgramsSummary, ProgramUsage } from "@/api/programs";
import type { DatasetInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getPrograms: vi.fn(),
  getProgramUsage: vi.fn(),
  enrichProgram: vi.fn(),
  runPrograms: vi.fn(),
}));
vi.mock("@/api/programs", async (orig) => ({ ...(await orig<typeof import("@/api/programs")>()), ...api }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; y?: unknown[] }[] }) => (
    <div data-testid="plot">{data.map((t) => `${t.type}:${(t.y ?? []).join(",")}`).join(";")}</div>
  ),
}));

import { ProgramsSubtab } from "./ProgramsSubtab";
import { useDatasetStore } from "@/stores/datasetStore";
import { useProgramsRunStore } from "@/stores/programsRunStore";

const DEFAULTS = {
  k: 10, seeds: 5, sweep: [5, 10, 15, 20], min_k: 2, max_k: 30, max_seeds: 20, max_sweep: 8,
  min_genes: 50, n_top: 30, stable_correlation: 0.9,
};
const RUNNABLE = { ok: true, reason: null, n_flagged: 2000, source: "log-normalised .raw" };

const SAVED: ProgramsSummary = {
  available: true,
  runnable: RUNNABLE,
  defaults: DEFAULTS,
  running: false,
  n_cells_total: 4,
  run_id: "r1",
  k: 2,
  seeds: [0, 1, 2, 3, 4],
  n_cells: 4,
  n_genes: 2000,
  source: "log-normalised .raw",
  notes: [],
  sweep: [
    { k: 2, silhouette: 0.95, mean_correlation: 0.99, n_stable: 2, error: 0.81, seconds: 2 },
    { k: 4, silhouette: 0.7, mean_correlation: 0.9, n_stable: 3, error: 0.8, seconds: 4 },
  ],
  programs: [
    {
      index: 1, name: "P1", n_members: 5, runs_found: 5, correlation: 0.99, silhouette: 0.9, stable: true,
      mean_usage: 0.6, pct_top: 0.75, top_genes: [{ gene: "CD3D", score: 2 }, { gene: "LTB", score: 1 }],
    },
    {
      index: 2, name: "P2", n_members: 4, runs_found: 4, correlation: 0.95, silhouette: 0.4, stable: false,
      mean_usage: 0.4, pct_top: 0.25, top_genes: [{ gene: "ISG15", score: 3 }, { gene: "IFI6", score: 2 }],
      top_term: { term: "HALLMARK_INTERFERON_ALPHA_RESPONSE", adjusted_pvalue: 1e-9, overlap_count: 12, gene_count: 60 },
    },
  ],
  by_group: { column: "stim", groups: ["CTRL", "STIM"], n_cells: [2, 2], mean: [[0.9, 0.3], [0.1, 0.7]] },
  method: "Consensus NMF (Kotliar et al. 2019): ...",
};

const USAGE: ProgramUsage = {
  index: 2, name: "P2", raw: false, run_id: "r1", top_genes: ["ISG15", "IFI6"], values: [0, 0.2, 0.8, 0.9],
  violin: { CTRL: [0, 0.2], STIM: [0.8, 0.9] },
};

function setDataset() {
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: {
      id: "d1", name: "d1", filename: "", n_cells: 4, n_genes: 100, status: "ready", available_embeddings: [],
      created_at: "", obs_columns: [{ name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] }],
    } as DatasetInfo,
  });
}

describe("ProgramsSubtab", () => {
  beforeEach(() => {
    Object.values(api).forEach((f) => f.mockReset());
    api.getPrograms.mockResolvedValue(SAVED);
    api.getProgramUsage.mockResolvedValue(USAGE);
    useProgramsRunStore.getState().reset();
    useProgramsRunStore.setState({ version: 0 });
    setDataset();
  });

  it("lists the saved programs with stability, genes, the sweep and the group heatmap", async () => {
    render(<ProgramsSubtab onUsage={vi.fn()} activeIndex={null} groupByColumn="stim" setGroupByColumn={() => {}} />);
    await waitFor(() => screen.getByTestId("program-row-1"));
    expect(api.getPrograms).toHaveBeenCalledWith("d1", "stim");
    const row1 = within(screen.getByTestId("program-row-1"));
    expect(row1.getByText("stable")).toBeTruthy();
    expect(row1.getByText("CD3D")).toBeTruthy();
    expect(row1.getByText("highest in CTRL (90%)")).toBeTruthy();
    expect(row1.getByText("75%")).toBeTruthy();
    const row2 = within(screen.getByTestId("program-row-2"));
    expect(row2.getByText("4/5 runs").getAttribute("title")).toMatch(/missing from 1 of the 5 runs/);
    expect(row2.getByText("Interferon alpha response")).toBeTruthy();
    expect(screen.getByText(/2 programs from 5 NMF runs on 2,000 variable genes/)).toBeTruthy();
    expect(screen.getByText(/1 stable/)).toBeTruthy();
    const plots = screen.getAllByTestId("plot").map((p) => p.textContent);
    expect(plots).toContain("scatter:0.95,0.7;scatter:0.81,0.8");
    expect(plots).toContain("heatmap:P1 (CD3D, LTB),P2 (ISG15, IFI6)");
    expect(screen.getByText(/k = 4: 3 of 4 stable/)).toBeTruthy();
  });

  it("colours the cells by a program and enriches one", async () => {
    const onUsage = vi.fn();
    render(<ProgramsSubtab onUsage={onUsage} activeIndex={2} groupByColumn="stim" setGroupByColumn={() => {}} />);
    await waitFor(() => screen.getByTestId("program-row-2"));
    expect(screen.getByTestId("program-row-2").className).toMatch(/bg-blue-50/);
    fireEvent.click(screen.getByTestId("program-row-2"));
    await waitFor(() => expect(onUsage).toHaveBeenCalledWith(USAGE));
    expect(api.getProgramUsage).toHaveBeenCalledWith("d1", 2, "stim");

    api.enrichProgram.mockResolvedValue({
      index: 1, name: "P1", run_id: "r1", collections: ["h.all"], query: ["CD3D", "LTB"], n_query: 2,
      n_excluded_ribo_mito: 0, background_n: 2000, n_terms: 1, n_significant: 1, warning: "",
      results: [{ term: "HALLMARK_IL2_STAT5_SIGNALING", pvalue: 1e-4, adjusted_pvalue: 2e-4, overlap_count: 2,
        gene_count: 150, genes: ["CD3D", "LTB"], collection: "h.all" }],
    });
    fireEvent.click(within(screen.getByTestId("program-row-1")).getByText("Enrich"));
    await waitFor(() => screen.getByTestId("programs-enrichment"));
    expect(api.enrichProgram).toHaveBeenCalledWith("d1", 1);
    expect(onUsage).toHaveBeenCalledTimes(1); // the button does not also colour the cells
    const table = within(screen.getByTestId("programs-enrichment"));
    expect(table.getByText("Il2 stat5 signaling")).toBeTruthy();
    expect(table.getByText("2/150")).toBeTruthy();
    expect(table.getByText(/n = 2 top genes tested; 2,000 genes in the universe/)).toBeTruthy();
    await waitFor(() => expect(api.getPrograms).toHaveBeenCalledTimes(2)); // its best term in the list
  });

  it("runs with k, seeds and a sweep, shows progress and reloads when it saves", async () => {
    api.getPrograms.mockResolvedValue({ available: false, runnable: RUNNABLE, defaults: DEFAULTS, running: false, n_cells_total: 4 });
    let finish: (s: ProgramsSummary) => void = () => {};
    api.runPrograms.mockImplementation((_id, _body, h) => {
      h.onStart?.({ k: 8, seeds: [0, 1, 2], ks: [5, 8], n_cells: 4, n_genes: 2000, n_flagged: 2000,
        source: "log-normalised .raw", notes: [], estimate_seconds: 40 });
      h.onProgress?.({ k: 5, run: 2, runs: 3, done: 2, total: 6, elapsed_s: 7 });
      return new Promise((resolve) => { finish = resolve; });
    });
    render(<ProgramsSubtab onUsage={vi.fn()} activeIndex={null} groupByColumn="" setGroupByColumn={() => {}} />);
    await waitFor(() => screen.getByText("Find programs"));
    fireEvent.change(screen.getByLabelText("Number of programs (k)"), { target: { value: "8" } });
    fireEvent.change(screen.getByLabelText("Number of seeds"), { target: { value: "3" } });
    fireEvent.click(screen.getByLabelText(/compare k/));
    fireEvent.change(screen.getByLabelText("Values of k to compare"), { target: { value: "5, 8" } });
    fireEvent.click(screen.getByText("Find programs"));
    expect(api.runPrograms.mock.calls[0]![1]).toEqual({ k: 8, seeds: 3, sweep: [5, 8] });
    await waitFor(() => expect(screen.getByTestId("programs-progress").textContent).toMatch(/2 of 6 NMF runs done \(k = 5\)/));
    expect(screen.getByTestId("programs-progress").textContent).toMatch(/about 40 s in all/);
    expect(screen.getByText("Cancel")).toBeTruthy();
    api.getPrograms.mockResolvedValue(SAVED);
    finish(SAVED);
    await waitFor(() => screen.getByTestId("program-row-1"));
    expect(screen.getByText("Run again")).toBeTruthy();
  });

  it("says why a dataset cannot run and flags a bad sweep", async () => {
    api.getPrograms.mockResolvedValueOnce({
      available: false, defaults: DEFAULTS, running: false, n_cells_total: 4,
      runnable: { ok: false, reason: "X holds scaled (centred) values", n_flagged: null, source: null },
    });
    const { unmount } = render(<ProgramsSubtab onUsage={vi.fn()} activeIndex={null} groupByColumn="" setGroupByColumn={() => {}} />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/can't run on this dataset: X holds scaled/));
    expect(screen.queryByText("Find programs")).toBeNull();
    unmount();
    api.getPrograms.mockResolvedValue({ available: false, runnable: RUNNABLE, defaults: DEFAULTS, running: false, n_cells_total: 4 });
    render(<ProgramsSubtab onUsage={vi.fn()} activeIndex={null} groupByColumn="" setGroupByColumn={() => {}} />);
    await waitFor(() => screen.getByText("Find programs"));
    fireEvent.click(screen.getByLabelText(/compare k/));
    fireEvent.change(screen.getByLabelText("Values of k to compare"), { target: { value: "5, 99" } });
    expect(screen.getByText("Each k must be a whole number from 2 to 30.")).toBeTruthy();
    expect((screen.getByText("Find programs").closest("button") as HTMLButtonElement).disabled).toBe(true);
  });
});
