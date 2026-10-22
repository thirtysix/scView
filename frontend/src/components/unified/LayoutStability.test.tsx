import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const api = vi.hoisted(() => ({
  getLayoutsInfo: vi.fn(),
  runLayouts: vi.fn(),
  getLayoutThumb: vi.fn(),
  keepLayout: vi.fn(),
  getDataset: vi.fn(),
  fetchEmbeddingBinary: vi.fn(),
  decodeArrowBuffer: vi.fn(),
}));
vi.mock("@/api/layouts", () => ({
  getLayoutsInfo: api.getLayoutsInfo,
  runLayouts: api.runLayouts,
  getLayoutThumb: api.getLayoutThumb,
  keepLayout: api.keepLayout,
}));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary: api.fetchEmbeddingBinary }));
vi.mock("@/lib/arrow", () => ({ decodeArrowBuffer: api.decodeArrowBuffer }));

import type { LayoutPlan, LayoutRelation, LayoutsInfo, LayoutSummary, LayoutThumb } from "@/api/layouts";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { HALO } from "@/lib/layoutStability";
import { LayoutStabilityPanel } from "./LayoutStability";

const COL = "cluster";
const plan: LayoutPlan = {
  recorded: { use_rep: "X_pca", n_pcs: null, metric: "euclidean", n_neighbors: 15, min_dist: 0.5, a: null, b: null, notes: [] },
  seeds: [0, 1],
  n_neighbors: [10, 30],
  layouts: [
    { index: 0, seed: 0, n_neighbors: 10, key: "X_umap_s0_k10", label: "seed 0, 10 neighbours" },
    { index: 1, seed: 1, n_neighbors: 10, key: "X_umap_s1_k10", label: "seed 1, 10 neighbours" },
    { index: 2, seed: 0, n_neighbors: 30, key: "X_umap_s0_k30", label: "seed 0, 30 neighbours" },
    { index: 3, seed: 1, n_neighbors: 30, key: "X_umap_s1_k30", label: "seed 1, 30 neighbours" },
  ],
  n_cells: 600,
  n_laid_out: 600,
  sampled: false,
  group_by: COL,
  current: "X_umap",
  notes: ["For each neighbour count the kNN graph is rebuilt from X_pca."],
};

const info: LayoutsInfo = {
  available: true,
  reason: null,
  n_cells: 600,
  max_cells: 20000,
  max_layouts: 12,
  first_run_seconds: 30,
  rule: "Two groups touch when ...",
  running: false,
  plan,
  estimate_seconds: 12,
  result: null,
};

const rel = (a: string, b: string, layouts: boolean[], current: boolean, status: LayoutRelation["status"],
             flag: LayoutRelation["flag"] = null): LayoutRelation => ({
  a, b, n_a: 200, n_b: 200, layouts, n_touch: layouts.filter(Boolean).length, n_layouts: 4,
  share: layouts.filter(Boolean).length / 4, current, status, flag, contact_a: [], contact_b: [],
  current_contact: current ? [8, 9] : [0, 0],
});

const relations = [
  rel("A", "B", [false, false, true, false], true, "some", "rare_in_plot"),
  rel("B", "C", [true, true, true, true], true, "every"),
  rel("A", "C", [false, false, false, false], false, "never"),
];

const summary: LayoutSummary = {
  ...plan,
  complete: true,
  stopped: false,
  n_done: 4,
  seconds: 10,
  done: plan.layouts.map((l) => ({ ...l, seconds: 2, same_as_current: l.index === 2, exists: false })),
  rule: "Two groups touch when at least 10 of their cells ...",
  plot_k: 15,
  min_contact: 10,
  body_share: 0.05,
  relations,
  n_pairs: 3,
  n_listed: 3,
  counts: { every: 1, some: 1, never: 1, rare_in_plot: 1, hidden_in_plot: 0 },
  groups: [{ group: "A", n_cells: 200, placement: [1, 1, 0.8, 1], current: 0.8, min: 0.8, max: 1, range: 0.2 }],
};

const thumb = (index: number | "current"): LayoutThumb => ({
  index, seed: index === "current" ? null : 0, n_neighbors: 10, key: null, label: String(index),
  cells: [0, 1, 2], x: [0, 1, 2], y: [0, 1, 2], n_laid_out: 600, n_shown: 3,
});

function renderPanel() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <LayoutStabilityPanel datasetId="ds" onClose={() => {}} />
    </QueryClientProvider>,
  );
}

function runWithSummary() {
  api.runLayouts.mockImplementation(async (_id, _body, h) => {
    h.onStart({ ...plan, estimate_seconds: 12, first_run_seconds: 30 });
    for (const l of plan.layouts) h.onLayout({ ...l, seconds: 2, exists: false });
    h.onSummary(summary);
  });
}

const fills: string[] = [];

describe("LayoutStabilityPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fills.length = 0;
    useDatasetStore.setState({
      currentDatasetId: "ds",
      currentDataset: {
        id: "ds",
        available_embeddings: ["X_umap", "X_pca"],
        obs_columns: [
          { name: COL, dtype: "category", n_unique: 3, values: ["A", "B", "C"] },
          { name: "score", dtype: "float64", n_unique: 600 },
        ],
      } as unknown as DatasetInfo,
    });
    useSettingsStore.setState({ colorBy: COL, embedding: "X_umap" });
    api.getLayoutsInfo.mockResolvedValue(info);
    api.getLayoutThumb.mockImplementation(async (_id, index) => thumb(index));
    api.fetchEmbeddingBinary.mockResolvedValue(new ArrayBuffer(8));
    api.decodeArrowBuffer.mockReturnValue({ [COL]: new Int32Array([0, 1, 2]) });
    let style = "";
    const ctx = {
      setTransform: () => undefined,
      clearRect: () => undefined,
      fillRect: () => fills.push(style),
      set fillStyle(v: string) {
        style = v;
      },
      get fillStyle() {
        return style;
      },
    };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
  });
  afterEach(() => vi.restoreAllMocks());

  it("explains why the check can't run", async () => {
    api.getLayoutsInfo.mockResolvedValue({ ...info, available: false, reason: "There is no X_pca to lay the cells out from." });
    renderPanel();
    expect(await screen.findByRole("alert")).toHaveTextContent("no X_pca");
    expect(screen.queryByRole("button", { name: "Run" })).not.toBeInTheDocument();
  });

  it("runs, shows a thumbnail per layout and the plot, and lists rare contacts first", async () => {
    runWithSummary();
    renderPanel();
    expect(await screen.findByText(/About 10 s for 4 layouts of 600 cells/)).toBeInTheDocument();
    expect((screen.getByLabelText("Grouping column") as HTMLSelectElement).value).toBe(COL);
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(api.getLayoutThumb).toHaveBeenCalledTimes(5));
    expect(api.runLayouts).toHaveBeenCalledWith("ds", { group_by: COL, embedding: "X_umap" }, expect.anything());
    expect(api.getLayoutThumb.mock.calls.map((c) => c[1]).sort()).toEqual([0, 1, 2, 3, "current"].sort());
    expect(api.getLayoutThumb).toHaveBeenCalledWith("ds", "current", { embedding: "X_umap" });
    for (const id of ["thumb-current", "thumb-0", "thumb-1", "thumb-2", "thumb-3"])
      expect(await screen.findByTestId(id)).toBeInTheDocument();
    expect(within(screen.getByTestId("thumb-2")).getByText("= your plot")).toBeInTheDocument();
    expect(screen.getByTestId("layouts-headline")).toHaveTextContent(
      "The contact between A and B in your plot appears in 1 of 4 layouts.",
    );
    const rows = screen.getAllByTestId("relation-row");
    expect(rows.map((r) => r.textContent)).toEqual([
      expect.stringContaining("A — B"),
      expect.stringContaining("B — C"),
    ]);
    expect(rows[0]).toHaveTextContent("in your plot, rare");
    // The pair that never touches shows on request.
    fireEvent.click(screen.getByLabelText(/Show the 1 pairs that never touch/));
    expect(screen.getAllByTestId("relation-row")).toHaveLength(3);
    expect(screen.getByText(/n = 600 cells laid out; 4 layouts\./)).toBeInTheDocument();
  });

  it("hovering a relation outlines its two groups in every thumbnail", async () => {
    runWithSummary();
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    const rows = await screen.findAllByTestId("relation-row");
    await waitFor(() => expect(api.decodeArrowBuffer).toHaveBeenCalled());
    fills.length = 0;
    fireEvent.mouseEnter(rows[0]!);
    for (const id of ["thumb-current", "thumb-0", "thumb-1", "thumb-2", "thumb-3"])
      expect(screen.getByTestId(id)).toHaveAttribute("data-outline", "A|B");
    // Touch / apart per layout: A and B touch only in layout 2 (and in the plot).
    expect(within(screen.getByTestId("thumb-2")).getByText("touch")).toBeInTheDocument();
    expect(within(screen.getByTestId("thumb-0")).getByText("apart")).toBeInTheDocument();
    expect(within(screen.getByTestId("thumb-current")).getByText("touch")).toBeInTheDocument();
    // Redrawn with the halo around A's and B's cells.
    await waitFor(() => expect(fills).toContain(`rgb(${HALO.join(",")})`));
    expect(screen.getByText(/Outlined: A and B\./)).toBeInTheDocument();
    fireEvent.mouseLeave(rows[0]!);
    expect(screen.getByTestId("thumb-0")).toHaveAttribute("data-outline", "");
  });

  it("keeps a layout as an embedding and then shows it", async () => {
    runWithSummary();
    api.keepLayout.mockResolvedValue({ key: "X_umap_s1_k10", created: true, n_cells: 600, recomputed: false, note: null });
    api.getDataset.mockResolvedValue({ id: "ds", available_embeddings: ["X_umap", "X_pca", "X_umap_s1_k10"], obs_columns: [] });
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    const t1 = await screen.findByTestId("thumb-1");
    fireEvent.click(await within(t1).findByRole("button", { name: "Keep this layout" }));
    await waitFor(() => expect(api.keepLayout).toHaveBeenCalledWith("ds", 1));
    await waitFor(() => expect(api.getDataset).toHaveBeenCalledWith("ds"));
    // The refresh refetches the plot's colours (the embedding list changed).
    await waitFor(() => expect(api.fetchEmbeddingBinary).toHaveBeenCalledTimes(2));
    fireEvent.click(await within(screen.getByTestId("thumb-1")).findByRole("button", { name: "Show" }));
    expect(useSettingsStore.getState().embedding).toBe("X_umap_s1_k10");
    // The kept layout is now the plot: the comparison is asked for it.
    await waitFor(() =>
      expect(api.getLayoutsInfo).toHaveBeenLastCalledWith("ds", { groupBy: COL, embedding: "X_umap_s1_k10" }),
    );
  });

  it("shows a run the server still holds without running again", async () => {
    api.getLayoutsInfo.mockResolvedValue({ ...info, result: summary });
    renderPanel();
    expect(await screen.findByRole("button", { name: "Run again" })).toBeInTheDocument();
    await waitFor(() => expect(api.getLayoutThumb).toHaveBeenCalledTimes(5));
    expect(api.runLayouts).not.toHaveBeenCalled();
    expect(screen.getAllByTestId("relation-row")).toHaveLength(2);
  });

  it("asks the server again for another grouping, without new layouts", async () => {
    useDatasetStore.setState({
      currentDataset: {
        ...useDatasetStore.getState().currentDataset!,
        obs_columns: [
          { name: COL, dtype: "category", n_unique: 3, values: ["A", "B", "C"] },
          { name: "cell_type", dtype: "category", n_unique: 2, values: ["T", "B"] },
        ],
      } as unknown as DatasetInfo,
    });
    api.getLayoutsInfo.mockResolvedValue({ ...info, result: summary });
    renderPanel();
    await screen.findByRole("button", { name: "Run again" });
    fireEvent.change(screen.getByLabelText("Grouping column"), { target: { value: "cell_type" } });
    await waitFor(() =>
      expect(api.getLayoutsInfo).toHaveBeenLastCalledWith("ds", { groupBy: "cell_type", embedding: "X_umap" }),
    );
    expect(api.runLayouts).not.toHaveBeenCalled();
  });
});
