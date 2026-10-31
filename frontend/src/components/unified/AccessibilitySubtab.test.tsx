import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { AtacDa, AtacState } from "@/api/atac";
import type { DatasetInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getAtac: vi.fn(),
  getAtacQc: vi.fn(),
  getAtacActivity: vi.fn(),
  getLatestDa: vi.fn(),
  getDaPage: vi.fn(),
  attachAtacMetrics: vi.fn(),
  runAtac: vi.fn(),
  runAtacWnn: vi.fn(),
  runAtacDa: vi.fn(),
}));
vi.mock("@/api/atac", async (orig) => ({ ...(await orig<typeof import("@/api/atac")>()), ...api }));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => vi.fn().mockResolvedValue({}) }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string }[] }) => <div data-testid="plot">{data.map((t) => t.type).join(";")}</div>,
}));
vi.mock("@/components/common/GeneSearch", () => ({
  GeneSearch: ({ onSelect }: { onSelect: (g: string) => void }) => <button onClick={() => onSelect("CD3D")}>pick CD3D</button>,
}));

import { AccessibilitySubtab } from "./AccessibilitySubtab";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const DATASET = {
  id: "d1",
  n_cells: 100,
  available_embeddings: ["X_umap", "X_umap_atac", "X_umap_wnn_atac"],
  obs_columns: [
    { name: "cell_type", dtype: "category", n_unique: 3 },
    { name: "atac_leiden", dtype: "category", n_unique: 5 },
    { name: "wnn_atac_leiden", dtype: "category", n_unique: 5 },
    { name: "wnn_atac_rna_weight", dtype: "float32", n_unique: 100 },
  ],
  atac: { n_peaks: 1000, genome: "GRCh38", computed: ["peak counts"] },
} as unknown as DatasetInfo;

const BASE: AtacState = {
  available: true,
  reason: null,
  n_cells: 100,
  info: { n_peaks: 1000, nnz: 2_000_000, genome: "GRCh38", source: "pbmc.h5", computed: ["peak counts"] },
  lsi: null,
  wnn: null,
  activity: null,
  plan: { n_cells: 100, n_peaks: 1000, nnz: 2_000_000, genome: "GRCh38", n_lsi: 50, dims_to: 30, drop_first: "auto",
          percentile: 5, resolution: 1, activity: true, estimate_seconds: 60, estimate_bytes: 1024 ** 3, estimate_disk_bytes: 1024 ** 3 },
  plan_reason: null,
  wnn_plan: null,
  wnn_reason: "Run ATAC analysis first: WNN needs the LSI components of the Accessibility tab.",
  qc_columns: ["atac_total_counts"],
  running: null,
};

const LSI = {
  run_id: "r1", created_at: "", sdev: [1, 2], variance_ratio: [0.5], depth_r: [-0.92, 0.1, 0.2], depth_limit: 0.75,
  dropped_first: true, dims_used: [2, 3], n_selected: 900, n_clusters: 7, seconds: 10, method: "Signac's LSI.", stale: false,
  clusters_column: "atac_leiden",
};

const DA: AtacDa = {
  id: "abc", column: "cell_type", groups: ["T", "B"], n_tested: { T: 500, B: 400 }, n_significant: { T: 20, B: 2 },
  rows: {
    T: [{ group: "T", peak: "chr1:10-20", z: 9, log2fc: 2, pct_in: 0.5, pct_rest: 0.1, pval: 1e-20, pval_adj: 1e-18,
          nearest_gene: "CD3D", distance_to_tss: -300, peak_class: "promoter" }],
    B: [],
  },
  method: "Wilcoxon.", kept_per_group: 5000, created_at: "",
};

function renderTab(onActivity = vi.fn()) {
  render(<AccessibilitySubtab datasetId="d1" groupByColumn="cell_type" setGroupByColumn={vi.fn()} activeGene={null} onActivity={onActivity} />);
  return onActivity;
}

beforeEach(() => {
  vi.clearAllMocks();
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "d1" });
  useAtacRunStore.setState({ datasetId: null, kind: null, running: false, stage: null, progress: null, warnings: [], error: null, da: null, version: 0 });
  api.getAtac.mockResolvedValue(BASE);
  api.getAtacQc.mockResolvedValue({ columns: [{ key: "atac_total_counts", label: "Peak counts per cell", values: [1, 2, 3] }],
                                    n_cells: 3, groups: ["a", "b", "a"], group_order: ["a", "b"], column: "cell_type" });
  api.getLatestDa.mockResolvedValue(null);
});

describe("AccessibilitySubtab", () => {
  it("says why when a dataset has no ATAC", async () => {
    api.getAtac.mockResolvedValue({ available: false, reason: "This dataset has no ATAC peaks." });
    renderTab();
    expect(await screen.findByText("This dataset has no ATAC peaks.")).toBeTruthy();
  });

  it("shows the plan before a run, and a run goes through the store", async () => {
    api.runAtac.mockResolvedValue({ ...BASE, lsi: LSI });
    renderTab();
    const plan = await screen.findByTestId("atac-plan");
    expect(plan.textContent).toContain("100 cells");
    expect(plan.textContent).toContain("1.0 GB");
    fireEvent.click(screen.getByRole("button", { name: "Run ATAC analysis" }));
    await waitFor(() => expect(api.runAtac).toHaveBeenCalledWith("d1", {}, expect.anything()));
    await waitFor(() => expect(useAtacRunStore.getState().version).toBe(1));
  });

  it("draws QC with a violin per group on a category axis", async () => {
    renderTab();
    const card = await screen.findByTestId("atac-qc-atac_total_counts");
    await waitFor(() => expect(card.textContent).toContain("histogram"));
    expect(card.textContent).toContain("violin;violin");
    expect(screen.getByRole("button", { name: "Attach per-barcode metrics" })).toBeTruthy();
  });

  it("shows the depth choice and its bar after a run, and the embedding and clusters", async () => {
    api.getAtac.mockResolvedValue({ ...BASE, lsi: LSI, activity: { available: false, reason: "Gene positions are off." } });
    renderTab();
    expect((await screen.findByTestId("atac-depth-choice")).textContent).toContain("r = -0.92");
    expect(screen.getByTestId("atac-activity-reason").textContent).toBe("Gene positions are off.");
    fireEvent.click(screen.getByRole("button", { name: "Show ATAC UMAP" }));
    expect(useSettingsStore.getState().embedding).toBe("X_umap_atac");
    fireEvent.click(screen.getByRole("button", { name: "Colour by ATAC clusters" }));
    expect(useSettingsStore.getState().colorBy).toBe("atac_leiden");
  });

  it("says WNN needs LSI first, then offers it and shows the weight by group", async () => {
    renderTab();
    expect(await screen.findByText(/Run ATAC analysis first/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Run WNN" })).toBeNull();
  });

  it("lists RNA weight per group, highest first", async () => {
    api.getAtac.mockResolvedValue({
      ...BASE, lsi: LSI, wnn_reason: null,
      wnn_plan: { n_cells: 100, rep: "X_pca", d_rna: 50, lsi_dims: [2, 50], notes: [], estimate_seconds: 60, estimate_bytes: 1024 ** 3 },
      wnn: { run_id: "w", n_clusters: 5, mean_rna_weight: 0.55, method: "WNN.", stale: false, clusters_column: "wnn_atac_leiden",
             plan: { n_cells: 100, rep: "X_pca", d_rna: 50, lsi_dims: [2, 50], notes: [], estimate_seconds: 60, estimate_bytes: 1 },
             by_group: { column: "cell_type", rows: [
               { group: "B", n_cells: 40, mean_rna_weight: 0.3, median_rna_weight: 0.3 },
               { group: "T", n_cells: 60, mean_rna_weight: 0.7, median_rna_weight: 0.7 }] } },
    });
    renderTab();
    const table = await screen.findByTestId("atac-wnn-by-group");
    const rows = table.querySelectorAll("tbody tr");
    expect(rows[0]?.textContent).toContain("T");
    fireEvent.click(screen.getByRole("button", { name: "Run WNN again" }));
    await waitFor(() => expect(api.runAtacWnn).toHaveBeenCalled());
  });

  it("draws a gene's activity on the plot and against its RNA, saying it is peak-based", async () => {
    api.getAtac.mockResolvedValue({ ...BASE, lsi: LSI, activity: { available: true, n_genes: 30000, n_genes_with_peaks: 25000, source: "GENCODE v32 basic",
                                                                     caption: "Peak-based gene activity: counts in called peaks." } });
    const activity = { gene: "CD3D", caption: "c", n_cells: 3, n_nonzero: 2, values: [0, 1, 2], violin: null, column: null, r: 0.61,
                       rna: { rna: [0, 1, 2], activity: [0, 1, 2], n_shown: 3 } };
    api.getAtacActivity.mockResolvedValue(activity);
    const onActivity = renderTab();
    expect(await screen.findByText(/Peak-based gene activity/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "pick CD3D" }));
    await waitFor(() => expect(onActivity).toHaveBeenCalledWith(activity));
    const panel = await screen.findByTestId("atac-activity-vs-rna");
    expect(panel.textContent).toContain("r = 0.61");
    expect(panel.textContent).toContain("scattergl");
  });

  it("finds differential peaks and lists nearest gene, distance and class", async () => {
    api.runAtacDa.mockResolvedValue(DA);
    renderTab();
    await screen.findByTestId("atac-plan");
    fireEvent.click(screen.getByRole("button", { name: "Find differential peaks" }));
    await waitFor(() => expect(api.runAtacDa).toHaveBeenCalledWith("d1", { column: "cell_type" }, expect.anything()));
    const table = await screen.findByTestId("atac-da");
    expect(table.textContent).toContain("chr1:10-20");
    expect(table.textContent).toContain("CD3D");
    expect(table.textContent).toContain("-300 b");
    expect(table.textContent).toContain("promoter");
    expect(table.textContent).toContain("T (20 at FDR 5%)");
    expect(screen.getByRole("link", { name: "CSV" }).getAttribute("href")).toContain("/atac/differential/abc/csv");
  });

  it("cancels a running stream: nothing saved is said", async () => {
    api.runAtac.mockImplementation((_d: string, _b: unknown, h: { signal: AbortSignal }) => new Promise((_res, rej) => {
      h.signal.addEventListener("abort", () => rej(new Error("aborted")));
    }));
    renderTab();
    await screen.findByTestId("atac-plan");
    fireEvent.click(screen.getByRole("button", { name: "Run ATAC analysis" }));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(await screen.findByText("Cancelled: nothing was saved.")).toBeTruthy();
  });
});
