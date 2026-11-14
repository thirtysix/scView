import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({
  getLatestDa: vi.fn(),
  getDaEstimate: vi.fn(),
  getDaPage: vi.fn(),
  daCsvUrl: vi.fn(() => "/csv"),
  runAtacDa: vi.fn(),
}));
vi.mock("@/api/atac", async (orig) => ({ ...(await orig<typeof import("@/api/atac")>()), ...api }));

import type { AtacDa, AtacState } from "@/api/atac";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { AtacDifferentialCard } from "./AtacDifferentialCard";

const STATE = { available: true, running: null } as unknown as AtacState;

const PAIR: AtacDa = {
  id: "r1",
  column: "celltype",
  groups: ["CD4 naive"],
  vs: "CD14 mono",
  n_tested: { "CD4 naive": 900 },
  n_significant: { "CD4 naive": 310 },
  rows: {
    "CD4 naive": [{ peak: "chr17-82126458-82127345", nearest_gene: "CCDC57", distance_to_tss: 0, peak_class: "gene body",
                    z: 12, log2fc: 4.2, pct_in: 0.7, pct_rest: 0.1, pval: 1e-30, pval_adj: 1e-27 }],
  } as unknown as AtacDa["rows"],
  method: "Logistic regression ... CD4 naive vs CD14 mono (the cells of no other group used) ...",
  test: "lr",
  n_fit: null,
  kept_per_group: 5000,
  created_at: "2026-11-14",
};

beforeEach(() => {
  vi.clearAllMocks();
  useAtacRunStore.setState({ datasetId: null, kind: null, running: false, da: null, error: null, warnings: [], progress: null });
  api.getLatestDa.mockResolvedValue(null);
  api.getDaEstimate.mockImplementation((_d: string, column: string, test: string, group = "", vs = "") =>
    Promise.resolve({ column, test, n_groups: vs ? 1 : 3, n_cells: vs ? 800 : 1200, estimate_seconds: 60,
                      estimate_bytes: 1024 ** 3, groups: ["B", "CD14 mono", "CD4 naive"], group: vs ? group : null, vs: vs || null }));
});

describe("AtacDifferentialCard", () => {
  it("runs one group against another and says what that costs", async () => {
    const runDa = vi.fn(() => Promise.resolve());
    useAtacRunStore.setState({ runDa });
    render(<AtacDifferentialCard datasetId="d" state={STATE} columns={["celltype"]} defaultColumn="celltype" />);
    expect((await screen.findByTestId("atac-da-estimate")).textContent).toContain("for the 3 groups of this column");
    fireEvent.change(screen.getByLabelText("Compare"), { target: { value: "pair" } });
    fireEvent.change(screen.getByLabelText("Group"), { target: { value: "CD4 naive" } });
    fireEvent.change(screen.getByLabelText("Against"), { target: { value: "CD14 mono" } });
    await waitFor(() => expect(screen.getByTestId("atac-da-estimate").textContent).toContain("for CD4 naive vs CD14 mono (800 cells)"));
    expect(api.getDaEstimate).toHaveBeenLastCalledWith("d", "celltype", "lr", "CD4 naive", "CD14 mono");
    fireEvent.click(screen.getByRole("button", { name: "Find differential peaks" }));
    expect(runDa).toHaveBeenCalledWith("d", { column: "celltype", test: "lr", groups: ["CD4 naive"], vs: "CD14 mono", fold_change: "signac" });
  });

  it("each group vs the rest sends no pair", async () => {
    const runDa = vi.fn(() => Promise.resolve());
    useAtacRunStore.setState({ runDa });
    render(<AtacDifferentialCard datasetId="d" state={STATE} columns={["celltype"]} defaultColumn="celltype" />);
    await screen.findByTestId("atac-da-estimate");
    fireEvent.click(screen.getByRole("button", { name: "Find differential peaks" }));
    expect(runDa).toHaveBeenCalledWith("d", { column: "celltype", test: "lr", fold_change: "signac" });
  });

  it("sends the older fold change when chosen", async () => {
    const runDa = vi.fn(() => Promise.resolve());
    useAtacRunStore.setState({ runDa });
    render(<AtacDifferentialCard datasetId="d" state={STATE} columns={["celltype"]} defaultColumn="celltype" />);
    await screen.findByTestId("atac-da-estimate");
    fireEvent.change(screen.getByLabelText("Fold change"), { target: { value: "tfidf" } });
    fireEvent.click(screen.getByRole("button", { name: "Find differential peaks" }));
    expect(runDa).toHaveBeenCalledWith("d", { column: "celltype", test: "lr", fold_change: "tfidf" });
  });

  it("a result names its fold change; one saved before v0.79 reads as the older formula", async () => {
    api.getLatestDa.mockResolvedValue({ ...PAIR, fold_change: "signac" });
    const { unmount } = render(<AtacDifferentialCard datasetId="d" state={STATE} columns={["celltype"]} defaultColumn="celltype" />);
    let table = await screen.findByTestId("atac-da");
    expect(table.textContent).toContain("Signac's log2FC");
    expect(screen.getByText("log2FC", { selector: "th" }).getAttribute("title")).toContain("Signac's FoldChange for peaks");
    unmount();
    useAtacRunStore.setState({ da: null });
    api.getLatestDa.mockResolvedValue(PAIR);
    render(<AtacDifferentialCard datasetId="d" state={STATE} columns={["celltype"]} defaultColumn="celltype" />);
    table = await screen.findByTestId("atac-da");
    expect(table.textContent).toContain("log2FC of mean TF-IDF + 1");
    expect(screen.getByText("log2FC", { selector: "th" }).getAttribute("title")).toContain("before v0.79");
    expect((screen.getByLabelText("Fold change") as HTMLSelectElement).value).toBe("tfidf");
  });

  it("a pair's table names the other group, not the rest", async () => {
    api.getLatestDa.mockResolvedValue(PAIR);
    render(<AtacDifferentialCard datasetId="d" state={STATE} columns={["celltype"]} defaultColumn="celltype" />);
    const table = await screen.findByTestId("atac-da");
    expect(table.textContent).toContain("CD4 naive against CD14 mono, by");
    expect(table.textContent).not.toContain("% rest");
    expect((screen.getByLabelText("Compare") as HTMLSelectElement).value).toBe("pair");
  });
});
