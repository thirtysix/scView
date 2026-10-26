import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import { GENE_CCL2, MILODE, MILODE_PLAN, NHOOD_0 } from "@/test/milodeFixture";

const { planMiloDE, runMiloDE, miloDEGene, miloDENhood, miloDECells, downloadCsv } = vi.hoisted(() => ({
  planMiloDE: vi.fn(),
  runMiloDE: vi.fn(),
  miloDEGene: vi.fn(),
  miloDENhood: vi.fn(),
  miloDECells: vi.fn(),
  downloadCsv: vi.fn(),
}));
vi.mock("@/api/milode", () => ({ planMiloDE, runMiloDE, miloDEGene, miloDENhood, miloDECells }));
vi.mock("@/api/datasets", () => ({ getDataset: vi.fn() }));
vi.mock("@/lib/csv", () => ({ downloadCsv }));
vi.mock("@/components/plots/Plot", () => ({
  default: (p: { onClick?: (e: unknown) => void }) => (
    <button data-testid="plot" onClick={() => p.onClick?.({ points: [{ customdata: 0 }] })} />
  ),
}));

import { MiloDETest } from "./MiloDETest";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

const KANG: DatasetInfo = {
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 13836,
  n_genes: 14053,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "cluster", dtype: "category", n_unique: 3, values: ["B", "CD14 Mono", "pDC"] },
    { name: "Donor", dtype: "category", n_unique: 8 },
  ],
  created_at: "2026-10-17T00:00:00Z",
};

describe("MiloDETest", () => {
  beforeEach(() => {
    planMiloDE.mockReset().mockResolvedValue(MILODE_PLAN);
    runMiloDE.mockReset().mockImplementation(async (_id, _body, h) => {
      h?.onProgress?.({ step: "test", message: "Tested 20 of 42 neighbourhoods", done: 20, total: 42 });
      return MILODE;
    });
    miloDEGene.mockReset().mockResolvedValue(GENE_CCL2);
    miloDENhood.mockReset().mockResolvedValue(NHOOD_0);
    miloDECells.mockReset().mockResolvedValue({ cells: [3, 4, 5], n_cells: 3 });
    downloadCsv.mockReset();
    useSelectionStore.setState({ selectedCellIndices: null });
    useSettingsStore.setState({ embedding: "X_umap" });
  });

  it("plans with the detected columns and the miloDE defaults", async () => {
    render(<MiloDETest datasetId="kang" info={KANG} />);
    const plan = await screen.findByTestId("milode-plan");
    expect(plan.textContent).toContain("49 neighbourhoods");
    expect(plan.textContent).toContain("25-neighbour graph");
    expect(plan.textContent).toContain("42 testable");
    const body = planMiloDE.mock.calls.at(-1)![1];
    expect(body).toMatchObject({ condition_key: "stim", conditions: ["CTRL", "STIM"], k: 25, genes: "hvg" });
    expect(body.sample_key).toBeUndefined();
    fireEvent.change(screen.getByLabelText("Neighbourhood DE genes"), { target: { value: "expressed" } });
    await waitFor(() => expect(planMiloDE.mock.calls.at(-1)![1].genes).toBe("expressed"));
  });

  it("runs, lists genes, shows the first DE gene and selects a neighbourhood's cells", async () => {
    render(<MiloDETest datasetId="kang" info={KANG} />);
    await screen.findByTestId("milode-plan");
    fireEvent.click(screen.getByRole("button", { name: /Run neighbourhood DE/ }));
    const head = await screen.findByTestId("milode-headline");
    expect(head.textContent).toContain("2 of 3 genes DE in at least one of 2 tested neighbourhoods");
    await waitFor(() => expect(miloDEGene).toHaveBeenCalledWith("kang", "ISG15"));
    fireEvent.click(screen.getByRole("button", { name: "CCL2" }));
    await waitFor(() => expect(miloDEGene).toHaveBeenLastCalledWith("kang", "CCL2"));
    expect(screen.getByText("(few cells of one condition)")).toBeTruthy();
    expect(screen.getByText("not tested")).toBeTruthy();

    fireEvent.click(await screen.findByTestId("plot"));
    await waitFor(() => expect(miloDECells).toHaveBeenCalledWith("kang", [0]));
    expect(await screen.findByTestId("milode-picked")).toBeTruthy();
    expect(useSelectionStore.getState().selectedCellIndices).toEqual(new Set([3, 4, 5]));
    const detail = await screen.findByTestId("milode-nhood");
    expect(detail.textContent).toContain("2 genes DE (2 within)");

    fireEvent.click(screen.getByRole("button", { name: /CSV/ }));
    expect(downloadCsv).toHaveBeenCalledTimes(1);
    expect(downloadCsv.mock.calls[0]![2]).toHaveLength(3);
  });

  it("filters the gene table", async () => {
    render(<MiloDETest datasetId="kang" info={KANG} />);
    await screen.findByTestId("milode-plan");
    fireEvent.click(screen.getByRole("button", { name: /Run neighbourhood DE/ }));
    await screen.findByTestId("milode-headline");
    fireEvent.change(screen.getByLabelText("Filter genes"), { target: { value: "isg" } });
    const table = screen.getAllByRole("table")[0]!;
    expect(within(table).queryByText("CCL2")).toBeNull();
    expect(within(table).getByText("ISG15")).toBeTruthy();
  });

  it("can cancel a run", async () => {
    let signal: AbortSignal | undefined;
    runMiloDE.mockImplementation((_id, _body, h) => {
      signal = h?.signal;
      h?.onProgress?.({ step: "test", message: "Tested 20 of 42 neighbourhoods", done: 20, total: 42 });
      return new Promise(() => {});
    });
    render(<MiloDETest datasetId="kang" info={KANG} />);
    await screen.findByTestId("milode-plan");
    fireEvent.click(screen.getByRole("button", { name: /Run neighbourhood DE/ }));
    expect(await screen.findByText("Tested 20 of 42 neighbourhoods")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Cancel/ }));
    expect(signal?.aborted).toBe(true);
    expect(await screen.findByText(/Cancelled/)).toBeTruthy();
  });

  it("shows why a plan cannot run", async () => {
    planMiloDE.mockRejectedValue(new Error("This dataset has no neighbour graph"));
    render(<MiloDETest datasetId="kang" info={KANG} />);
    expect(await screen.findByText(/no neighbour graph/)).toBeTruthy();
    expect((screen.getByRole("button", { name: /Run neighbourhood DE/ }) as HTMLButtonElement).disabled).toBe(true);
  });
});
