import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { MiloPlan } from "@/api/milo";
import type { DatasetInfo } from "@/api/types";
import { MILO } from "@/test/miloFixture";

const { planMilo, runMilo, miloCells, getDataset, downloadCsv } = vi.hoisted(() => ({
  planMilo: vi.fn(),
  runMilo: vi.fn(),
  miloCells: vi.fn(),
  getDataset: vi.fn(),
  downloadCsv: vi.fn(),
}));
vi.mock("@/api/milo", () => ({ planMilo, runMilo, miloCells }));
vi.mock("@/api/datasets", () => ({ getDataset }));
vi.mock("@/lib/csv", () => ({ downloadCsv }));
vi.mock("@/components/plots/Plot", () => ({
  default: (p: { onClick?: (e: unknown) => void }) => (
    <button data-testid="plot" onClick={() => p.onClick?.({ points: [{ customdata: 1 }] })} />
  ),
}));

import { MiloTest } from "./MiloTest";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

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
    { name: "batch", dtype: "category", n_unique: 2, values: ["b1", "b2"] },
  ],
  created_at: "2026-10-17T00:00:00Z",
};

const PLAN: MiloPlan = {
  condition_key: "stim",
  sample_key: "Donor",
  sample_key_detected: true,
  conditions: ["CTRL", "STIM"],
  reference: "CTRL",
  design: "paired",
  covariates: [],
  formula: "~ Donor + stim",
  n_cells: 13836,
  n_samples: { CTRL: 8, STIM: 8 },
  prop: 0.1,
  n_nhoods_estimate: 1384,
  estimate_seconds: 7.6,
  representation: "X_pca_harmony",
  warnings: ["The neighbour graph was built on an embedding batch-corrected for 'stim'."],
  notes: [],
  dropped_samples: [],
};

describe("MiloTest", () => {
  beforeEach(() => {
    planMilo.mockReset().mockResolvedValue(PLAN);
    runMilo.mockReset().mockImplementation(async (_id, _body, h) => {
      h?.onProgress?.({ step: "test", message: "Fitting" });
      return { ...MILO, warnings: PLAN.warnings };
    });
    miloCells.mockReset().mockResolvedValue({ cells: [3, 4, 5], n_cells: 3 });
    downloadCsv.mockReset();
    useSelectionStore.setState({ selectedCellIndices: null });
    useSettingsStore.setState({ embedding: "X_umap" });
    useViewStore.setState({ pendingAsk: null, activePanel: "compare" });
  });

  it("plans with the detected columns and shows the design and estimate before running", async () => {
    render(<MiloTest datasetId="kang" info={KANG} />);
    const plan = await screen.findByTestId("milo-plan");
    expect(planMilo).toHaveBeenLastCalledWith("kang", {
      condition_key: "stim", conditions: ["CTRL", "STIM"], label_key: "cluster",
    });
    expect(plan.textContent).toContain("8 CTRL and 8 STIM samples of Donor; ~ Donor + stim (paired)");
    expect(plan.textContent).toContain("Up to 1,384 neighbourhoods");
    expect(plan.textContent).toContain("about 8 s");
    expect(screen.getByLabelText("Milo sample column")).toHaveDisplayValue("Auto (Donor)");
    expect(screen.getByRole("alert").textContent).toContain("batch-corrected for 'stim'");
    expect(runMilo).not.toHaveBeenCalled();
  });

  it("sends the chosen design, covariate and sampled share", async () => {
    render(<MiloTest datasetId="kang" info={KANG} />);
    await screen.findByTestId("milo-plan");
    fireEvent.change(screen.getByLabelText("Milo design"), { target: { value: "unpaired" } });
    fireEvent.change(screen.getByLabelText("Milo covariate"), { target: { value: "batch" } });
    fireEvent.change(screen.getByLabelText("Milo sampled share"), { target: { value: "0.05" } });
    fireEvent.change(screen.getByLabelText("Milo reference"), { target: { value: "STIM" } });
    fireEvent.change(screen.getByLabelText("Milo tested condition"), { target: { value: "CTRL" } });
    await waitFor(() =>
      expect(planMilo).toHaveBeenLastCalledWith("kang", expect.objectContaining({
        paired: false, covariates: ["batch"], prop: 0.05, conditions: ["STIM", "CTRL"],
      })),
    );
    fireEvent.change(screen.getByLabelText("Milo tested condition"), { target: { value: "STIM" } });
    expect(await screen.findByText("Pick two different values to compare.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Run Milo/ })).toBeDisabled();
  });

  it("runs, shows the result, selects a clicked neighbourhood's cells and exports CSV", async () => {
    render(<MiloTest datasetId="kang" info={KANG} />);
    await screen.findByTestId("milo-plan");
    fireEvent.click(screen.getByRole("button", { name: /Run Milo/ }));
    const head = await screen.findByTestId("milo-headline");
    expect(head.textContent).toContain("2 of 4 neighbourhoods shift (spatial FDR < 0.1): 1 more abundant in STIM, 1 less");
    expect(runMilo).toHaveBeenCalledWith("kang", expect.objectContaining({ condition_key: "stim" }), expect.anything());
    expect(screen.getByLabelText("Milo graph layout")).toHaveValue("X_umap");
    expect(screen.getAllByTestId("plot")).toHaveLength(2);
    expect(screen.getByRole("cell", { name: "CD14 Mono" })).toBeInTheDocument();

    fireEvent.click(screen.getAllByTestId("plot")[0]!);
    await waitFor(() => expect(miloCells).toHaveBeenCalledWith("kang", [1]));
    expect((await screen.findByTestId("milo-picked")).textContent).toContain("Selected 3 cells (neighbourhood 1, CD14 Mono)");
    expect(useSelectionStore.getState().selectedCellIndices).toEqual(new Set([3, 4, 5]));
    fireEvent.click(screen.getByRole("button", { name: "Show on the scatter" }));
    expect(useViewStore.getState().activePanel).toBe("unified");

    fireEvent.click(screen.getByRole("button", { name: /Select up/ }));
    await waitFor(() => expect(miloCells).toHaveBeenLastCalledWith("kang", [0]));

    fireEvent.click(screen.getByRole("button", { name: /CSV/ }));
    const [name, headers] = downloadCsv.mock.calls[0]!;
    expect(name).toBe("milo_stim_STIM_vs_CTRL_kang.csv");
    expect(headers[0]).toBe("nhood");

    fireEvent.click(screen.getByRole("button", { name: /Ask/ }));
    expect(useViewStore.getState().pendingAsk).toContain("Milo differential abundance");
  });

  it("reports a plan the backend refuses and a failed run", async () => {
    planMilo.mockRejectedValue(new Error("Milo needs at least 2 per condition"));
    render(<MiloTest datasetId="kang" info={KANG} />);
    expect(await screen.findByText("Milo needs at least 2 per condition")).toBeInTheDocument();
    expect(planMilo).toHaveBeenCalledTimes(1);
    planMilo.mockResolvedValue(PLAN);
    fireEvent.change(screen.getByLabelText("Milo sample column"), { target: { value: "Donor" } });
    await screen.findByTestId("milo-plan");
    runMilo.mockRejectedValueOnce(new Error("Milo stopped before it finished."));
    fireEvent.click(screen.getByRole("button", { name: /Run Milo/ }));
    expect(await screen.findByText("Milo stopped before it finished.")).toBeInTheDocument();
  });
});
