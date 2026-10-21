import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { CommunicationOptions } from "@/api/communication";
import type { DatasetInfo } from "@/api/types";
import { fakeResult } from "@/test/communicationFixture";

const { getCommunicationOptions, runCommunication } = vi.hoisted(() => ({
  getCommunicationOptions: vi.fn(),
  runCommunication: vi.fn(),
}));
vi.mock("@/api/communication", async (orig) => ({
  ...(await orig<typeof import("@/api/communication")>()),
  getCommunicationOptions,
  runCommunication,
}));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; mode?: string; y?: unknown[] }[] }) => (
    <div data-testid="plot">{data.map((t) => `${t.type}${t.mode ? `/${t.mode}` : ""}:${(t.y ?? []).length}`).join(";")}</div>
  ),
}));

import { UnifiedCommunicationSubtab } from "./UnifiedCommunicationSubtab";
import { useCommunicationStore } from "@/stores/communicationStore";
import { useDatasetStore } from "@/stores/datasetStore";

const OPTIONS: CommunicationOptions = {
  conditions: [{ column: "stim", values: ["CTRL", "STIM"] }],
  species: "human",
  species_evidence: "Ensembl ids",
  resource: "consensus",
  max_groups: 60,
  max_perms: 5000,
  default_expr_prop: 0.1,
  default_n_perms: 1000,
  min_cells: 5,
  n_cells: 300,
  n_groups: 3,
  estimate_seconds: 2.4,
};

describe("UnifiedCommunicationSubtab", () => {
  beforeEach(() => {
    useCommunicationStore.getState().reset();
    getCommunicationOptions.mockReset().mockResolvedValue(OPTIONS);
    runCommunication.mockReset().mockImplementation(async (_id, _body, h) => {
      h.onStart?.({ n_cells: 300, n_groups: 3, resource: "consensus", resource_label: "Consensus (human)",
                    species: "human", species_evidence: "", n_perms: 1000, estimate_seconds: 2 });
      h.onProgress?.(500, 1000);
      return fakeResult();
    });
    useDatasetStore.setState({
      currentDatasetId: "d1",
      currentDataset: {
        id: "d1", name: "d1", filename: "", n_cells: 300, n_genes: 900, status: "ready", available_embeddings: [],
        created_at: "",
        obs_columns: [
          { name: "cell", dtype: "category", n_unique: 3, values: ["B", "NK", "Mono"] },
          { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
        ],
      } as DatasetInfo,
    });
  });

  it("runs on one condition, then shows the table, plots and CSV link", async () => {
    render(<UnifiedCommunicationSubtab groupByColumn="cell" />);
    await waitFor(() => screen.getByText(/about 2 s with 1000 permutations/));
    fireEvent.change(screen.getByDisplayValue("All cells"), { target: { value: "stim\u0000STIM" } });
    await waitFor(() =>
      expect(getCommunicationOptions).toHaveBeenLastCalledWith("d1", {
        groupby: "cell", subset_column: "stim", subset_value: "STIM",
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: /Run/ }));
    await waitFor(() => screen.getByText("CXCL10"));
    expect(runCommunication).toHaveBeenCalledWith(
      "d1",
      { groupby: "cell", subset_column: "stim", subset_value: "STIM", expr_prop: 0.1 },
      expect.anything(),
    );
    expect(screen.getByText(/3 interactions scored between 3 cell groups \(stim = STIM, 300 cells\)/)).toBeTruthy();
    expect(screen.getByText(/Left out \(under 5 cells\): Eryth/)).toBeTruthy();
    expect(screen.getByTestId("plot").textContent).toBe("scatter/markers:3");
    expect(screen.getByText("CSV").closest("a")!.getAttribute("href")).toBe(
      "/api/v1/datasets/d1/communication/results/r1.csv",
    );

    fireEvent.click(screen.getByRole("button", { name: "Circle" }));
    // Two edges (NK -> Mono, Mono -> NK), then the nodes.
    expect(screen.getByTestId("plot").textContent).toBe("scatter/lines:17;scatter/lines:17;scatter/text+markers:3");
    fireEvent.click(screen.getByRole("button", { name: "Roles" }));
    expect(screen.getByTestId("plot").textContent).toBe("bar:3;bar:3");

    fireEvent.change(screen.getByLabelText("Sender"), { target: { value: "NK" } });
    expect(screen.getByText("1 rows")).toBeTruthy();
    expect(screen.queryByText("CXCL10")).toBeNull();
  });

  it("says why a run failed", async () => {
    runCommunication.mockRejectedValueOnce(new Error("is it the right species?"));
    render(<UnifiedCommunicationSubtab groupByColumn="cell" />);
    await waitFor(() => screen.getByText(/about 2 s/));
    fireEvent.click(screen.getByRole("button", { name: /Run/ }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/right species/));
  });
});

describe("UnifiedCommunicationSubtab table cap", () => {
  it("names the sort column and the total when more than 200 rows match", async () => {
    const n = 230;
    const col = <T,>(v: T) => Array.from({ length: n }, () => v);
    const seq = (f: (i: number) => number) => Array.from({ length: n }, (_, i) => f(i));
    runCommunication.mockReset().mockResolvedValue(
      fakeResult({
        n_rows: n, n_interactions: n, rows_shown: n,
        columns: {
          source: col("Mono"), target: col("NK"),
          ligand_complex: seq((i) => i).map((i) => `L${i}`), receptor_complex: seq((i) => i).map((i) => `R${i}`),
          ligand: seq((i) => i).map((i) => `L${i}`), receptor: seq((i) => i).map((i) => `R${i}`),
          magnitude_rank: seq((i) => (i + 1) / n), specificity_rank: seq((i) => (i + 1) / n),
          lr_means: seq((i) => n - i), cellphone_pvals: col(0.01), expr_prod: col(1), scaled_weight: col(1),
          lr_logfc: col(1), spec_weight: col(0.1), lrscore: col(0.5), ligand_props: col(0.5), receptor_props: col(0.5),
        },
      }),
    );
    render(<UnifiedCommunicationSubtab groupByColumn="cell" />);
    await waitFor(() => screen.getByText(/about 2 s with 1000 permutations/));
    fireEvent.click(screen.getByRole("button", { name: /Run/ }));
    await screen.findByText("L0");
    expect(screen.getByText("Top 200 of 230 rows by Magn.")).toBeTruthy();
    expect(screen.queryByText("L229")).toBeNull();
    fireEvent.click(screen.getByText("LR mean"));
    expect(screen.getByText("Top 200 of 230 rows by LR mean")).toBeTruthy();
  });
});
