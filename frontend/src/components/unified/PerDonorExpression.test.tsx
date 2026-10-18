import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { PerDonorExpression as Result } from "@/api/donors";
import type { DatasetInfo } from "@/api/types";

const { getPerDonorExpression } = vi.hoisted(() => ({ getPerDonorExpression: vi.fn() }));
vi.mock("@/api/donors", () => ({ getPerDonorExpression }));
vi.mock("@/components/plots/Plot", () => ({ default: () => <div data-testid="plot" /> }));

import { PerDonorExpression } from "./PerDonorExpression";
import { useDatasetStore } from "@/stores/datasetStore";

const KANG = {
  id: "kang",
  name: "Kang",
  n_cells: 13836,
  obs_columns: [
    { name: "cluster", dtype: "category", n_unique: 2, values: ["B", "CD14 Mono"] },
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "Donor", dtype: "category", n_unique: 8 },
  ],
} as unknown as DatasetInfo;

const pts = (cond: string, n: number, base: number) =>
  Array.from({ length: n }, (_, i) => ({ group: "B", condition: cond, sample: `d${i}`, n_cells: 40, value: base + i / 10 }));

const RESULT: Result = {
  gene: "ISG15", group_key: "cluster", sample_key: "Donor", sample_key_detected: true,
  condition_key: "stim", method: "mean", value_label: "mean log-normalised expression per donor",
  source: "log-normalised .raw", min_cells: 10, groups: ["B"], conditions: ["CTRL", "STIM"],
  points: [...pts("CTRL", 8, 0.5), ...pts("STIM", 8, 4.3)],
  n_donors: { CTRL: 8, STIM: 8 }, n_dropped: 3, warnings: [], message: null,
};

describe("PerDonorExpression", () => {
  beforeEach(() => {
    getPerDonorExpression.mockReset().mockResolvedValue(RESULT);
    useDatasetStore.setState({ currentDatasetId: "kang", currentDataset: KANG });
  });

  it("asks for the gene by group, split by the suggested condition, donors detected", async () => {
    render(<PerDonorExpression datasetId="kang" gene="ISG15" groupColumn="cluster" />);
    const caption = await screen.findByTestId("donor-caption");
    expect(getPerDonorExpression).toHaveBeenCalledWith("kang", {
      gene: "ISG15", groupby: "cluster", condition: "stim", method: "mean",
    });
    expect(caption.textContent).toContain("One point per Donor (n = 8: CTRL 8, STIM 8)");
    expect(caption.textContent).toContain("mean log-normalised expression per donor (log-normalised .raw)");
    expect(caption.textContent).toContain("3 donor-group samples under 10 cells left out");
    expect(screen.getByRole("option", { name: "Auto (Donor)" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("re-asks with log2 CPM, no split or a chosen donor column", async () => {
    render(<PerDonorExpression datasetId="kang" gene="ISG15" groupColumn="cluster" />);
    await screen.findByTestId("donor-caption");
    fireEvent.change(screen.getByLabelText("Pseudobulk value"), { target: { value: "cpm" } });
    await waitFor(() =>
      expect(getPerDonorExpression).toHaveBeenLastCalledWith("kang", expect.objectContaining({ method: "cpm" })),
    );
    fireEvent.change(screen.getByLabelText("Condition column"), { target: { value: "" } });
    await waitFor(() =>
      expect(getPerDonorExpression).toHaveBeenLastCalledWith("kang", { gene: "ISG15", groupby: "cluster", method: "cpm" }),
    );
    fireEvent.change(screen.getByLabelText("Donor column"), { target: { value: "Donor" } });
    await waitFor(() =>
      expect(getPerDonorExpression).toHaveBeenLastCalledWith("kang", expect.objectContaining({ sample: "Donor" })),
    );
  });

  it("cautions when a condition has fewer than three donors", async () => {
    getPerDonorExpression.mockResolvedValue({
      ...RESULT,
      points: [...pts("CTRL", 2, 0.5), ...pts("STIM", 8, 4.3)],
      n_donors: { CTRL: 2, STIM: 8 },
    });
    render(<PerDonorExpression datasetId="kang" gene="ISG15" groupColumn="cluster" />);
    expect((await screen.findByRole("alert")).textContent).toContain("Fewer than 3 donors (CTRL: 2)");
  });

  it("says a donor column is needed when the dataset has none (paul15)", async () => {
    getPerDonorExpression.mockResolvedValue({
      ...RESULT, sample_key: null, sample_key_detected: false, condition_key: null, points: [],
      groups: [], conditions: [], n_donors: {}, source: null,
      message: "Per-donor values need a sample or donor column (one value per person or sample), and none was found in this dataset.",
    });
    render(<PerDonorExpression datasetId="paul" gene="Hba-a2" groupColumn="cluster" />);
    expect((await screen.findByTestId("needs-sample")).textContent).toContain("need a sample or donor column");
    expect(screen.queryByTestId("plot")).toBeNull();
  });

  it("shows a request error", async () => {
    getPerDonorExpression.mockRejectedValue(new Error("Gene 'NOPE' not found."));
    render(<PerDonorExpression datasetId="kang" gene="NOPE" groupColumn="cluster" />);
    expect(await screen.findByText("Gene 'NOPE' not found.")).toBeInTheDocument();
  });

  it("explains why there is nothing to show without a gene", () => {
    render(<PerDonorExpression datasetId="kang" gene={null} noGeneReason="Pick a gene." groupColumn="cluster" />);
    expect(screen.getByText("Pick a gene.")).toBeInTheDocument();
    expect(getPerDonorExpression).not.toHaveBeenCalled();
  });
});
