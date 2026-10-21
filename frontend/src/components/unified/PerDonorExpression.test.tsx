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
  design: { paired: false, n_pairs: 0, conditions: ["CTRL", "STIM"] }, paired_tests: {},
};
const PAIRED: Result = {
  ...RESULT,
  design: { paired: true, n_pairs: 8, conditions: ["CTRL", "STIM"] },
  paired_tests: { B: { n_pairs: 8, mean_diff: 3.7694, wilcoxon_p: 0.0078125, ttest_p: 1.28e-9, caution: null } },
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

  it("draws a paired design as lines per donor with both tests, boxes a toggle away", async () => {
    getPerDonorExpression.mockResolvedValue(PAIRED);
    render(<PerDonorExpression datasetId="kang" gene="ISG15" groupColumn="cluster" />);
    const caption = await screen.findByTestId("donor-caption");
    expect(caption.textContent).toContain("One line per Donor (n = 8 donors, each in CTRL and STIM)");
    expect(caption.textContent).toContain("heavier marker is the mean per condition");
    const table = screen.getByTestId("paired-tests");
    expect(table.textContent).toContain("Wilcoxon signed-rank p");
    expect(table.textContent).toContain("Paired t p");
    expect(table.textContent).toContain("+3.77 log-norm units");
    expect(table.textContent).toContain("0.0078");
    expect(table.textContent).toContain("1.28e-9");
    expect(screen.queryByTestId("paired-caution")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Boxes" }));
    expect(screen.getByTestId("donor-caption").textContent).toContain("One point per Donor (n = 8: CTRL 8, STIM 8)");
    expect(screen.queryByTestId("paired-tests")).toBeNull();
    expect(screen.getByRole("button", { name: "Panel per group" })).toBeInTheDocument();
  });

  it("marks few-pair tests as descriptive and names the unit for log2 CPM", async () => {
    getPerDonorExpression.mockResolvedValue({
      ...PAIRED, method: "cpm", value_label: "log2(CPM + 1) of the donor's summed counts",
      points: [...pts("CTRL", 3, 0.5), ...pts("STIM", 3, 4.3)], n_donors: { CTRL: 3, STIM: 3 },
      design: { paired: true, n_pairs: 3, conditions: ["CTRL", "STIM"] },
      paired_tests: { B: { n_pairs: 3, mean_diff: 3.8, wilcoxon_p: 0.25, ttest_p: 0.001, caution: "3 pairs: read these p-values as descriptive." } },
    });
    render(<PerDonorExpression datasetId="kang" gene="ISG15" groupColumn="cluster" />);
    await screen.findByTestId("donor-caption");
    fireEvent.change(screen.getByLabelText("Pseudobulk value"), { target: { value: "cpm" } });
    await waitFor(() => expect(screen.getByTestId("paired-tests").textContent).toContain("+3.80 log2 CPM"));
    expect(screen.getByTestId("paired-tests").textContent).toContain("0.2500*");
    expect(screen.getByTestId("paired-caution").textContent).toContain("Under 5 pairs: read as descriptive");
  });

  it("draws three timepoints as trajectories without a test", async () => {
    const times = ["T0", "T1", "T2"];
    getPerDonorExpression.mockResolvedValue({
      ...PAIRED, conditions: times, n_donors: { T0: 4, T1: 4, T2: 4 },
      points: times.flatMap((t, i) => pts(t, 4, i)),
      design: { paired: true, n_pairs: 4, conditions: times }, paired_tests: {},
    });
    render(<PerDonorExpression datasetId="kang" gene="ISG15" groupColumn="cluster" />);
    const caption = await screen.findByTestId("donor-caption");
    expect(caption.textContent).toContain("n = 4 donors, each in T0, T1 and T2");
    expect(caption.textContent).toContain("3 conditions: the lines follow each Donor across them, and there is no test");
    expect(screen.queryByTestId("paired-tests")).toBeNull();
  });

  it("explains why there is nothing to show without a gene", () => {
    render(<PerDonorExpression datasetId="kang" gene={null} noGeneReason="Pick a gene." groupColumn="cluster" />);
    expect(screen.getByText("Pick a gene.")).toBeInTheDocument();
    expect(getPerDonorExpression).not.toHaveBeenCalled();
  });
});
