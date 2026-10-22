import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { CompositionPlan, DEPlan, PilotInfo } from "@/api/planner";
import type { DatasetInfo } from "@/api/types";

const { planPilot, runCompositionPlan, runDEPlan, getDataset, plots } = vi.hoisted(() => ({
  planPilot: vi.fn(),
  runCompositionPlan: vi.fn(),
  runDEPlan: vi.fn(),
  getDataset: vi.fn(),
  plots: [] as { data: { name: string; line: { dash: string } }[] }[],
}));
vi.mock("@/api/planner", () => ({ planPilot, runCompositionPlan, runDEPlan }));
vi.mock("@/api/datasets", () => ({ getDataset }));
vi.mock("@/components/plots/Plot", () => ({
  default: (p: { data: { name: string; line: { dash: string } }[] }) => {
    plots.push(p);
    return <div data-testid="plot" />;
  },
}));

import { StudyPlanner } from "./StudyPlanner";

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

const PILOT: PilotInfo = {
  group_key: "cluster", condition_key: "stim", sample_key: "Donor", sample_key_detected: true,
  conditions: ["CTRL", "STIM"], design: "paired", replicated: true, n_samples: { CTRL: 8, STIM: 8 },
  n_donors: 8, cells_per_sample: { median: 814, min: 279, max: 1622 },
  groups: [
    { group: "B", share: 0.066, n_cells: 998 },
    { group: "CD14 Mono", share: 0.35, n_cells: 4225 },
    { group: "pDC", share: 0.0057, n_cells: 108 },
  ],
  suggested: ["CD14 Mono", "pDC"], message: null, assumed_sd_default: 0.35,
  assumed_sd_source: "median spread of Kang", dropped_samples: [],
};

const DONORS = [3, 4, 6, 8, 12, 16];
const curve = (group: string, cells: number, power: number[], pessimistic = false) =>
  DONORS.map((donors, i) => ({ group, fold_change: 2, donors, cells, power: power[i]!, se: 0.03, power_raw: 0.9, pessimistic }));

const PLAN: CompositionPlan = {
  kind: "composition", group_key: "cluster", condition_key: "stim", sample_key: "Donor",
  sample_key_detected: true, conditions: ["CTRL", "STIM"], reference: "CTRL", source: "pilot",
  assumed_sd: null, assumed_sd_default: 0.35, assumed_sd_source: "Kang",
  pilot: { design: "paired", n_samples: { CTRL: 8, STIM: 8 }, n_donors: 8, df: 7,
    cells_per_sample: { median: 814, min: 279, max: 1622 }, dropped_samples: [] },
  design: "paired", can_pair: true, alpha: 0.05, target_power: 0.8, reps: 100, donors: DONORS,
  cells: [500, 2000], fold_changes: [2], middle_cells: 2000,
  types: [{ group: "CD14 Mono", share: 0.35, share_tested: 0.3, n_cells: 4225, sd_within: 0.2, sd_donor: 0.4,
    sd_total: 0.45, sd_used: 0.2, sd_low: 0.1, sd_high: 0.4, sd_low_used: 0.1, sd_upper_used: 0.4, targeted: true }],
  points: [
    ...curve("CD14 Mono", 500, [0.3, 0.5, 0.7, 0.85, 0.95, 1]),
    ...curve("CD14 Mono", 2000, [0.4, 0.6, 0.86, 0.95, 1, 1]),
    ...curve("CD14 Mono", 2000, [0.1, 0.2, 0.4, 0.6, 0.85, 0.95], true),
  ],
  needed: [], assumptions: ["Pilot: 8 donors.", "Test: the app's composition test."],
  notes: ["Within a donor, 4 of 13 types vary no more than counting noise allows."], elapsed_s: 31,
};

beforeEach(() => {
  planPilot.mockReset().mockResolvedValue(PILOT);
  runCompositionPlan.mockReset().mockResolvedValue(PLAN);
  runDEPlan.mockReset();
  plots.length = 0;
});

describe("StudyPlanner", () => {
  it("plans the suggested types and answers in donors per group", async () => {
    render(<StudyPlanner datasetId="kang" info={KANG} />);
    await waitFor(() => expect(screen.getByTestId("planner-pilot")).toHaveTextContent(
      "Pilot: 8 donors of Donor, 8 CTRL and 8 STIM samples (paired); 814 cells per sample (median)."));
    expect(planPilot).toHaveBeenCalledWith("kang", { group_key: "cluster", condition_key: "stim" });
    fireEvent.click(screen.getByRole("button", { name: /Simulate/ }));
    await waitFor(() => expect(runCompositionPlan).toHaveBeenCalled());
    const body = runCompositionPlan.mock.calls[0]![1];
    expect(body).toMatchObject({
      group_key: "cluster", condition_key: "stim", groups: ["CD14 Mono", "pDC"],
      donors: DONORS, cells: [500, 2000, 5000], fold_changes: [1.5, 2, 3], alpha: 0.05, target_power: 0.8,
    });
    expect(body.assumed_sd).toBeUndefined();
    expect(body.design).toBeUndefined();

    const sentence = await screen.findByTestId("planner-sentence");
    expect(sentence).toHaveTextContent(
      "To detect a 2-fold increase in CD14 Mono's share (35.0% of the cells in CTRL) with 80% power you need about 6 donors per group at 2,000 cells per donor",
    );
    // One line per cells value plus the dashed upper-spread curve.
    const last = plots[plots.length - 1]!;
    expect(last.data.map((t) => [t.name, t.line.dash])).toEqual([
      ["500", "solid"], ["2,000", "solid"], ["2,000 (upper spread)", "dash"],
    ]);
    const table = screen.getByTestId("planner-needed");
    const row = within(table).getAllByRole("row")[1]!;
    // Cell type, effect, spread, then 8 at 500 cells, 6 at 2,000 and 12 at the upper spread.
    expect(within(row).getAllByRole("cell").map((c) => c.textContent)).toEqual(
      ["CD14 Mono", "×2", "0.20", "8", "6", "12"]);
    expect(screen.getByTestId("planner-assumptions")).toHaveTextContent("What this assumes (2)");
    expect(screen.getByRole("alert")).toHaveTextContent("Within a donor, 4 of 13 types");
  });

  it("a pilot without replicates says so and plans with an entered spread only", async () => {
    planPilot.mockResolvedValue({
      ...PILOT, sample_key: null, sample_key_detected: false, design: "none", replicated: false,
      message: "This dataset has no donor or sample column, so it holds no donor-to-donor variability to plan from.",
    });
    runCompositionPlan.mockResolvedValue({ ...PLAN, source: "assumed", assumed_sd: 0.35 });
    render(<StudyPlanner datasetId="kang" info={KANG} />);
    const box = await screen.findByTestId("planner-no-replicates");
    expect(screen.getByTestId("planner-pilot")).toHaveTextContent(
      "Pilot: no donor or sample column, so all 5,331 cells count as one sample per condition.");
    expect(box).toHaveTextContent("holds no donor-to-donor variability");
    expect(box).toHaveTextContent("an ASSUMPTION");
    expect(screen.getByLabelText("Assumed spread")).toHaveValue("0.35");
    fireEvent.change(screen.getByLabelText("Assumed spread"), { target: { value: "0.5" } });
    fireEvent.click(screen.getByRole("button", { name: /Simulate/ }));
    await waitFor(() => expect(runCompositionPlan).toHaveBeenCalled());
    expect(runCompositionPlan.mock.calls[0]![1].assumed_sd).toBe(0.5);
    expect(await screen.findByText(/ENTERED spread/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Pseudobulk DE" }));
    expect(screen.getByTestId("planner-no-replicates")).toHaveTextContent("only a pilot with replicate donors has");
    expect(screen.getByRole("button", { name: /Simulate/ })).toBeDisabled();
  });

  it("plans pseudobulk DE for the most abundant type by default", async () => {
    const de: DEPlan = {
      kind: "de", group_key: "cluster", group: "CD14 Mono", condition_key: "stim", sample_key: "Donor",
      sample_key_detected: true, conditions: ["CTRL", "STIM"], reference: "CTRL",
      pilot: { design: "paired", n_samples: { CTRL: 8, STIM: 8 }, n_donors: 8,
        cells_per_sample: { median: 227, min: 99, max: 638 }, dropped_samples: [], n_tested: 3454,
        n_simulated: 2000, counts_source: "layers['counts']", n_significant: 2318 },
      dispersion: { pilot_median: 0.027, bio_median: 0.019, cell_median: 1, donor_sd_median: 0.06 },
      expression_tertiles: [0.05, 0.2], design: "paired", can_pair: true, log2fc: [1], donors: DONORS,
      cells: [200, 1000], de_share: 0.1, alpha: 0.05, target_power: 0.8, reps: 11,
      points: [200, 1000].flatMap((cells) => DONORS.map((donors, i) => ({
        log2fc: 1, donors, cells, power: (cells === 200 ? [0.37, 0.53, 0.71, 0.78, 0.82, 0.83] : [0.75, 0.87, 0.95, 0.97, 0.98, 0.99])[i]!,
        se: 0.01, fdp: 0.05, tested_share: 0.95, power_by_expression: [0.5, 0.8, 0.95] }))),
      needed: [], assumptions: ["Test: DESeq2's negative binomial Wald test."], notes: [], elapsed_s: 50,
    };
    runDEPlan.mockResolvedValue(de);
    render(<StudyPlanner datasetId="kang" info={KANG} />);
    await screen.findByTestId("planner-pilot");
    fireEvent.click(screen.getByRole("button", { name: "Pseudobulk DE" }));
    expect(screen.getByLabelText("Planner DE cell type")).toHaveValue("CD14 Mono");
    fireEvent.click(screen.getByRole("button", { name: /Simulate/ }));
    await waitFor(() => expect(runDEPlan).toHaveBeenCalled());
    expect(runDEPlan.mock.calls[0]![1]).toMatchObject({ group: "CD14 Mono", log2fc: [0.5, 1], cells: [50, 200, 1000] });
    expect(await screen.findByTestId("planner-sentence")).toHaveTextContent(
      "Finding 80% of the genes changed by log2FC 1 (2-fold) in CD14 Mono (FDR < 0.05) takes about 4 donors per group at 1,000 CD14 Mono cells per donor; more cells still help (10 at 200 CD14 Mono cells per donor).",
    );
  });

  it("refuses a bad grid before calling the backend", async () => {
    render(<StudyPlanner datasetId="kang" info={KANG} />);
    await screen.findByTestId("planner-pilot");
    fireEvent.change(screen.getByLabelText("Donors per group"), { target: { value: "1, 4" } });
    expect(screen.getByText("Donors per group: whole numbers from 2 to 100.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Simulate/ })).toBeDisabled();
  });
});
