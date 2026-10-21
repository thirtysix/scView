import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { QcStateGroup, QcStates as Result } from "@/api/qcStates";

const { getQcStates, saveQcStateLabel, refresh } = vi.hoisted(() => ({
  getQcStates: vi.fn(),
  saveQcStateLabel: vi.fn(),
  refresh: vi.fn(),
}));
vi.mock("@/api/qcStates", () => ({ getQcStates, saveQcStateLabel }));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));
// Plotly needs a real canvas; the traces it is handed are what matter here.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { name: string }[] }) => (
    <div data-testid="plot">{data.map((t) => t.name).join("|")}</div>
  ),
}));

import { QcStates } from "./QcStates";

const group = (name: string, over: Partial<QcStateGroup> = {}): QcStateGroup => ({
  name,
  n_cells: 500,
  n_fitted: 500,
  fitted: true,
  n_states: 1,
  delta_bic: 72,
  separation: 1.3,
  flagged_label: null,
  flagged_share: null,
  minor_weight: 0.3,
  components: [{ label: "main", n_cells: 500, weight: 1, mean_rp_share: 30, mean_pct_mt: 4 }],
  per_condition: { ctrl: { n: 250, share: null }, drug: { n: 250, share: null } },
  reason: "One broad state: the BIC favours two components (+72) but they overlap (D = 1.3, under 2).",
  ...over,
});

const LNCAP = group("LNCaP", {
  n_states: 2,
  delta_bic: 412,
  separation: 2.6,
  flagged_label: "low-RP high-mito",
  flagged_share: 0.38,
  components: [
    { label: "main", n_cells: 310, weight: 0.62, mean_rp_share: 30, mean_pct_mt: 4 },
    { label: "low-RP high-mito", n_cells: 190, weight: 0.38, mean_rp_share: 10, mean_pct_mt: 20 },
  ],
  per_condition: { ctrl: { n: 250, share: 0.25 }, drug: { n: 250, share: 0.65 } },
  reason: "Two states: a second component improves the BIC by 412.",
});

const DATA: Result = {
  column: "cell_type",
  reason: "celltype",
  n_cells: 1040,
  has_mt: true,
  fit: "rp_mito",
  note: "",
  rules: { min_cells: 100, min_delta_bic: 10, min_separation: 2, min_weight: 0.05 },
  condition: { column: "treatment", kind: "condition", levels: ["ctrl", "drug"] },
  groups: [LNCAP, group("CAF"), group("tiny", { fitted: false, n_cells: 40, n_fitted: 40, delta_bic: null, separation: null, components: [], per_condition: {} })],
  two_state_groups: ["LNCaP"],
  overall: { n_fitted: 1000, n_flagged: 190, share: 0.19 },
  scatter: {
    x: [30, 10, 25],
    y: [4, 20, 3],
    group: [0, 0, 1],
    state: [0, 1, 0],
    x_label: "RP share of non-mito counts (%)",
    y_label: "% mito",
    n_shown: 3,
    max_per_group: 2000,
  },
};

describe("QcStates", () => {
  beforeEach(() => {
    getQcStates.mockReset().mockResolvedValue(DATA);
    saveQcStateLabel.mockReset();
    refresh.mockReset().mockResolvedValue({});
  });

  it("fits on Show only and tabulates states, delta BIC, D and the share per condition", async () => {
    render(<QcStates datasetId="d1" />);
    expect(getQcStates).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    const table = await screen.findByTestId("qc-states-table");
    expect(getQcStates).toHaveBeenCalledWith("d1", undefined);
    const rows = within(table).getAllByRole("row");
    expect(rows[1]!.textContent).toBe("LNCaP50024122.638%30.0 / 10.04.0 / 20.025%65%");
    expect(rows[1]!.getAttribute("title")).toMatch(/^Two states/);
    expect(rows[2]!.getAttribute("title")).toMatch(/One broad state/);
    expect(rows[3]!.textContent).toContain("not fitted");
    // Every header explains itself.
    for (const th of within(table).getAllByRole("columnheader")) expect(th.getAttribute("title")).toBeTruthy();
    expect(screen.getByTestId("qc-states-caption").textContent).toContain("n = 3 of 1,000 cells shown.");
    expect(screen.getByText(/Two states in 1 of 2 fitted groups of cell_type/)).toBeTruthy();
    expect(screen.getByTestId("plot").textContent).toBe("LNCaP|LNCaP: low-RP high-mito|CAF");
  });

  it("without mito says the fit is on RP share alone and drops the mito column", async () => {
    const note = "No mitochondrial (MT-) genes in this dataset, so % mito can't be measured.";
    getQcStates.mockResolvedValue({
      ...DATA,
      has_mt: false,
      fit: "rp_only",
      note,
      scatter: { ...DATA.scatter!, y: null, y_label: null },
    });
    render(<QcStates datasetId="d1" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    expect((await screen.findByTestId("qc-states-note")).textContent).toBe(note);
    expect(screen.getByTestId("qc-states-caption").textContent).toContain("no % mito: the fit is 1-D");
    expect(screen.queryByRole("columnheader", { name: "Mito %" })).toBeNull();
    expect(screen.getByTestId("plot").textContent).toBe("main|flagged state");
  });

  it("without a condition or sample column has no per-condition columns", async () => {
    getQcStates.mockResolvedValue({ ...DATA, condition: null, groups: [group("0", { per_condition: {} })], two_state_groups: [] });
    render(<QcStates datasetId="d1" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    const table = await screen.findByTestId("qc-states-table");
    expect(within(table).getAllByRole("columnheader").map((th) => th.textContent)).toEqual([
      "cell_type", "Cells", "States", "ΔBIC", "D", "Flagged", "RP %", "Mito %",
    ]);
    expect(screen.getByText(/No condition or sample column/)).toBeTruthy();
    // Nothing to label: every cell would be "main".
    expect((screen.getByRole("button", { name: /Save as label/ }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("saves the label, then refreshes the dataset so colour-by lists it", async () => {
    saveQcStateLabel.mockResolvedValue({
      name: "qc_state",
      column: "cell_type",
      counts: { main: 850, "low-RP high-mito": 190 },
      two_state_groups: ["LNCaP"],
      fit: "rp_mito",
    });
    render(<QcStates datasetId="d1" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await screen.findByTestId("qc-states-table");
    fireEvent.click(screen.getByRole("button", { name: /Save as label/ }));
    await waitFor(() => expect(screen.getByTestId("qc-states-saved")).toBeTruthy());
    expect(saveQcStateLabel).toHaveBeenCalledWith("d1", "cell_type", "qc_state");
    expect(refresh).toHaveBeenCalledWith("d1", "qc_state");
    expect(screen.getByTestId("qc-states-saved").textContent).toContain("850 main, 190 low-RP high-mito");
  });

  it("shows the server's reason when the fit is refused", async () => {
    getQcStates.mockRejectedValue(new Error('API error 400: {"detail":"No ribosomal-protein share for this dataset"}'));
    render(<QcStates datasetId="d1" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    expect((await screen.findByRole("alert")).textContent).toBe("No ribosomal-protein share for this dataset");
  });
});
