import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { GateApplyResult, GateCheck, LearnedGateSummary, TrainResult } from "@/api/learnedGates";

const api = vi.hoisted(() => ({
  trainLearnedGate: vi.fn(),
  saveLearnedGate: vi.fn(),
  listLearnedGates: vi.fn(),
  learnedGateFile: vi.fn(),
  importLearnedGate: vi.fn(),
  deleteLearnedGate: vi.fn(),
  checkLearnedGate: vi.fn(),
  applyLearnedGate: vi.fn(),
  refresh: vi.fn(),
  downloadText: vi.fn(),
}));
vi.mock("@/api/learnedGates", () => ({
  trainLearnedGate: api.trainLearnedGate,
  saveLearnedGate: api.saveLearnedGate,
  listLearnedGates: api.listLearnedGates,
  learnedGateFile: api.learnedGateFile,
  importLearnedGate: api.importLearnedGate,
  deleteLearnedGate: api.deleteLearnedGate,
  checkLearnedGate: api.checkLearnedGate,
  applyLearnedGate: api.applyLearnedGate,
}));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => api.refresh }));
vi.mock("@/lib/geneLists", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/geneLists")>()),
  downloadText: api.downloadText,
}));

import type { DatasetInfo } from "@/api/types";
import { LearnedGateCard } from "./LearnedGateCard";
import { LearnedGateForm } from "./LearnedGateForm";
import { LearnedGatesLibrary } from "./LearnedGatesLibrary";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

const dataset: DatasetInfo = {
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 100,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "cluster", dtype: "category", n_unique: 2, values: ["B", "T"] },
    { name: "Donor", dtype: "category", n_unique: 2, values: ["d1", "d2"] },
  ],
  cells_fingerprint: "fp",
  created_at: "",
} as DatasetInfo;

const CV = { auroc: 0.991, precision: 0.94, recall: 0.92, f1: 0.93, n_pos: 40, n_neg: 50, folds: 5, grouped: true, C: 0.03, mid_share: 0.05 };
const RESULT: TrainResult = {
  gate: {
    id: "g1", name: "B cells", genes: ["MS4A1", "CD79A", "CD3E"], cv: CV, threshold: 0.55, excluded_rules: [],
    top_for: [{ gene: "MS4A1", weight: 1.8, mean_in: 2, mean_out: 0.1, pct_in: 0.9, pct_out: 0.05 }],
    top_against: [{ gene: "CD3E", weight: -0.7, mean_in: 0.1, mean_out: 1, pct_in: 0.04, pct_out: 0.6 }],
  },
  summary: {} as LearnedGateSummary,
  report: {
    verdict: { level: "good", text: "Expression separates these cells well." }, n_candidates: 300, n_genes: 3, C: 0.03,
    n_selected: 40, n_pool: 50, n_pos: 40, n_neg: 50, sample_column: "Donor", condition_column: null,
    samples: ["d1", "d2"], negatives: "own_samples", expression: "counts", warnings: [],
    disagree: { n_missed: 2, n_extra: 1, n_training: 90, cells: { n_cells: 100, indices: [3, 7, 9] } },
    seconds: 4, score_note: "A class-balanced model probability, not a frequency.",
  },
  overlap: { column: "cluster", group: "B", jaccard: 0.95, is_cluster: true, note: "This selection is almost exactly cluster = B." },
  saved: false,
};

const SUMMARY: LearnedGateSummary = {
  id: "g1", name: "B cells", created: "2026-10-29T10:00:00Z", species: "human", gene_kind: "symbol", n_genes: 3,
  threshold: 0.55, cv: CV, verdict: { level: "good", text: "" }, top_for: ["MS4A1"], top_against: ["CD3E"],
  source: { dataset_id: "kang", name: "Kang", n_selected: 40, sample_column: "Donor", condition_column: null, negatives: "own_samples" },
  scview_version: "0.47.1",
};
const CHECK: GateCheck = {
  ok: true, refusal: null, warnings: ["2 of the gate's genes are missing here"], species: "human", gate_species: "human",
  via: "name", coverage: { n_genes: 3, n_found: 3, gene_share: 1, weight_share: 1, missing_top: [] },
  columns: { column: "gate_b_cells", score_column: "gate_b_cells_score" }, replaces: false, sample_column: "Donor",
};
const APPLIED: GateApplyResult = {
  gate_id: "g1", name: "B cells", column: "gate_b_cells", score_column: "gate_b_cells_score", replaced: false,
  n_cells: 100, n_in: 12, share: 0.12, threshold: 0.55, mid_share: 0.03, training_mid_share: 0.05, sample_column: "Donor",
  per_sample: [{ sample: "d1", n_cells: 50, n_in: 5, share: 0.1 }, { sample: "d2", n_cells: 50, n_in: 7, share: 0.14 }],
  coverage: { n_genes: 3, n_found: 3, gene_share: 1, weight_share: 1, missing_top: [] }, via: "name", warnings: [],
  verdict: { level: "good", text: "" }, same_dataset: true, score_note: "Not a frequency.",
};

beforeEach(() => {
  vi.clearAllMocks();
  useSelectionStore.setState({ selectedCellIndices: new Set([1, 2, 3]) });
  api.refresh.mockResolvedValue({});
});

describe("LearnedGateForm", () => {
  it("trains on the selection with the chosen settings and shows the card", async () => {
    api.trainLearnedGate.mockImplementation(async (_id, _body, h) => {
      h.onStart?.({ n_pos: 3, n_neg: 50, n_candidates: 120 });
      return RESULT;
    });
    render(<LearnedGateForm dataset={dataset} onCancel={vi.fn()} />);
    const train = screen.getByRole("button", { name: "Train" });
    expect(train).toBeDisabled(); // a name first
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "B cells" } });
    fireEvent.change(screen.getByLabelText("Samples"), { target: { value: "Donor" } });
    fireEvent.click(screen.getByLabelText("Leave out cell-cycle and stress genes"));
    fireEvent.click(train);
    await screen.findByTestId("learned-gate-card");
    const [id, body] = api.trainLearnedGate.mock.calls[0]!;
    expect(id).toBe("kang");
    expect(body).toEqual({
      name: "B cells", cells: { n_cells: 100, indices: [1, 2, 3] }, sample_column: "Donor", exclude_cc_stress: false,
    });
    expect(screen.getByTestId("learned-gate-verdict")).toHaveTextContent("separates these cells well");
  });

  it("asks the server to pick the sample column by default, and Cancel aborts the run", async () => {
    let signal: AbortSignal | undefined;
    api.trainLearnedGate.mockImplementation(
      (_id, _body, h) =>
        new Promise((_res, rej) => {
          signal = h.signal;
          h.signal.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError")));
        }),
    );
    render(<LearnedGateForm dataset={dataset} onCancel={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "Train" }));
    expect(api.trainLearnedGate.mock.calls[0]![1].sample_column).toBeNull();
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(signal?.aborted).toBe(true);
    await screen.findByRole("button", { name: "Train" });
    expect(screen.queryByRole("alert")).toBeNull(); // a cancelled run is not an error
  });

  it("shows the server's refusal", async () => {
    api.trainLearnedGate.mockRejectedValue(new Error("The selection has 3 cells; a learned gate needs at least 20"));
    render(<LearnedGateForm dataset={dataset} onCancel={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "Train" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("at least 20");
  });
});

describe("LearnedGateCard", () => {
  it("shows the metrics with n, genes for and against, the cluster note; selects disagreements and saves", async () => {
    api.saveLearnedGate.mockResolvedValue(SUMMARY);
    const onSaved = vi.fn();
    render(<LearnedGateCard result={RESULT} nCells={100} onSaved={onSaved} />);
    expect(screen.getByText("0.991")).toBeInTheDocument();
    expect(screen.getByText(/n = 40 selected; 50 other cells/)).toBeInTheDocument();
    expect(within(screen.getByTestId("learned-gate-for")).getByText("MS4A1")).toBeInTheDocument();
    expect(within(screen.getByTestId("learned-gate-against")).getByText("CD3E")).toBeInTheDocument();
    expect(screen.getByTestId("learned-gate-cluster")).toHaveTextContent("almost exactly cluster = B");
    fireEvent.click(screen.getByRole("button", { name: "Select disagreements (3)" }));
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([3, 7, 9]);
    fireEvent.click(screen.getByRole("button", { name: "Save to library" }));
    await screen.findByText("In the library");
    expect(api.saveLearnedGate).toHaveBeenCalledWith("g1");
    expect(onSaved).toHaveBeenCalledWith("B cells");
  });
});

describe("LearnedGatesLibrary", () => {
  it("lists, checks first, then applies and shows the counts per sample", async () => {
    api.listLearnedGates.mockResolvedValue({ gates: [SUMMARY], score_note: "note", max_file_bytes: 262144 });
    api.checkLearnedGate.mockResolvedValue(CHECK);
    api.applyLearnedGate.mockResolvedValue(APPLIED);
    render(<LearnedGatesLibrary dataset={dataset} onClose={vi.fn()} />);
    const row = await screen.findByTestId("learned-gate-row-B cells");
    expect(row).toHaveTextContent("good fit");
    fireEvent.click(within(row).getByRole("button", { name: "Apply…" }));
    const panel = await screen.findByTestId("learned-gate-check");
    expect(panel).toHaveTextContent("3 of 3 genes found (100%)");
    expect(panel).toHaveTextContent("2 of the gate's genes are missing here");
    expect(api.applyLearnedGate).not.toHaveBeenCalled();
    fireEvent.click(within(panel).getByRole("button", { name: "Apply to this dataset" }));
    const done = await screen.findByTestId("learned-gate-applied");
    expect(api.applyLearnedGate).toHaveBeenCalledWith("kang", "g1", "Donor");
    expect(api.refresh).toHaveBeenCalledWith("kang", "gate_b_cells");
    expect(done).toHaveTextContent("n = 12 of 100 cells in the gate");
    expect(within(screen.getByTestId("learned-gate-per-sample")).getByText("14.0%")).toBeInTheDocument();
    fireEvent.click(within(done).getByRole("button", { name: "Colour by gate_b_cells" }));
    expect(useSettingsStore.getState().colorBy).toBe("gate_b_cells");
  });

  it("will not apply a refused gate", async () => {
    api.listLearnedGates.mockResolvedValue({ gates: [SUMMARY], score_note: "", max_file_bytes: 1 });
    api.checkLearnedGate.mockResolvedValue({ ...CHECK, ok: false, refusal: "Only 1 of the gate's 3 genes" });
    render(<LearnedGatesLibrary dataset={dataset} onClose={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Apply…" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Only 1 of the gate's 3 genes");
    expect(screen.getByRole("button", { name: "Apply to this dataset" })).toBeDisabled();
  });

  it("exports, deletes and imports gate files", async () => {
    api.listLearnedGates.mockResolvedValue({ gates: [SUMMARY], score_note: "", max_file_bytes: 1 });
    api.learnedGateFile.mockResolvedValue('{"format":"scview-learned-gate"}');
    api.deleteLearnedGate.mockResolvedValue({ deleted: "g1" });
    api.importLearnedGate.mockResolvedValue({ ...SUMMARY, name: "Imported" });
    render(<LearnedGatesLibrary dataset={dataset} onClose={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Export B cells" }));
    await waitFor(() =>
      expect(api.downloadText).toHaveBeenCalledWith("learned_gate_B_cells.json", '{"format":"scview-learned-gate"}', "application/json"),
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete B cells" }));
    await waitFor(() => expect(api.deleteLearnedGate).toHaveBeenCalledWith("g1"));
    const file = new File(['{"format":"scview-learned-gate"}'], "gate.json", { type: "application/json" });
    fireEvent.change(screen.getByLabelText("Import a learned gate file"), { target: { files: [file] } });
    await waitFor(() => expect(api.importLearnedGate).toHaveBeenCalledWith('{"format":"scview-learned-gate"}'));
    expect(await screen.findByRole("status")).toHaveTextContent("Imported “Imported”");
  });
});
