import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { LabelTransferSummary } from "@/api/labelTransfer";

const { getLabelTransfer, checkReference, runLabelTransfer, refresh } = vi.hoisted(() => ({
  getLabelTransfer: vi.fn(),
  checkReference: vi.fn(),
  runLabelTransfer: vi.fn(),
  refresh: vi.fn(),
}));
vi.mock("@/api/labelTransfer", () => ({ getLabelTransfer, checkReference, runLabelTransfer }));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));

import { ReferenceMappingCard } from "./ReferenceMappingCard";
import { unmappedLevel, referenceChoices, aboutGb } from "@/lib/labelTransfer";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const INFO = {
  methods: ["projection", "integration", "celltypist"], method: "projection", k: 15, min_k: 3, max_k: 100,
  min_shared_genes: 500, note: "note", saved: [], available: true, reason: null, n_cells: 400, running: false,
};
const RESULT: LabelTransferSummary = {
  prefix: "kang", columns: { label: "kang_transfer", score: "kang_transfer_score", unmapped: "kang_unmapped" },
  method: "projection", method_text: "projected onto the reference", k: 15, n_cells: 400, n_shared_genes: 900,
  n_hvg: 300, threshold: 1.6, percentile: 99, n_unmapped: 80, unmapped_share: 0.2, median_score: 0.93,
  labels: [{ label: "A", n_cells: 120, mean_score: 0.9, unmapped_share: 0 }],
  reference_labels: [{ label: "A", n_cells: 150, n_query: 120 }],
  score_hist: { counts: [1, 2, 3], edges: [0, 0.33, 0.66, 1] },
  cluster_column: "leiden",
  clusters: [
    { cluster: "3", n_cells: 80, unmapped_share: 0.95, top_label: "B", top_label_share: 0.5, candidate_absent: true },
    { cluster: "0", n_cells: 120, unmapped_share: 0.01, top_label: "A", top_label_share: 1, candidate_absent: false },
  ],
  reference: { id: "r", name: "Kang ref", label_column: "cell_type", n_reference_cells: 450 },
  sources: { reference: "counts", query: "counts", renamed: null }, seconds: 3, note: "not in reference note",
};

beforeEach(() => {
  vi.clearAllMocks();
  useDatasetStore.setState({
    availableDatasets: [
      { id: "q", name: "Query", status: "ready" },
      { id: "r", name: "Kang ref", status: "ready" },
      { id: "p", name: "Pending", status: "pending" },
    ] as never,
  });
  getLabelTransfer.mockResolvedValue(INFO);
  checkReference.mockResolvedValue({
    ok: true, reason: null, n_cells: 450, has_umap: true, label_columns: [{ column: "batch", n_labels: 2 }, { column: "cell_type", n_labels: 3 }],
    shared_genes_by_name: 900, min_shared_genes: 500, estimate: { peak_bytes: 2 ** 30, seconds: 10, seconds_range: [5, 20] },
  });
  runLabelTransfer.mockResolvedValue(RESULT);
  refresh.mockResolvedValue({});
});

describe("ReferenceMappingCard", () => {
  it("checks a reference, runs, shows the summary and cluster flags, and opens the cross-check", async () => {
    const onCrosscheck = vi.fn();
    render(<ReferenceMappingCard datasetId="q" onCrosscheck={onCrosscheck} />);
    const select = await screen.findByLabelText("Reference dataset");
    expect(screen.queryByText("Query")).toBeNull(); // not itself
    expect(screen.queryByText("Pending")).toBeNull();
    fireEvent.change(select, { target: { value: "r" } });
    await waitFor(() => expect(checkReference).toHaveBeenCalledWith("q", "r", "projection"));
    // The cell-type column is picked over the first one.
    await waitFor(() => expect((screen.getByLabelText("Reference label column") as HTMLSelectElement).value).toBe("cell_type"));
    expect(screen.getByText(/Needs about 1.0 GB, 5-20 s/)).toBeTruthy();
    fireEvent.click(screen.getByText("Map cells"));
    await screen.findByTestId("reference-mapping-result");
    expect(runLabelTransfer.mock.calls[0]?.[1]).toEqual({ reference_id: "r", label_column: "cell_type", method: "projection", k: 15 });
    expect(refresh).toHaveBeenCalledWith("q", "kang_transfer");
    expect(screen.getByText("absent from reference?")).toBeTruthy();
    fireEvent.click(screen.getByText("Colour by kang_unmapped"));
    expect(useSettingsStore.getState().colorBy).toBe("kang_unmapped");
    fireEvent.click(screen.getByText("Cross-check against clusters"));
    expect(onCrosscheck).toHaveBeenCalledWith(["kang_transfer", "leiden"]);
  });

  it("shows a refusal and the last saved mapping", async () => {
    getLabelTransfer.mockResolvedValue({ ...INFO, saved: [RESULT] });
    checkReference.mockResolvedValue({ ok: false, reason: "pick a mouse reference" });
    render(<ReferenceMappingCard datasetId="q" />);
    expect(await screen.findByTestId("reference-mapping-result")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Reference dataset"), { target: { value: "r" } });
    expect(await screen.findByText("pick a mouse reference")).toBeTruthy();
    expect((screen.getByText("Map cells").closest("button") as HTMLButtonElement).disabled).toBe(true);
  });

  it("notes a shift between the datasets only when it is large", async () => {
    getLabelTransfer.mockResolvedValue({ ...INFO, saved: [{ ...RESULT, shift: 2.1 }] });
    const { unmount } = render(<ReferenceMappingCard datasetId="q" />);
    expect((await screen.findByTestId("reference-mapping-shift")).textContent).toContain("2.1x");
    unmount();
    getLabelTransfer.mockResolvedValue({ ...INFO, saved: [{ ...RESULT, shift: 1.02 }] });
    render(<ReferenceMappingCard datasetId="q" />);
    await screen.findByTestId("reference-mapping-result");
    expect(screen.queryByTestId("reference-mapping-shift")).toBeNull();
  });

  it("explains a large dataset", async () => {
    getLabelTransfer.mockResolvedValue({ ...INFO, available: false, reason: "Reference mapping is not available" });
    render(<ReferenceMappingCard datasetId="q" />);
    expect(await screen.findByText("Reference mapping is not available")).toBeTruthy();
  });
});

describe("labelTransfer helpers", () => {
  it("levels, choices, sizes", () => {
    expect(unmappedLevel({ unmapped_share: 0.6, candidate_absent: true })).toBe("candidate");
    expect(unmappedLevel({ unmapped_share: 0.2, candidate_absent: false })).toBe("elevated");
    expect(unmappedLevel({ unmapped_share: 0.02, candidate_absent: false })).toBe("normal");
    expect(referenceChoices([{ id: "a", status: "ready" }, { id: "b", status: "ready" }], "a").map((d) => d.id)).toEqual(["b"]);
    expect(aboutGb(12 * 1024 ** 3)).toBe("about 12 GB");
  });
});
