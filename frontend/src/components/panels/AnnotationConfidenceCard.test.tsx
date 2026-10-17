import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { ConfidenceInfo, ConfidenceResult } from "@/api/annotation";
import type { DatasetInfo } from "@/api/types";

const { getAnnotationConfidence, runAnnotationConfidence, makeUnknownColumn, refresh } = vi.hoisted(() => ({
  getAnnotationConfidence: vi.fn(),
  runAnnotationConfidence: vi.fn(),
  makeUnknownColumn: vi.fn(),
  refresh: vi.fn(),
}));
vi.mock("@/api/annotation", () => ({ getAnnotationConfidence, runAnnotationConfidence, makeUnknownColumn }));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));

import { AnnotationConfidenceCard } from "./AnnotationConfidenceCard";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

const DATASET = {
  id: "d1", name: "Kang", filename: "k.h5ad", n_cells: 13836, n_genes: 100, status: "ready",
  available_embeddings: ["X_umap"], created_at: "", active_clustering: "scview_leiden_r0.5",
  obs_columns: [
    { name: "scview_leiden_r0.5", dtype: "category", n_unique: 14 },
    { name: "cluster", dtype: "category", n_unique: 13 },
    { name: "cell_type", dtype: "category", n_unique: 12 },
  ],
} as DatasetInfo;

const NOTE = "This measures how well each cell's label agrees with the expression neighbourhood of the cells carrying that label, not whether the label is right.";
const REFERENCE_NOTE = "This measures how well each cell's label agrees with the CellTypist reference model's call for that cell, not whether the label is right.";

function info(result: ConfidenceResult | null = null, extra: Partial<ConfidenceInfo> = {}): ConfidenceInfo {
  return {
    available: true, reason: null, representation: "X_pca_harmony", n_pcs: 50, default_alpha: 0.1,
    min_alpha: 0.01, max_alpha: 0.5, min_label_cells: 20, max_labels: 100, classifiers: ["logreg", "knn", "celltypist"],
    note: NOTE, reference_note: REFERENCE_NOTE, reference_columns: {},
    checked_columns: result ? [result.column] : [], result, ...extra,
  };
}

function result(column = "cell_type", extra: Partial<ConfidenceResult> = {}): ConfidenceResult {
  const row = (label: string, unsure: number, not_fit: number, other: string) => ({
    label, n_cells: 1000, confident: 1 - unsure - not_fit, ambiguous: 0, unsure, not_fit,
    coverage_held_out: 1 - unsure - not_fit, confused_with: [{ label: other, n_cells: 120, share: 0.12 }],
  });
  return {
    column, alpha: 0.1, score: "lac", classifier: "logreg", representation: "X_pca_harmony", n_pcs: 50,
    n_cells: 13836, n_checked: 13836, labels_not_checked: [{ label: "Eryth", n_cells: 12 }],
    thresholds: [0.36, 0.39],
    coverage: { target: 0.9, held_out: 0.8941, n_held_out: 6911, calibration: 0.9004, n_calibration: 6925, sd: 0.0051, within_sampling_error: true },
    accuracy: 0.9219, mean_set_size: 0.95,
    counts: { confident: 12414, ambiguous: 0, unsure: 673, "doesn't fit": 749 },
    labels: [row("CD14 Mono", 0.016, 0.018, "DC"), row("CD4 Memory T", 0.119, 0.093, "CD4 Naive T")],
    confusions: [],
    columns: { flag: `scview_conf_${column}_flag`, fit: `scview_conf_${column}_fit`, set_size: `scview_conf_${column}_set_size`, alt: `scview_conf_${column}_alt` },
    unknown_column: null, stale: false, note: NOTE, elapsed_seconds: 0.3, timestamp: "",
    ...extra,
  };
}

describe("AnnotationConfidenceCard", () => {
  beforeEach(() => {
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
    useViewStore.getState().setPanel("assessment");
    getAnnotationConfidence.mockReset().mockResolvedValue(info());
    runAnnotationConfidence.mockReset().mockResolvedValue(result());
    makeUnknownColumn.mockReset();
    refresh.mockReset().mockResolvedValue({});
  });

  it("says what it measures, opens on the cell-type column and runs a check", async () => {
    render(<AnnotationConfidenceCard datasetId="d1" />);
    expect(screen.getByText(NOTE)).toBeTruthy();
    const select = screen.getByLabelText("Labels to check") as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe("cell_type"));
    // Cell types first, then clusterings (the active one first).
    expect(Array.from(select.options).map((o) => o.value)).toEqual(["cell_type", "scview_leiden_r0.5", "cluster"]);
    await waitFor(() => expect(getAnnotationConfidence).toHaveBeenLastCalledWith("d1", "cell_type"));

    fireEvent.click(screen.getByRole("button", { name: /Check labels/ }));
    await screen.findByTestId("confidence-result");
    expect(runAnnotationConfidence).toHaveBeenCalledWith("d1", "cell_type", 0.1, undefined); // the PCA classifier
    expect(refresh).toHaveBeenCalledWith("d1", "scview_conf_cell_type_flag");
    expect(screen.getByText(/89.4% of 6,911 held-out cells kept their label/)).toBeTruthy();
    expect(screen.getByText(/unsure 673 \(4.9%\)/)).toBeTruthy();
    expect(screen.getByText(/No cell is ambiguous at α = 0.1/)).toBeTruthy();
    // Worst-fitting label first, its rejected share flagged against alpha.
    const rows = within(screen.getByRole("table")).getAllByRole("row").slice(1);
    expect(rows[0]!.textContent).toMatch(/^CD4 Memory T/);
    expect(within(rows[0]!).getByText("21.2%").className).toMatch(/amber/);
    // Number columns don't wrap or run together when a label's neighbours wrap.
    const headers = within(screen.getByRole("table")).getAllByRole("columnheader");
    for (const h of headers.slice(1, 5)) expect(h.className).toMatch(/whitespace-nowrap.*pl-2/);
    expect(rows[0]!.textContent).toMatch(/CD4 Naive T 12.0%/);
    expect(screen.getByText(/Not checked \(fewer than 20 cells\): Eryth \(12\)/)).toBeTruthy();
  });

  it("scores a CellTypist column by CellTypist's probabilities, or by the PCA classifier", async () => {
    getAnnotationConfidence.mockResolvedValue(info(null, { reference_columns: { cell_type: "Immune_All_Low.pkl" } }));
    render(<AnnotationConfidenceCard datasetId="d1" />);
    const ct = await screen.findByRole("button", { name: "CellTypist (Immune_All_Low)" });
    expect(ct.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText(REFERENCE_NOTE)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Check labels/ }));
    await screen.findByTestId("confidence-result");
    expect(runAnnotationConfidence).toHaveBeenLastCalledWith("d1", "cell_type", 0.1, "celltypist");

    fireEvent.click(screen.getByRole("button", { name: "Classifier on the PCA" }));
    fireEvent.click(screen.getByRole("button", { name: /Check labels/ }));
    await waitFor(() => expect(runAnnotationConfidence).toHaveBeenLastCalledWith("d1", "cell_type", 0.1, undefined));
  });

  it("offers no choice for a column CellTypist did not make", async () => {
    getAnnotationConfidence.mockResolvedValue(info(null, { reference_columns: { cluster: "Immune_All_Low.pkl" } }));
    render(<AnnotationConfidenceCard datasetId="d1" />);
    await waitFor(() => expect(getAnnotationConfidence).toHaveBeenLastCalledWith("d1", "cell_type"));
    await act(async () => {});
    expect(screen.queryByRole("button", { name: /CellTypist \(/ })).toBeNull();
    expect(screen.getByText(NOTE)).toBeTruthy();
  });

  it("opens on the column's last check, at its alpha", async () => {
    getAnnotationConfidence.mockResolvedValue(info(result("cell_type", { alpha: 0.05 })));
    render(<AnnotationConfidenceCard datasetId="d1" />);
    await screen.findByTestId("confidence-result");
    expect((screen.getByLabelText("Error rate α") as HTMLInputElement).value).toBe("0.05");
    expect(runAnnotationConfidence).not.toHaveBeenCalled();
  });

  it("shows the flags on the embedding and makes the Unknown copy", async () => {
    getAnnotationConfidence.mockResolvedValue(info(result()));
    makeUnknownColumn.mockResolvedValue({ column: "cell_type", unknown_column: "scview_conf_cell_type_unknown", n_unknown: 1422, n_cells: 13836 });
    render(<AnnotationConfidenceCard datasetId="d1" />);
    await screen.findByTestId("confidence-result");

    fireEvent.click(screen.getByRole("button", { name: "Copy with rejected cells as Unknown" }));
    await screen.findByText(/1,422 rejected cells set to Unknown in scview_conf_cell_type_unknown/);
    expect(makeUnknownColumn).toHaveBeenCalledWith("d1", "cell_type");
    expect(refresh).toHaveBeenCalledWith("d1", "scview_conf_cell_type_unknown");
    fireEvent.click(screen.getByRole("button", { name: "Colour by the copy" }));
    expect(useSettingsStore.getState().colorBy).toBe("scview_conf_cell_type_unknown");

    useViewStore.getState().setPanel("assessment");
    fireEvent.click(screen.getByRole("button", { name: "Show flags on the embedding" }));
    expect(useSettingsStore.getState().colorBy).toBe("scview_conf_cell_type_flag");
    expect(useViewStore.getState().activePanel).toBe("unified");
  });

  it("says why it can't run, and shows the server's reason when a check fails", async () => {
    getAnnotationConfidence.mockResolvedValue(info(null, { available: false, reason: "The confidence check needs a PCA." }));
    const { unmount } = render(<AnnotationConfidenceCard datasetId="d1" />);
    await screen.findByText("The confidence check needs a PCA.");
    expect(screen.queryByRole("button", { name: /Check labels/ })).toBeNull();
    unmount();

    getAnnotationConfidence.mockResolvedValue(info());
    runAnnotationConfidence.mockRejectedValue(new Error('API error 400: {"detail":"The confidence check needs at least two labels."}'));
    render(<AnnotationConfidenceCard datasetId="d1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Check labels/ }));
    expect((await screen.findByRole("alert")).textContent).toBe("The confidence check needs at least two labels.");
  });

  it("warns when the labels changed since the check, and won't copy then", async () => {
    getAnnotationConfidence.mockResolvedValue(info(result("cell_type", { stale: true })));
    render(<AnnotationConfidenceCard datasetId="d1" />);
    await screen.findByText(/changed since this check ran; run it again/);
    expect((screen.getByRole("button", { name: "Copy with rejected cells as Unknown" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("won't run with an alpha out of range", async () => {
    render(<AnnotationConfidenceCard datasetId="d1" />);
    await waitFor(() => expect(getAnnotationConfidence).toHaveBeenLastCalledWith("d1", "cell_type"));
    await act(async () => {}); // its answer lands
    fireEvent.change(screen.getByLabelText("Error rate α"), { target: { value: "0.9" } });
    expect((screen.getByRole("button", { name: /Check labels/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("α must be between 0.01 and 0.5.")).toBeTruthy();
  });
});
