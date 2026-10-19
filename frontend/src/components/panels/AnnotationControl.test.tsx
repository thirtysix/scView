import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { CellTypistSummary } from "@/api/annotation";
import type { DatasetInfo } from "@/api/types";

const { getCellTypistModels, getCellTypistSummary } = vi.hoisted(() => ({
  getCellTypistModels: vi.fn(),
  getCellTypistSummary: vi.fn(),
}));
vi.mock("@/api/annotation", () => ({ getCellTypistModels, getCellTypistSummary }));

import { AnnotationControl } from "./AnnotationControl";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

const DATASET = {
  id: "d1", name: "paul15", filename: "p.h5ad", n_cells: 2730, n_genes: 3451, status: "ready",
  available_embeddings: ["X_umap"], created_at: "", active_clustering: "paul15_clusters",
  obs_columns: [{ name: "paul15_clusters", dtype: "category", n_unique: 19 }],
} as DatasetInfo;

const MODELS = {
  default: "Immune_All_Low.pkl",
  models: [
    { model: "Immune_All_Low.pkl", description: "immune sub-populations", species: "human" },
    { model: "Adult_Mouse_Gut.pkl", description: "adult mouse gut", species: "mouse" },
  ],
};

function summary(extra: Partial<CellTypistSummary> = {}): CellTypistSummary {
  return {
    column: "ct", confidence_column: "ct_confidence", percell_column: "ct_percell",
    model: "Adult_Mouse_Gut.pkl", model_species: "mouse", dataset_species: "mouse",
    gene_overlap: 0.2239, n_model_genes: 8258, n_shared_genes: 1849,
    n_cells: 2730, median_confidence: 0.2861, pct_below: 72.6, n_low_labels: 1,
    labels: [
      { label: "NK cell", n_cells: 31, median_confidence: 0.6569, pct_below: 38.7, low_confidence: false, pct_agree: 74.2 },
      { label: "Mast cell", n_cells: 91, median_confidence: 0.2819, pct_below: 74.7, low_confidence: true, pct_agree: 34.1 },
    ],
    cautions: ["The dataset has only 1,849 of the model's 8,258 genes (22%), under 50%."],
    low_confidence: 0.5, min_overlap: 0.5, note: "Confidence is each cell's highest class probability.",
    ...extra,
  };
}

describe("AnnotationControl (CellTypist)", () => {
  beforeEach(() => {
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
    useViewStore.getState().setPanel("assessment");
    getCellTypistModels.mockReset().mockResolvedValue(MODELS);
    getCellTypistSummary.mockReset().mockImplementation((_id: string, column?: string) =>
      Promise.resolve({ columns: ["ct"], dataset_species: "mouse", result: column ? summary() : null }));
  });

  it("lists mouse models first for a mouse dataset, preselects none, and warns on a human pick", async () => {
    render(<AnnotationControl onAnnotate={vi.fn()} running={false} species="mouse" />);
    fireEvent.click(screen.getByText("CellTypist (reference)"));
    const select = (await screen.findByLabelText("CellTypist model")) as HTMLSelectElement;
    // No general mouse model: nothing preselected, and Annotate waits for a pick.
    await waitFor(() => expect(select.value).toBe(""));
    expect(screen.getByText("Annotate cell types").closest("button")).toBeDisabled();
    const groups = Array.from(select.querySelectorAll("optgroup")).map((g) => g.label);
    expect(groups).toEqual(["Mouse models (this dataset) · tissue-specific, no tissue match guaranteed", "Human models"]);
    expect(screen.queryByRole("alert")).toBeNull();
    fireEvent.change(select, { target: { value: "Adult_Mouse_Gut.pkl" } });
    expect(screen.getByText("Annotate cell types").closest("button")).toBeEnabled();
    fireEvent.change(select, { target: { value: "Immune_All_Low.pkl" } });
    expect(screen.getByRole("alert")).toHaveTextContent(/looks mouse, but this model was trained on human cells/);
  });

  it("shows a run's per-label summary and colours the embedding by confidence", async () => {
    const onAnnotate = vi.fn().mockResolvedValue(undefined);
    render(<AnnotationControl onAnnotate={onAnnotate} running={false} species="mouse" />);
    fireEvent.click(screen.getByText("CellTypist (reference)"));
    fireEvent.change(screen.getByDisplayValue("paul15_clusters_celltypeAnno"), { target: { value: "ct" } });
    const select = (await screen.findByLabelText("CellTypist model")) as HTMLSelectElement;
    await waitFor(() => expect(select.querySelector("optgroup")).not.toBeNull());
    fireEvent.change(select, { target: { value: "Adult_Mouse_Gut.pkl" } });
    fireEvent.click(screen.getByText("Annotate cell types"));
    await waitFor(() => expect(onAnnotate).toHaveBeenCalled());
    expect(onAnnotate.mock.calls[0]![0]).toMatchObject({ annotation_method: "celltypist", annotation_target: "ct" });

    const card = await screen.findByTestId("celltypist-summary");
    expect(card).toHaveTextContent("1,849 of 8,258 model genes");
    expect(card).toHaveTextContent("1 label with a low median");
    expect(screen.getByText("Mast cell").closest("tr")).toHaveTextContent("low");
    expect(screen.getByText(/only 1,849 of the model's 8,258 genes/)).toBeInTheDocument();

    fireEvent.click(screen.getByText("Colour embedding by confidence"));
    expect(useSettingsStore.getState().colorBy).toBe("ct_confidence");
    expect(useViewStore.getState().activePanel).toBe("unified");
  });

  it("shows no summary for a column without a CellTypist run", async () => {
    getCellTypistSummary.mockResolvedValue({ columns: [], dataset_species: "human", result: null });
    render(<AnnotationControl onAnnotate={vi.fn()} running={false} species="human" />);
    await waitFor(() => expect(getCellTypistSummary).toHaveBeenCalledWith("d1"));
    expect(screen.queryByTestId("celltypist-summary")).toBeNull();
  });

  it("points to one output column per method for the cross-check, without changing the default", async () => {
    render(<AnnotationControl onAnnotate={vi.fn()} running={false} species="human" />);
    expect(screen.getByText(/run each into its own output column/)).toBeTruthy();
    expect(screen.getByText("cell_type_celltypist")).toBeTruthy();
    const input = screen.getByDisplayValue("paul15_clusters_celltypeAnno") as HTMLInputElement;
    expect(input.value).toBe("paul15_clusters_celltypeAnno");
  });
});
