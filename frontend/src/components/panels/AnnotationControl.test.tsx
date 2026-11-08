import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { CellTypistSummary } from "@/api/annotation";
import type { DatasetInfo } from "@/api/types";

const { getCellTypistModels, getCellTypistSummary } = vi.hoisted(() => ({
  getCellTypistModels: vi.fn(),
  getCellTypistSummary: vi.fn(),
}));
vi.mock("@/api/annotation", () => ({ getCellTypistModels, getCellTypistSummary }));
const { getLlmStatus } = vi.hoisted(() => ({ getLlmStatus: vi.fn() }));
vi.mock("@/api/assistant", () => ({ getLlmStatus }));
const { getLabelMatches } = vi.hoisted(() => ({ getLabelMatches: vi.fn() }));
vi.mock("@/api/ontology", () => ({ getLabelMatches }));
const { getProvenance } = vi.hoisted(() => ({ getProvenance: vi.fn() }));
vi.mock("@/api/provenance", () => ({ getProvenance }));

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
    getLlmStatus.mockReset().mockResolvedValue({ available: true });
    getProvenance.mockReset().mockResolvedValue({ recorded: { history: [] } });
    getCellTypistSummary.mockReset().mockImplementation((_id: string, column?: string) =>
      Promise.resolve({ columns: ["ct"], dataset_species: "mouse", result: column ? summary() : null }));
    getLabelMatches.mockReset().mockImplementation((_id: string, column: string) =>
      Promise.resolve({
        column, n_labels: 2, n_matched: 1, annotator: true, version: "cl/releases/2026-06-08/cl-basic.owl",
        labels: [
          { label: "NK cells", n_cells: 31, cl_id: "CL:0000623", cl_name: "natural killer cell", how: "normalised",
            how_text: "same name or exact synonym, case, plurals, '+' and hyphens aside", source: "model",
            model_cl_id: "CL:0000623", url: "https://www.ebi.ac.uk/ols4/ontologies/cl/classes?obo_id=CL:0000623" },
          { label: "CD14 Mono", n_cells: 91, cl_id: null, cl_name: null, how: "none", how_text: "no term",
            source: "", model_cl_id: null, url: null },
        ],
      }));
  });

  it("lists the written column's Cell Ontology ids after a run, a link only where a label matched", async () => {
    const onAnnotate = vi.fn().mockResolvedValue(undefined);
    render(<AnnotationControl onAnnotate={onAnnotate} running={false} species="human" />);
    expect(screen.queryByTestId("ontology-ids")).toBeNull(); // the output column does not exist yet
    fireEvent.click(screen.getByText("Annotate cell types"));
    const card = await screen.findByTestId("ontology-ids");
    expect(getLabelMatches).toHaveBeenCalledWith("d1", "paul15_clusters_celltypeAnno");
    expect(card).toHaveTextContent("1 of 2 labels in paul15_clusters_celltypeAnno matched · CL 2026-06-08");
    const link = screen.getByRole("link", { name: /CL:0000623/ });
    expect(link).toHaveAttribute("href", "https://www.ebi.ac.uk/ols4/ontologies/cl/classes?obo_id=CL:0000623");
    expect(screen.getByText("NK cells").closest("tr")).toHaveTextContent("AI · normalised");
    const mono = screen.getByText("CD14 Mono").closest("tr")!;
    expect(mono.querySelector("a")).toBeNull();
    expect(mono).not.toHaveTextContent("none");
    for (const h of ["Label", "CL id", "Match"]) expect(screen.getByText(h)).toHaveAttribute("title");
  });

  it("points to the qualified column when the run wrote one, and not otherwise", async () => {
    const onAnnotate = vi.fn().mockResolvedValue(undefined);
    render(<AnnotationControl onAnnotate={onAnnotate} running={false} species="human" />);
    fireEvent.click(screen.getByText("Annotate cell types"));
    await screen.findByTestId("ontology-ids");
    expect(screen.queryByText(/Also wrote/)).toBeNull();
    useDatasetStore.setState({
      currentDataset: { ...DATASET, obs_columns: [...DATASET.obs_columns,
        { name: "paul15_clusters_celltypeAnno_qualified", dtype: "category", n_unique: 6 }] } as DatasetInfo,
    });
    expect(await screen.findByText(/Also wrote/)).toHaveTextContent(
      "paul15_clusters_celltypeAnno_qualified: where clusters share a label");
  });

  it("shows the pipeline's cell_type column when the output column does not exist yet", async () => {
    useDatasetStore.setState({
      currentDataset: { ...DATASET, obs_columns: [...DATASET.obs_columns, { name: "cell_type", dtype: "category", n_unique: 9 }] } as DatasetInfo,
    });
    render(<AnnotationControl onAnnotate={vi.fn()} running={false} species="human" />);
    const card = await screen.findByTestId("ontology-ids");
    expect(getLabelMatches).toHaveBeenCalledWith("d1", "cell_type");
    expect(card).toHaveTextContent("labels in cell_type matched");
  });

  it("says why when the ontology cannot be had", async () => {
    getLabelMatches.mockRejectedValue(new Error("The Cell Ontology is not downloaded and MARKER_DB_AUTO_DOWNLOAD is off."));
    useDatasetStore.setState({
      currentDataset: { ...DATASET, obs_columns: [...DATASET.obs_columns, { name: "paul15_clusters_celltypeAnno", dtype: "category", n_unique: 5 }] } as DatasetInfo,
    });
    render(<AnnotationControl onAnnotate={vi.fn()} running={false} species="human" />);
    expect(await screen.findByTestId("ontology-ids-error")).toHaveTextContent("MARKER_DB_AUTO_DOWNLOAD is off");
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

  it("sends the marker sets choice for marker-score annotation, databases by default", async () => {
    const onAnnotate = vi.fn().mockResolvedValue(undefined);
    render(<AnnotationControl onAnnotate={onAnnotate} running={false} species="human" />);
    fireEvent.click(screen.getByText("Marker score (offline)"));
    fireEvent.click(screen.getByText("Annotate cell types"));
    await waitFor(() => expect(onAnnotate).toHaveBeenCalledTimes(1));
    expect(onAnnotate.mock.calls[0]![0]).toMatchObject({
      annotation_method: "marker_score", annotation_marker_sets: "databases",
    });
    fireEvent.change(await screen.findByLabelText("Marker sets"), { target: { value: "builtin" } });
    fireEvent.click(screen.getByText("Annotate cell types"));
    await waitFor(() => expect(onAnnotate).toHaveBeenCalledTimes(2));
    expect(onAnnotate.mock.calls[1]![0]).toMatchObject({ annotation_marker_sets: "builtin" });
  });

  it("starts on marker scores without an API key and says what the AI tab would do", async () => {
    getLlmStatus.mockResolvedValue({ available: false });
    const onAnnotate = vi.fn().mockResolvedValue(undefined);
    render(<AnnotationControl onAnnotate={onAnnotate} running={false} species="human" />);
    await screen.findByText("AI (needs an API key)");
    // The switch to marker scores is an effect after the status resolves: wait for it.
    expect(await screen.findByLabelText("Marker sets")).toBeTruthy();
    fireEvent.click(screen.getByText("Annotate cell types"));
    await waitFor(() => expect(onAnnotate).toHaveBeenCalledTimes(1));
    expect(onAnnotate.mock.calls[0]![0]).toMatchObject({ annotation_method: "marker_score" });
    fireEvent.click(screen.getByText("AI (needs an API key)"));
    expect(screen.getByText(/runs CellTypist instead/)).toBeTruthy();
  });

  it("warns when the LLM reply was cut short, from the annotation step's record", async () => {
    const step = (target: string, cut?: string) => ({
      step: "cell_type_annotation", tool: "t", timestamp: "",
      params: { annotation_method: "llm", annotation_target: target, ...(cut ? { llm_reply_cut: cut } : {}) },
    });
    const cut = "reply cut at 8,192 tokens: 3 of 40 clusters left Unknown (37, 38, 39)";
    getProvenance.mockResolvedValue({ recorded: { history: [step("cell_type", cut), step("paul15_clusters_celltypeAnno", cut)] } });
    const onAnnotate = vi.fn().mockResolvedValue(undefined);
    render(<AnnotationControl onAnnotate={onAnnotate} running={false} species="human" />);
    fireEvent.click(screen.getByText("Annotate cell types"));
    expect((await screen.findByRole("alert")).textContent).toContain("3 of 40 clusters left Unknown");

    // A clean reply (or a cut one for another column) shows nothing.
    getProvenance.mockResolvedValue({ recorded: { history: [step("paul15_clusters_celltypeAnno", cut), step("other")] } });
    fireEvent.click(screen.getByText("Annotate cell types"));
    await waitFor(() => expect(onAnnotate).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(getProvenance).toHaveBeenCalledTimes(2));
    await screen.findByText(/group by it in Unified View/);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
