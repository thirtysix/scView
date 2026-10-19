import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { CrosscheckInfo, CrosscheckResult } from "@/api/annotation";
import type { DatasetInfo } from "@/api/types";

const { getCrosscheckInfo, runCrosscheck, saveCrosscheck, refresh, plots } = vi.hoisted(() => ({
  getCrosscheckInfo: vi.fn(),
  runCrosscheck: vi.fn(),
  saveCrosscheck: vi.fn(),
  refresh: vi.fn(),
  plots: [] as { data: { type: string; x?: string[]; y?: string[]; node?: { label: string[] } }[] }[],
}));
vi.mock("@/api/annotation", () => ({ getCrosscheckInfo, runCrosscheck, saveCrosscheck }));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));
vi.mock("@/lib/plotly", () => ({ default: { register: vi.fn() } }));
vi.mock("plotly.js/lib/sankey", () => ({ default: {} }));
vi.mock("@/lib/plotConfig", () => ({ plotConfig: () => ({}) }));
vi.mock("@/components/plots/Plot", () => ({
  default: (props: (typeof plots)[number]) => {
    plots.push(props);
    return <div data-testid={`plot-${props.data[0]!.type}`} />;
  },
}));

import { AnnotationCrosscheckCard } from "./AnnotationCrosscheckCard";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const DATASET = {
  id: "d1", name: "Kang", filename: "k.h5ad", n_cells: 100, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], created_at: "", active_clustering: "leiden",
  obs_columns: [
    { name: "leiden", dtype: "category", n_unique: 3, values: ["0", "1", "2"] },
    { name: "cell_type_llm", dtype: "category", n_unique: 2, values: ["T", "Mono"] },
    { name: "Donor", dtype: "category", n_unique: 4 },
  ],
} as DatasetInfo;

const INFO: CrosscheckInfo = {
  suggested: { cell_type_llm: "LLM", leiden: "clustering" }, active_clustering: "leiden",
  min_columns: 2, max_columns: 6, max_categories: 200, default_name: "crosscheck", saved: {}, note: "note",
};

// Reference cell_type_llm (T 60, Mono 40) against leiden: 0 and 2 fall in T, 1 in Mono.
const RESULT: CrosscheckResult = {
  columns: ["cell_type_llm", "leiden"], reference: "cell_type_llm", reference_labels: ["T", "Mono"],
  n_cells: 100, n_compared: 100,
  pairwise: { ari: { cell_type_llm: { cell_type_llm: 1, leiden: 0.7 }, leiden: { leiden: 1, cell_type_llm: 0.7 } },
    ami: { cell_type_llm: { cell_type_llm: 1, leiden: 0.6 }, leiden: { leiden: 1, cell_type_llm: 0.6 } } },
  mean_agreement: 0.9, agreement_counts: { "0": 10, "1": 90 }, exact_name_agreement: 0,
  consensus_counts: { T: 55, Mono: 35, Unresolved: 10 }, n_unresolved: 10,
  comparisons: {
    leiden: {
      ari: 0.7, ami: 0.6, n_cells: 100, mapped_agreement: 0.9,
      mapping: { "0": { to: "T", share: 1 }, "1": { to: "Mono", share: 0.8 }, "2": { to: "T", share: 1 } },
      exact_name: { n_shared_labels: 0, shared_labels: [], agreement: 0 },
      best_match: {},
      contingency: {
        rows: ["T", "Mono"], cols: ["0", "1", "2"],
        counts: { T: { "0": 40, "1": 10, "2": 10 }, Mono: { "1": 40 } },
        row_fraction: { T: { "0": 0.667, "1": 0.167, "2": 0.167 }, Mono: { "1": 1 } },
      },
    },
  },
  labels: {
    T: { n_cells: 60, mean_agreement: 0.833, disagreeing: [], annotators: { leiden: { agreement: 0.833, disagrees: false,
      calls: [{ label: "0", share: 0.667, maps_to: "T" }, { label: "1", share: 0.167, maps_to: "Mono" }] } } },
    Mono: { n_cells: 40, mean_agreement: 1, disagreeing: [], annotators: { leiden: { agreement: 1, disagrees: false,
      calls: [{ label: "1", share: 1, maps_to: "Mono" }] } } },
  },
  note: "mapped by majority", unresolved_label: "Unresolved", elapsed_seconds: 0.05,
};

describe("AnnotationCrosscheckCard", () => {
  beforeEach(() => {
    plots.length = 0;
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
    useSettingsStore.getState().setColorBy("leiden");
    getCrosscheckInfo.mockReset().mockResolvedValue(INFO);
    runCrosscheck.mockReset().mockResolvedValue(RESULT);
    saveCrosscheck.mockReset().mockResolvedValue({
      name: "crosscheck", columns: { consensus: "crosscheck_consensus", agreement: "crosscheck_agreement" },
      reference: "cell_type_llm", compared: RESULT.columns, n_unresolved: 10, mean_agreement: 0.9,
    });
    refresh.mockReset().mockResolvedValue({});
  });

  it("preselects the annotation and the clustering, compares and shows the pair", async () => {
    render(<AnnotationCrosscheckCard datasetId="d1" />);
    const group = screen.getByRole("group", { name: "Columns to compare" });
    await waitFor(() =>
      expect(within(group).getByRole("button", { name: /Cell Type Llm|cell_type_llm/i }).getAttribute("aria-pressed")).toBe("true"),
    );
    expect(within(group).getByRole("button", { name: /Donor/ }).getAttribute("aria-pressed")).toBe("false");
    // The annotation is the reference, not the clustering.
    expect((screen.getByLabelText("Reference") as HTMLSelectElement).value).toBe("cell_type_llm");

    fireEvent.click(screen.getByRole("button", { name: /Compare/ }));
    await waitFor(() => expect(screen.getByTestId("crosscheck-result")).toBeTruthy());
    expect(runCrosscheck).toHaveBeenCalledWith("d1", ["cell_type_llm", "leiden"], "cell_type_llm");
    expect(screen.getByTestId("crosscheck-summary").textContent).toContain("90.0% of 100 cells");
    expect(screen.getByTestId("crosscheck-summary").textContent).toContain("share no label names");

    // Confusion heatmap: best matches on the diagonal (0 and 2 under T, then 1 under Mono).
    const heat = plots.filter((p) => p.data[0]!.type === "heatmap").at(-1)!;
    expect(heat.data[0]!.y).toEqual(["T", "Mono"]);
    expect(heat.data[0]!.x).toEqual(["0", "2", "1"]);
    const sankey = plots.filter((p) => p.data[0]!.type === "sankey").at(-1)!;
    expect(sankey.data[0]!.node!.label).toEqual(["T", "Mono", "0", "2", "1"]);

    // Per-label table, worst agreement first by default, saying what the other column calls it.
    const rows = within(screen.getByTestId("crosscheck-labels")).getAllByRole("row").slice(1);
    expect(rows.map((r) => (r as HTMLTableRowElement).cells[0]!.textContent)).toEqual(["T", "Mono"]);
    expect(rows[0]!.textContent).toContain("1 16.7% (→ Mono)");
  });

  it("sorts the label table on a header click", async () => {
    render(<AnnotationCrosscheckCard datasetId="d1" />);
    await waitFor(() => expect(getCrosscheckInfo).toHaveBeenCalled());
    await waitFor(() => expect((screen.getByLabelText("Reference") as HTMLSelectElement).value).toBe("cell_type_llm"));
    fireEvent.click(screen.getByRole("button", { name: /Compare/ }));
    await waitFor(() => expect(screen.getByTestId("crosscheck-labels")).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: /^Cells/ }));
    const rows = within(screen.getByTestId("crosscheck-labels")).getAllByRole("row").slice(1);
    expect(rows.map((r) => (r as HTMLTableRowElement).cells[0]!.textContent)).toEqual(["T", "Mono"]); // 60 then 40 cells
    fireEvent.click(screen.getByRole("button", { name: /^Cells/ }));
    const again = within(screen.getByTestId("crosscheck-labels")).getAllByRole("row").slice(1);
    expect(again.map((r) => (r as HTMLTableRowElement).cells[0]!.textContent)).toEqual(["Mono", "T"]);
  });

  it("saves the consensus and agreement, refreshes, and keeps the result", async () => {
    render(<AnnotationCrosscheckCard datasetId="d1" />);
    await waitFor(() => expect((screen.getByLabelText("Reference") as HTMLSelectElement).value).toBe("cell_type_llm"));
    fireEvent.click(screen.getByRole("button", { name: /Compare/ }));
    await waitFor(() => expect(screen.getByTestId("crosscheck-result")).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: /Save consensus \+ agreement as columns/ }));
    await waitFor(() => expect(screen.getByText(/Saved crosscheck_consensus/)).toBeTruthy());
    expect(saveCrosscheck).toHaveBeenCalledWith("d1", ["cell_type_llm", "leiden"], "cell_type_llm", "crosscheck");
    expect(refresh).toHaveBeenCalledWith("d1", "crosscheck_consensus");
    // A refreshed dataset listing the new columns does not clear the result.
    useDatasetStore.setState({
      currentDataset: {
        ...DATASET,
        obs_columns: [...DATASET.obs_columns, { name: "crosscheck_consensus", dtype: "category", n_unique: 3 }],
      },
    });
    await waitFor(() => expect(screen.getByTestId("crosscheck-result")).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "Colour by agreement" }));
    expect(useSettingsStore.getState().colorBy).toBe("crosscheck_agreement");
  });

  it("shows the server's reason when a comparison is refused", async () => {
    runCrosscheck.mockRejectedValue(new Error('API error 400: {"detail":"\'leiden\' has 201 labels"}'));
    render(<AnnotationCrosscheckCard datasetId="d1" />);
    await waitFor(() => expect((screen.getByLabelText("Reference") as HTMLSelectElement).value).toBe("cell_type_llm"));
    fireEvent.click(screen.getByRole("button", { name: /Compare/ }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("'leiden' has 201 labels"));
  });
});
