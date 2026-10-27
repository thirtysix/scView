/** The co-pilot's open_compare_tab: the tab opens with the columns it asked for. */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { testComposition, planMilo, planPilot, getDataset } = vi.hoisted(() => ({
  testComposition: vi.fn(),
  planMilo: vi.fn(),
  planPilot: vi.fn(),
  getDataset: vi.fn(),
}));
vi.mock("@/api/composition", () => ({ testComposition }));
vi.mock("@/api/milo", () => ({ planMilo, runMilo: vi.fn(), miloCells: vi.fn() }));
vi.mock("@/api/planner", () => ({ planPilot, runCompositionPlan: vi.fn(), runDEPlan: vi.fn() }));
vi.mock("@/api/datasets", () => ({ getDataset, listDatasets: vi.fn().mockResolvedValue([]) }));
vi.mock("@/components/plots/Plot", () => ({ default: () => null }));
vi.mock("@/components/compare/JointEmbedding", () => ({ JointEmbedding: () => <div data-testid="joint" /> }));

import { CompositionTest } from "./CompositionTest";
import { MiloTest } from "./MiloTest";
import { StudyPlanner } from "./StudyPlanner";
import { ComparePanel } from "@/components/panels/ComparePanel";
import { applyViewAction } from "@/lib/copilotActions";
import { useComparePrefillStore } from "@/stores/comparePrefillStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";

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
    { name: "batch", dtype: "category", n_unique: 2, values: ["b1", "b2"] },
    { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "CD14 Mono", "pDC"] },
    { name: "cluster", dtype: "category", n_unique: 3, values: ["0", "1", "2"] },
    { name: "Donor", dtype: "category", n_unique: 8 },
    { name: "ind", dtype: "category", n_unique: 8 },
  ],
  created_at: "",
};

beforeEach(() => {
  testComposition.mockReset().mockReturnValue(new Promise(() => {}));
  planMilo.mockReset().mockReturnValue(new Promise(() => {}));
  planPilot.mockReset().mockReturnValue(new Promise(() => {}));
  getDataset.mockReset().mockResolvedValue(KANG);
  useComparePrefillStore.setState({ prefill: null, applied: 0 });
  useDatasetStore.setState({ currentDatasetId: "kang", currentDataset: KANG });
});

describe("Compare tabs take the co-pilot's columns", () => {
  it("Composition: condition, groups and sample as asked", async () => {
    render(
      <CompositionTest datasetId="kang" info={KANG} defaultCondition="batch" defaultGroup="cluster" defaultSample="ind" />,
    );
    await waitFor(() =>
      expect(testComposition).toHaveBeenLastCalledWith(
        "kang",
        expect.objectContaining({ condition_key: "batch", group_key: "cluster", sample_key: "ind" }),
      ),
    );
    expect(screen.getByLabelText("Sample column")).toHaveValue("ind");
  });

  it("Composition: a column the pickers do not offer falls back to the usual default", async () => {
    render(<CompositionTest datasetId="kang" info={KANG} defaultCondition="nope" defaultSample="nope" />);
    await waitFor(() =>
      expect(testComposition).toHaveBeenLastCalledWith(
        "kang",
        expect.objectContaining({ condition_key: "stim" }),
      ),
    );
    expect(testComposition.mock.lastCall?.[1]).not.toHaveProperty("sample_key");
  });

  it("Milo: condition, label and sample as asked", async () => {
    render(<MiloTest datasetId="kang" info={KANG} defaultCondition="batch" defaultGroup="cluster" defaultSample="ind" />);
    await waitFor(() =>
      expect(planMilo).toHaveBeenLastCalledWith(
        "kang",
        expect.objectContaining({ condition_key: "batch", label_key: "cluster", sample_key: "ind" }),
      ),
    );
  });

  it("Study planner: cell type, condition and donor columns as asked", async () => {
    render(<StudyPlanner datasetId="kang" info={KANG} defaultCondition="batch" defaultGroup="cluster" defaultSample="ind" />);
    await waitFor(() =>
      expect(planPilot).toHaveBeenLastCalledWith("kang", {
        group_key: "cluster", condition_key: "batch", sample_key: "ind",
      }),
    );
    expect(screen.getByLabelText("Planner donor column")).toHaveValue("ind");
  });
});

describe("open_compare_tab through the panel", () => {
  it("opens Compare on the Milo mode with the columns, once", async () => {
    applyViewAction({
      type: "open_compare_tab", tab: "milo", panel: "compare", condition: "batch", column: "cluster",
    });
    expect(useViewStore.getState().activePanel).toBe("compare");
    const { unmount } = render(<ComparePanel />);
    expect(screen.getByRole("tab", { name: "Composition" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: "Neighbourhoods (Milo)" })).toHaveClass("bg-primary");
    await waitFor(() =>
      expect(planMilo).toHaveBeenLastCalledWith(
        "kang",
        expect.objectContaining({ condition_key: "batch", label_key: "cluster" }),
      ),
    );
    unmount();
    // A later visit opens as usual (the request was applied).
    render(<ComparePanel />);
    expect(screen.getByRole("tab", { name: "Joint embedding" })).toHaveAttribute("aria-selected", "true");
  });

  it("opens the study planner tab", () => {
    applyViewAction({ type: "open_compare_tab", tab: "planner", panel: "compare", condition: "stim" });
    render(<ComparePanel />);
    expect(screen.getByRole("tab", { name: "Plan a study" })).toHaveAttribute("aria-selected", "true");
  });
});
