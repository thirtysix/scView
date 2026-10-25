/**
 * axe-core over the app's main shells: no serious or critical violations.
 * (Colour contrast is checked by accessibility mode, not here: jsdom cannot
 * measure it.) The plot canvases and the panels' own fetches are stubbed.
 */
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { seriousViolations } from "@/test/axe";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";
import { useSelectionStore } from "@/stores/selectionStore";

const STEPS = [
  "qc_metrics", "doublet_detection", "filtering", "normalization", "log_transform",
  "highly_variable_genes", "scaling", "pca", "batch_correction", "neighbors", "clustering",
  "embeddings", "marker_genes", "enrichment", "cell_cycle", "dissociation_stress",
];

vi.mock("@/hooks/useSessionRestore", () => ({ useSessionRestore: () => {} }));
vi.mock("@/components/layout/PanelContainer", () => ({ PanelContainer: () => <div>panel</div> }));
vi.mock("@/hooks/useEmbedding", async () => ({
  ...(await vi.importActual<object>("@/hooks/useEmbedding")),
  useEmbedding: () => ({
    positions: Float32Array.from([0, 0, 1, 1, 2, 2]),
    colors: Int32Array.from([0, 1, 1]),
    numCells: 3,
    colorColumnName: "cell_type",
    dimensions: 2,
    isLoading: false,
    error: null,
  }),
}));
// WebGL and Plotly need a real canvas.
vi.mock("@/components/plots/EmbeddingScatter", () => ({ EmbeddingScatter: () => <div /> }));
vi.mock("@/components/plots/Plot", () => ({ default: () => null, Plot: () => null }));
vi.mock("@/api/client", async () => ({
  ...(await vi.importActual<object>("@/api/client")),
  apiFetch: vi.fn(async (path: string) => {
    if (path.endsWith("/assessment")) {
      return {
        ...Object.fromEntries(STEPS.map((s, i) => [s, { done: i > 3, confidence: "high", details: "" }])),
        normalization_method: null,
        qc_threshold_mode: null,
      };
    }
    return new Promise(() => {});
  }),
  apiFetchBinary: vi.fn(() => new Promise(() => {})),
}));

import { AppLayout } from "@/components/layout/AppLayout";
import { UnifiedViewPanel } from "@/components/panels/UnifiedViewPanel";
import { DataAssessmentPanel } from "@/components/panels/DataAssessmentPanel";
import { ComparePanel } from "@/components/panels/ComparePanel";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { ScatterExportMenu } from "@/components/plots/ScatterExportMenu";

const DATASET: DatasetInfo = {
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 3,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap", "X_pca"],
  obs_columns: [
    { name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "NK"] },
    { name: "n_genes", dtype: "int64", n_unique: 3 },
  ],
  created_at: "",
};

const withQuery = (node: ReactNode) => (
  <QueryClientProvider client={new QueryClient()}>{node}</QueryClientProvider>
);
const settle = () => act(() => new Promise((r) => setTimeout(r, 30)));

beforeAll(() => {
  Element.prototype.scrollTo ??= () => {};
  globalThis.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
});

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "kang" });
  useSettingsStore.setState({ colorBy: "cell_type", embedding: "X_umap" });
  useViewStore.setState({ copilotOpen: false, activePanel: "unified" });
  useSelectionStore.getState().clearSelection();
});

describe("app shell (sidebar, header, main, co-pilot drawer)", () => {
  it("has a skip link, the landmarks, and no serious violations with the drawer shut or open", async () => {
    const { container } = render(withQuery(<AppLayout />));
    const skip = screen.getByRole("link", { name: "Skip to main content" });
    expect(skip).toHaveAttribute("href", "#main-content");
    expect(container.querySelector("#main-content")).toBe(screen.getByRole("main"));
    expect(screen.getByRole("navigation", { name: "Sidebar" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Collapse sidebar" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("button", { name: /Unified View/ })).toHaveAttribute("aria-current", "page");
    // Shut: the drawer is inert, so nothing in it takes focus or is read out.
    const drawer = container.querySelector('aside[aria-label="AI co-pilot"]')!;
    expect(drawer).toHaveAttribute("inert");
    expect(await seriousViolations(container)).toEqual([]);

    act(() => useViewStore.setState({ copilotOpen: true }));
    await settle();
    expect(screen.getByRole("complementary", { name: "AI co-pilot" })).not.toHaveAttribute("inert");
    expect(await seriousViolations(container)).toEqual([]);
  });

  it("collapsed, the icon-only sidebar buttons keep their names", async () => {
    useViewStore.setState({ sidebarCollapsed: true });
    const { container } = render(withQuery(<AppLayout />));
    expect(screen.getByRole("button", { name: "Data Assessment" })).toBeInTheDocument();
    expect(await seriousViolations(container)).toEqual([]);
    useViewStore.setState({ sidebarCollapsed: false });
  });
});

describe("Unified View (toolbar, tabs, legend)", () => {
  it("has no serious violations, with the tabs as a tablist", async () => {
    const { container } = render(withQuery(<UnifiedViewPanel />));
    await settle();
    expect(screen.getByRole("tablist", { name: "Analysis" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Markers" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel", { name: "Markers" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Embedding" })).toBeInTheDocument();
    expect(screen.getByRole("slider", { name: "Point size" })).toBeInTheDocument();
    expect(await seriousViolations(container)).toEqual([]);
  });

  it("the By group dialog has no serious violations either", async () => {
    const { container } = render(withQuery(<UnifiedViewPanel />));
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "By group" }));
    expect(screen.getByRole("dialog", { name: "Select cells by group" })).toBeInTheDocument();
    expect(await seriousViolations(container)).toEqual([]);
  });
});

describe("Data Assessment", () => {
  it("has no serious violations: the step checkboxes and chevrons are named", async () => {
    const { container } = render(withQuery(<DataAssessmentPanel />));
    await screen.findByRole("checkbox", { name: "Run QC Metrics" });
    expect(screen.getAllByRole("button", { name: /Show .* details/ }).length).toBeGreaterThan(0);
    expect(await seriousViolations(container)).toEqual([]);
  });
});

describe("Compare", () => {
  it("has no serious violations on any of its tabs", async () => {
    const { container } = render(withQuery(<ComparePanel />));
    for (const name of ["Joint embedding", "Composition", "Plan a study"]) {
      fireEvent.click(screen.getByRole("tab", { name }));
      await settle();
      expect(screen.getByRole("tab", { name })).toHaveAttribute("aria-selected", "true");
      expect(await seriousViolations(container)).toEqual([]);
    }
  });
});

describe("legend and export menu", () => {
  it("a categorical legend with every action has no serious violations", async () => {
    const { container } = render(
      <ColorLegend
        type="categorical"
        categories={["B", "NK"]}
        categoryColors={[[1, 2, 3], [4, 5, 6]]}
        label="cell_type"
        onCategoryClick={() => {}}
        onRenameCategory={() => {}}
        onAskAbout={() => {}}
        onSubcluster={() => {}}
        onCategoryToggle={() => {}}
        pickedCategories={["NK"]}
        onSelectPicked={() => {}}
        onClearPicked={() => {}}
        onReorder={() => {}}
      />,
    );
    expect(await seriousViolations(container)).toEqual([]);
  });

  it("the open export menu has no serious violations", async () => {
    const { container } = render(<ScatterExportMenu nPoints={10} onPng={() => {}} onFigure={async () => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Download figure" }));
    expect(screen.getByRole("menu")).toBeInTheDocument();
    expect(await seriousViolations(container)).toEqual([]);
  });
});
