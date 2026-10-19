import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import type { MarkersResponse } from "@/api/markers";
import { useDatasetStore } from "@/stores/datasetStore";
import { useGeneListsStore } from "@/stores/geneListsStore";

const { apiFetch, plotProps } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  plotProps: { current: null as Record<string, unknown> | null },
}));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/components/plots/MarkerSpecificityPlot", () => ({
  MarkerSpecificityPlot: (props: Record<string, unknown>) => {
    plotProps.current = props;
    return <div data-testid="specificity-plot" />;
  },
}));

import { UnifiedMarkersSubtab } from "./UnifiedMarkersSubtab";

const DATASET: DatasetInfo = {
  id: "kang", name: "Kang", filename: "kang.h5ad", n_cells: 3, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], created_at: "",
  obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "pDC"] }],
};

const marker = (gene: string, lfc: number, tau: number | null, pin: number, pother: number) => ({
  gene, group: "pDC", logfoldchange: lfc, pval: 1e-9, pval_adj: 1e-7,
  pct_in: pin, pct_max_other: pother, max_other_group: "B", tau,
  pct_ratio: pother > 0 ? pin / pother : null,
});

const RESPONSE: MarkersResponse = {
  groups: ["pDC", "B"],
  markers: {
    pDC: [
      marker("CD74", 3.8, 0.52, 1, 1),
      marker("TSPAN13", 9.3, 0.98, 0.93, 0.089),
      marker("IGJ", 7.6, 0.979, 0.74, 0),
    ],
    B: [],
  },
};

const renderTab = () =>
  render(<UnifiedMarkersSubtab onGeneClick={vi.fn()} groupByColumn="cell_type" setGroupByColumn={vi.fn()} />);

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "kang" });
  apiFetch.mockReset();
  apiFetch.mockResolvedValue(RESPONSE);
  plotProps.current = null;
});

describe("UnifiedMarkersSubtab specificity", () => {
  it("adds the markers shown, in table order, to a gene list", async () => {
    useGeneListsStore.setState({ lists: [], selectedId: null });
    renderTab();
    await screen.findByText("TSPAN13");
    fireEvent.click(screen.getByText("Tau")); // sorted by tau
    fireEvent.click(screen.getByRole("button", { name: "Add to gene list" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "New list…" }));
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    const [list] = useGeneListsStore.getState().lists;
    expect([list!.name, list!.genes]).toEqual(["Markers: pDC", ["TSPAN13", "IGJ", "CD74"]]);
  });

  it("shows tau, detection here / next group, and the ratio", async () => {
    renderTab();
    const row = (await screen.findByText("TSPAN13")).closest("tr")!;
    expect(within(row).getByText("0.98")).toBeInTheDocument();
    expect(within(row).getByText("93% / 9%")).toBeInTheDocument();
    expect(within(row).getByText("10")).toBeInTheDocument();
    const igj = screen.getByText("IGJ").closest("tr")!;
    expect(within(igj).getByText("∞")).toBeInTheDocument();
    // The header says what tau means.
    expect(screen.getByText("Tau").closest("th")).toHaveAttribute("title", expect.stringMatching(/this group only/));
  });

  it("sorts by tau", async () => {
    renderTab();
    await screen.findByText("TSPAN13");
    fireEvent.click(screen.getByText("Tau"));
    const genes = screen.getAllByRole("row").slice(1).map((r) => within(r).getAllByRole("cell")[0]!.textContent);
    expect(genes).toEqual(["TSPAN13", "IGJ", "CD74"]);
  });

  it("plots the selected group's markers on request", async () => {
    renderTab();
    await screen.findByText("TSPAN13");
    expect(screen.queryByTestId("specificity-plot")).toBeNull();
    fireEvent.click(screen.getByText("Specificity vs fold change"));
    expect(screen.getByTestId("specificity-plot")).toBeInTheDocument();
    expect((plotProps.current!.markers as unknown[]).length).toBe(3);
  });

  it("leaves the columns out for markers without specificity", async () => {
    apiFetch.mockResolvedValue({
      groups: ["pDC"],
      markers: { pDC: [{ gene: "CD74", group: "pDC", logfoldchange: 3.8, pval: 1e-9, pval_adj: 1e-7 }] },
    });
    renderTab();
    await screen.findByText("CD74");
    expect(screen.queryByText("Tau")).toBeNull();
    expect(screen.queryByText("Specificity vs fold change")).toBeNull();
  });
});
