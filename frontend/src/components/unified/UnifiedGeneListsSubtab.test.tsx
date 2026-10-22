import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useGeneGridStore } from "@/stores/geneGridStore";
import { useGeneListsStore } from "@/stores/geneListsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { useViewStore } from "@/stores/viewStore";
import { useCustomGeneSetsStore } from "@/stores/customGeneSetsStore";
import { useScoringMethodStore } from "@/stores/scoringMethodStore";

const { apiFetch, enrichGeneLists, customCollectionFromLists } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  enrichGeneLists: vi.fn(),
  customCollectionFromLists: vi.fn(),
}));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/enrichment", () => ({ enrichGeneLists }));
vi.mock("@/api/genesets", async (orig) => ({ ...(await orig<typeof import("@/api/genesets")>()), customCollectionFromLists }));
vi.mock("@/components/panels/MSigDBCollectionTree", () => ({
  MSigDBCollectionTree: () => <div />,
  DEFAULT_MSIGDB_COLLECTIONS: ["h.all"],
}));

import { UnifiedGeneListsSubtab } from "./UnifiedGeneListsSubtab";

// A mouse dataset (paul15-like spelling).
const MOUSE_GENES = ["Cd3e", "Ms4a1", "Lyz2", "Gata1", "Klf1", ...Array.from({ length: 400 }, (_, i) => `G${i}`)];

const withQuery = (ui: ReactNode) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{ui}</QueryClientProvider>
);
const onGeneClick = vi.fn();
const onScoreGeneSet = vi.fn();
const renderTab = () =>
  render(withQuery(<UnifiedGeneListsSubtab onGeneClick={onGeneClick} onScoreGeneSet={onScoreGeneSet} />));
const lists = () => useGeneListsStore.getState().lists;

beforeEach(() => {
  localStorage.clear();
  onGeneClick.mockReset();
  onScoreGeneSet.mockReset();
  enrichGeneLists.mockReset();
  customCollectionFromLists.mockReset();
  apiFetch.mockReset();
  apiFetch.mockImplementation(async (path: string) => {
    if (path === "/datasets/ds1/genes") return { genes: MOUSE_GENES };
    if (path.includes("/genes/search")) return { query: "", results: [] };
    if (path.endsWith("/genesets/score")) return { scores: [0.1, 0.2] };
    throw new Error(`unexpected ${path}`);
  });
  useDatasetStore.setState({ currentDatasetId: "ds1" });
  useScoringMethodStore.setState({ method: "score_genes" });
  useGeneListsStore.setState({ lists: [], selectedId: null });
  useGeneGridStore.setState({ datasetId: null, genes: [], pair: null });
  useSplitViewStore.setState({ splitBy: null, compareColumns: null, geneGrid: null });
  useViewStore.setState({ activePanel: "unified", markerGenes: [], markerView: "table" });
});

/** A human-cased list on the mouse dataset: CD3E, MS4A1 exist as Cd3e, Ms4a1; FOO not at all. */
const seed = () => {
  const s = useGeneListsStore.getState();
  s.create("Other", ["Gata1"]);
  return s.create("Mixed", ["Lyz2", "CD3E", "Klf1", "MS4A1", "FOO", "Gata1"], "Markers: 3")!;
};

describe("UnifiedGeneListsSubtab", () => {
  it("lists every list with its gene count and how many the dataset lacks", async () => {
    seed();
    renderTab();
    const nav = screen.getByRole("list", { name: "Gene lists" });
    await waitFor(() => expect(within(nav).getByText(/3 missing/)).toBeInTheDocument());
    expect(within(nav).getByText("Other")).toBeInTheDocument();
    expect(within(nav).getByText(/^6 genes/)).toBeInTheDocument();
  });

  it("greys the missing genes, overlays a present one on click and removes a chip", async () => {
    seed();
    renderTab();
    await screen.findByText("3 not in this dataset");
    const chips = screen.getByLabelText("Genes in the list");
    const cd3e = within(chips).getByRole("button", { name: "CD3E" });
    expect(cd3e).toBeDisabled();
    expect(cd3e).toHaveAttribute("title", "Not in this dataset (it has Cd3e)");
    fireEvent.click(within(chips).getByRole("button", { name: "Lyz2" }));
    expect(onGeneClick).toHaveBeenCalledWith("Lyz2");
    fireEvent.click(within(chips).getByRole("button", { name: "Remove FOO" }));
    expect(lists()[1]!.genes).toEqual(["Lyz2", "CD3E", "Klf1", "MS4A1", "Gata1"]);
  });

  it("switches the list to the dataset's spelling where only the case differs", async () => {
    seed();
    renderTab();
    fireEvent.click(await screen.findByRole("button", { name: /Use the dataset's spelling for 2/ }));
    expect(lists()[1]!.genes).toEqual(["Lyz2", "Cd3e", "Klf1", "Ms4a1", "FOO", "Gata1"]);
  });

  it("opens the feature grid with the present genes, and the dot plot with them", async () => {
    seed();
    renderTab();
    await screen.findByText("3 not in this dataset");
    fireEvent.click(screen.getByRole("button", { name: "Feature grid" }));
    expect(useSplitViewStore.getState().geneGrid).toEqual({ datasetId: "ds1", kind: "features", genes: ["Lyz2", "Klf1", "Gata1"] });
    expect(useGeneGridStore.getState().genes).toEqual(["Lyz2", "Klf1", "Gata1"]);
    expect(screen.getByRole("button", { name: "Hide feature grid" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Heatmap" }));
    const v = useViewStore.getState();
    expect([v.markerGenes, v.markerView, v.activePanel]).toEqual([["Lyz2", "Klf1", "Gata1"], "heatmap", "markers"]);
  });

  it("caps the grid at nine genes and the plots at 300, and says so", async () => {
    const big = MOUSE_GENES.slice(5, 355); // 350 present genes
    useGeneListsStore.getState().create("Big", big);
    renderTab();
    await screen.findByText(/use the first 300 of the 350 genes/);
    fireEvent.click(screen.getByRole("button", { name: "Feature grid" }));
    expect(useSplitViewStore.getState().geneGrid!.genes).toEqual(big.slice(0, 9));
    fireEvent.click(screen.getByRole("button", { name: "Dot plot" }));
    expect(useViewStore.getState().markerGenes).toEqual(big.slice(0, 300));
  });

  it("scores the present genes as a set and hands the scores to the overlay", async () => {
    seed();
    renderTab();
    await screen.findByText("3 not in this dataset");
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(onScoreGeneSet).toHaveBeenCalled());
    const [path, init] = apiFetch.mock.calls.find((c) => String(c[0]).endsWith("/genesets/score"))!;
    expect(path).toBe("/datasets/ds1/genesets/score");
    expect(JSON.parse(init.body)).toEqual({ gene_set: ["Lyz2", "Klf1", "Gata1"], score_name: "Mixed", method: "score_genes" });
    expect(onScoreGeneSet).toHaveBeenCalledWith(
      new Float32Array([0.1, 0.2]), "Mixed", ["Lyz2", "Klf1", "Gata1"], undefined, "score_genes");
  });

  it("scores with the method chosen beside the Score button", async () => {
    seed();
    renderTab();
    await screen.findByText("3 not in this dataset");
    fireEvent.change(screen.getByRole("combobox", { name: "Gene-set scoring method" }), { target: { value: "aucell" } });
    expect(useScoringMethodStore.getState().method).toBe("aucell");
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(onScoreGeneSet).toHaveBeenCalled());
    const [, init] = apiFetch.mock.calls.find((c) => String(c[0]).endsWith("/genesets/score"))!;
    expect(JSON.parse(init.body).method).toBe("aucell");
    expect(onScoreGeneSet.mock.calls[0]![4]).toBe("aucell");
  });

  it("enriches the present genes against every measured gene and shows the terms", async () => {
    seed();
    enrichGeneLists.mockResolvedValue({
      background_n: MOUSE_GENES.length,
      collections: ["h.all"],
      exclude_ribo_mito: true,
      source: "gene_lists",
      up: {
        n_genes_requested: 3, n_genes_used: 3, n_excluded_ribo_mito: 0, n_terms: 1, n_significant: 1,
        results: [{ term: "HEME_METABOLISM", collection: "h.all", pvalue: 1e-4, adjusted_pvalue: 1e-3, overlap_count: 2, gene_count: 200, genes: ["Gata1", "Klf1"] }],
      },
      down: { n_genes_requested: 0, n_genes_used: 0, n_excluded_ribo_mito: 0, n_terms: 0, n_significant: 0, results: [] },
    });
    renderTab();
    await screen.findByText("3 not in this dataset");
    fireEvent.click(screen.getByRole("button", { name: "Enrichment" }));
    fireEvent.click(screen.getByRole("button", { name: "Enrich" }));
    expect(await screen.findByText("HEME_METABOLISM")).toBeInTheDocument();
    const [ds, body] = enrichGeneLists.mock.calls[0]!;
    expect(ds).toBe("ds1");
    expect(body.up).toEqual(["Lyz2", "Klf1", "Gata1"]);
    expect(body.down).toEqual([]);
    expect(body.background).toEqual(MOUSE_GENES);
  });

  it("saves this list or all lists as a collection, and says why one is refused", async () => {
    seed();
    useCustomGeneSetsStore.setState({ version: 0 });
    customCollectionFromLists.mockResolvedValueOnce({ name: "Mixed", n_sets: 1 });
    renderTab();
    fireEvent.click(screen.getByRole("button", { name: "Save as collection" }));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText(/Saved as the collection "Mixed" \(1 set\)/)).toBeInTheDocument();
    expect(customCollectionFromLists).toHaveBeenCalledWith("Mixed", {
      Mixed: ["Lyz2", "CD3E", "Klf1", "MS4A1", "FOO", "Gata1"],
    });
    // the collection lists fetch again
    expect(useCustomGeneSetsStore.getState().version).toBe(1);

    const { GeneSetUploadError } = await import("@/api/genesets");
    customCollectionFromLists.mockRejectedValueOnce(new GeneSetUploadError("A custom collection named 'Mixed' already exists."));
    fireEvent.click(screen.getByLabelText(/All 2 lists/));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("already exists");
    expect(Object.keys(customCollectionFromLists.mock.calls[1]![1])).toEqual(["Other", "Mixed"]);
  });

  it("renames on Enter, makes a new list and imports a file", async () => {
    seed();
    renderTab();
    const name = screen.getByLabelText("List name");
    fireEvent.change(name, { target: { value: "Renamed" } });
    fireEvent.keyDown(name, { key: "Enter" });
    fireEvent.blur(name);
    expect(lists()[1]!.name).toBe("Renamed");

    fireEvent.click(screen.getByRole("button", { name: /New/ }));
    expect(lists().map((l) => l.name)).toEqual(["Other", "Renamed", "New list"]);

    const file = new File(["S1\t\tGata1\tKlf1\n"], "sets.gmt");
    fireEvent.change(screen.getByLabelText("Import gene lists from a file"), { target: { files: [file] } });
    await waitFor(() => expect(lists().map((l) => l.name)).toContain("S1"));
  });
});
