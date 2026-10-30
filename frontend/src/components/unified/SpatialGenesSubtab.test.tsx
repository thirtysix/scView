import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { MoranResult } from "@/api/spatialAutocorr";
import type { DatasetInfo, SpatialInfo } from "@/api/types";

const api = vi.hoisted(() => ({ getSpatialAutocorr: vi.fn(), runSpatialAutocorr: vi.fn() }));
vi.mock("@/api/spatialAutocorr", async (orig) => ({
  ...(await orig<typeof import("@/api/spatialAutocorr")>()),
  ...api,
}));

import { SpatialGenesSubtab, sortMoranRows } from "./SpatialGenesSubtab";
import { useDatasetStore } from "@/stores/datasetStore";

const lib = (id: string, n: number) => ({
  id, n_obs: n, images: {}, spot_diameter_fullres: 89, bounds: [0, 0, 100, 100] as [number, number, number, number],
});
const SPATIAL: SpatialInfo = {
  technology: "visium", unit: "spot", library_column: "library_id", libraries: [lib("A", 600), lib("B", 400)],
};

function setDataset(spatial: SpatialInfo | null, nCells = 1000, nGenes = 18000) {
  const ds: DatasetInfo = {
    id: "d1", name: "d1", filename: "", n_cells: nCells, n_genes: nGenes, status: "ready",
    available_embeddings: [], obs_columns: [], created_at: "", spatial,
  };
  useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: ds });
}

const row = (gene: string, I: number, padj: number) => ({
  gene, I, expected_I: -0.001, var_I: 0.0007, z: 10, pval: padj / 2, pval_adj: padj,
});
const RESULT: MoranResult = {
  result_id: "r1", method: "moran", n_obs: 1000, n_genes: 3, neighbors: { kind: "grid", k: 6 },
  libraries: ["A", "B"], genes: "hvg", matrix_source: "log-normalised X", unit: "spot", n_constant: 0,
  notes: ["3 spot(s) have no neighbour within reach; they count in the mean only."], rows_shown: 3,
  n_significant: 2, n_dispersed: 0, elapsed_seconds: 0.4, created_at: 0,
  rows: [row("CCL21", 0.69, 1e-80), row("CD3E", 0.31, 2e-9), row("ACTB", 0.01, 0.4)],
};

beforeEach(() => {
  api.getSpatialAutocorr.mockReset();
  api.runSpatialAutocorr.mockReset();
  api.getSpatialAutocorr.mockResolvedValue(null);
});

describe("SpatialGenesSubtab", () => {
  it("runs on the chosen genes and section and lists the genes by Moran's I", async () => {
    setDataset(SPATIAL);
    api.runSpatialAutocorr.mockResolvedValue({ ...RESULT, libraries: ["B"], n_obs: 400 });
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    await waitFor(() => expect(api.getSpatialAutocorr).toHaveBeenCalledWith("d1"));
    fireEvent.change(screen.getByLabelText("Genes"), { target: { value: "all" } });
    fireEvent.change(screen.getByLabelText("Section"), { target: { value: "B" } });
    fireEvent.click(screen.getByRole("button", { name: "Find spatial genes" }));
    await screen.findByText("CCL21");
    expect(api.runSpatialAutocorr).toHaveBeenCalledWith("d1", { genes: "all", library_id: "B" });
    expect(screen.getByText(/in section B/)).toBeInTheDocument();
    expect(screen.getByText(/n = 400 spots; 3 genes/)).toBeInTheDocument();
    expect(screen.getByText(/no neighbour within reach/)).toBeInTheDocument();
    const genes = screen.getAllByRole("row").slice(1).map((r) => within(r).getAllByRole("cell")[0]?.textContent);
    expect(genes).toEqual(["CCL21", "CD3E", "ACTB"]);
    expect(screen.getByRole("link", { name: /CSV/ })).toHaveAttribute(
      "href", expect.stringContaining("/datasets/d1/spatial/autocorr/r1.csv"));
  });

  it("restores the last result on opening and sorts by a column", async () => {
    setDataset(SPATIAL);
    api.getSpatialAutocorr.mockResolvedValue(RESULT);
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    await screen.findByText("CCL21");
    expect(screen.getByRole("button", { name: "Run again" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Gene" }));
    const genes = () => screen.getAllByRole("row").slice(1).map((r) => within(r).getAllByRole("cell")[0]?.textContent);
    expect(genes()).toEqual(["ACTB", "CCL21", "CD3E"]);
    expect(screen.getByRole("columnheader", { name: "Gene" })).toHaveAttribute("aria-sort", "ascending");
    fireEvent.click(screen.getByRole("button", { name: "Moran's I" }));
    fireEvent.click(screen.getByRole("button", { name: "Moran's I" }));
    expect(genes()).toEqual(["ACTB", "CD3E", "CCL21"]);
  });

  it("clicking a gene asks for its overlay", async () => {
    setDataset(SPATIAL);
    api.getSpatialAutocorr.mockResolvedValue(RESULT);
    const onGeneClick = vi.fn();
    render(<SpatialGenesSubtab onGeneClick={onGeneClick} activeGene="CD3E" />);
    fireEvent.click(await screen.findByRole("button", { name: "CCL21" }));
    expect(onGeneClick).toHaveBeenCalledWith("CCL21");
    expect(screen.getByRole("button", { name: "CD3E" })).toHaveAttribute("aria-pressed", "true");
  });

  it("says how long all genes take on many spots, and shows the error of a failed run", async () => {
    setDataset({ ...SPATIAL, libraries: [lib("A", 20000)] }, 20000, 20000);
    api.runSpatialAutocorr.mockRejectedValue(new Error("API error 422: No spots in section 'Z'."));
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    expect(screen.queryByLabelText("Section")).toBeNull(); // one section: no choice
    expect(screen.queryByTestId("time-note")).toBeNull();
    fireEvent.change(screen.getByLabelText("Genes"), { target: { value: "all" } });
    expect(screen.getByTestId("time-note")).toHaveTextContent("All 20,000 genes on 20,000 spots: about 5 s.");
    fireEvent.click(screen.getByRole("button", { name: "Find spatial genes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No spots in section");
  });

  it("says bins on Visium HD, each compared with the 4 sharing a side", async () => {
    setDataset({ technology: "visium_hd", unit: "bin", bin_size_um: 8, library_column: "library_id", libraries: [lib("A", 1000)] });
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /About Moran/ }));
    expect((await screen.findAllByText(/4 nearest bins \(those sharing a side\)/)).length).toBeGreaterThan(0);
  });

  it("is empty without spatial data and asks the server nothing", () => {
    setDataset(null);
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    expect(screen.getByText(/no spot, bin or cell positions/)).toBeInTheDocument();
    expect(api.getSpatialAutocorr).not.toHaveBeenCalled();
  });

  it("sorts missing values last", () => {
    const rows = [row("A", 0.1, 0.5), { ...row("B", 0, 0), I: null }, row("C", 0.3, 0.1)];
    expect(sortMoranRows(rows, "I", "desc").map((r) => r.gene)).toEqual(["C", "A", "B"]);
    expect(sortMoranRows(rows, "I", "asc").map((r) => r.gene)).toEqual(["A", "C", "B"]);
  });
});
