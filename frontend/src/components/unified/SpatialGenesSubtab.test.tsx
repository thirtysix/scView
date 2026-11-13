import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { MoranResult } from "@/api/spatialAutocorr";
import type { DatasetInfo, SpatialInfo } from "@/api/types";

const api = vi.hoisted(() => ({ getSpatialAutocorr: vi.fn(), runSpatialAutocorr: vi.fn() }));
vi.mock("@/api/spatialAutocorr", async (orig) => ({
  ...(await orig<typeof import("@/api/spatialAutocorr")>()),
  ...api,
}));

import { SpatialGenesSubtab, sortMoranRows } from "./SpatialGenesSubtab";
import { hoverTip } from "@/test/hoverTip";
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
  library_id: "all", centring: "global", sections: [],
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
    expect(api.runSpatialAutocorr).toHaveBeenCalledWith("d1", { genes: "all", library_id: "B" }, expect.any(Function));
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

  it("counts the genes done while a large dataset's run streams", async () => {
    setDataset(SPATIAL);
    let finish: (r: unknown) => void = () => {};
    api.runSpatialAutocorr.mockImplementation((_d: string, _b: unknown, onProgress: (p: unknown) => void) => {
      onProgress({ done: 64, total: 2000 });
      return new Promise((ok) => {
        finish = ok;
      });
    });
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Find spatial genes" }));
    expect(await screen.findByText(/Computing… 64 of 2,000 genes/)).toBeTruthy();
    await act(async () => finish(RESULT));
  });

  it("says bins on Visium HD, each compared with the 4 sharing a side", async () => {
    setDataset({ technology: "visium_hd", unit: "bin", bin_size_um: 8, library_column: "library_id", libraries: [lib("A", 1000)] });
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    expect(await hoverTip(screen.getByText("Spatially variable genes"))).toMatch(/4 nearest bins \(those sharing a side\)/);
  });

  it("is empty without spatial data and asks the server nothing", () => {
    setDataset(null);
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    expect(screen.getByText(/no spot, bin or cell positions/)).toBeInTheDocument();
    expect(api.getSpatialAutocorr).not.toHaveBeenCalled();
  });

  it("shows each section's own I beside the pooled one and sorts by it", async () => {
    // QA (Kuppe 2022): the pooled I over 8 slides rose above every slide's own; now it is
    // within-section, and each section's own I shows a pattern carried by one slide.
    setDataset(SPATIAL);
    const by = (a: number, b: number) => ({
      A: { I: a, z: 5, pval: 1e-6, pval_adj: 1e-5 },
      B: { I: b, z: 5, pval: 1e-6, pval_adj: 1e-5 },
    });
    api.getSpatialAutocorr.mockResolvedValue({
      ...RESULT,
      centring: "section",
      sections: [{ library_id: "A", n_obs: 600 }, { library_id: "B", n_obs: 400 }],
      rows: [
        { ...row("CCL21", 0.69, 1e-80), by_library: by(0.9, 0.4) },
        { ...row("CD3E", 0.31, 2e-9), by_library: by(0.1, 0.6) },
        { ...row("ACTB", 0.01, 0.4), by_library: { A: { I: 0.02, z: 1, pval: 0.3, pval_adj: 0.5 } } },
      ],
    });
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    await screen.findByText("CCL21");
    expect(screen.getByText(/pooled over 2 sections, each centred on its own mean/)).toBeInTheDocument();
    const header = screen.getByRole("columnheader", { name: "B" });
    expect(header).toHaveAttribute("title", "Moran's I in section B alone (400 spots)");
    const cells = (gene: string) =>
      within(screen.getByRole("button", { name: gene }).closest("tr") as HTMLElement).getAllByRole("cell").map((c) => c.textContent);
    expect(cells("CCL21").slice(3)).toEqual(["0.900", "0.400"]);
    expect(cells("ACTB").slice(3)).toEqual(["0.020", "–"]);
    fireEvent.click(screen.getByRole("button", { name: "B" }));
    const genes = screen.getAllByRole("row").slice(1).map((r) => within(r).getAllByRole("cell")[0]?.textContent);
    expect(genes).toEqual(["CD3E", "CCL21", "ACTB"]);
  });

  it("opens on the last run's section and loads another section's own run", async () => {
    setDataset(SPATIAL);
    api.getSpatialAutocorr.mockImplementation((_d: string, lib?: string) =>
      Promise.resolve(lib === "all" ? null : { ...RESULT, library_id: lib ?? "A", libraries: [lib ?? "A"] }));
    render(<SpatialGenesSubtab onGeneClick={vi.fn()} />);
    await screen.findByText("CCL21");
    expect(screen.getByLabelText("Section")).toHaveValue("A");
    fireEvent.change(screen.getByLabelText("Section"), { target: { value: "B" } });
    await waitFor(() => expect(api.getSpatialAutocorr).toHaveBeenCalledWith("d1", "B"));
    expect(await screen.findByText(/in section B/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Section"), { target: { value: "" } });
    await waitFor(() => expect(api.getSpatialAutocorr).toHaveBeenCalledWith("d1", "all"));
    await waitFor(() => expect(screen.queryByText("CCL21")).toBeNull()); // no run over all sections yet
    expect(screen.getByRole("button", { name: "Find spatial genes" })).toBeInTheDocument();
  });

  it("sorts missing values last", () => {
    const rows = [row("A", 0.1, 0.5), { ...row("B", 0, 0), I: null }, row("C", 0.3, 0.1)];
    expect(sortMoranRows(rows, "I", "desc").map((r) => r.gene)).toEqual(["C", "A", "B"]);
    expect(sortMoranRows(rows, "I", "asc").map((r) => r.gene)).toEqual(["A", "C", "B"]);
  });
});
