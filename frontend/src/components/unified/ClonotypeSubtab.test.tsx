import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ClonotypeTable, Diversity, Overlap, Usage, VdjSummary } from "@/api/vdj";
import type { DatasetInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getVdj: vi.fn(),
  attachVdj: vi.fn(),
  redefineClonotypes: vi.fn(),
  removeVdj: vi.fn(),
  getClonotypes: vi.fn(),
  getDiversity: vi.fn(),
  getOverlap: vi.fn(),
  getUsage: vi.fn(),
}));
vi.mock("@/api/vdj", async (orig) => ({ ...(await orig<typeof import("@/api/vdj")>()), ...api }));
const refresh = vi.hoisted(() => vi.fn());
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; x?: unknown[] }[] }) => (
    <div data-testid="plot">{data.map((t) => `${t.type}:${(t.x ?? []).join(",")}`).join(";")}</div>
  ),
}));

import { ClonotypeSubtab } from "./ClonotypeSubtab";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

const EMPTY: VdjSummary = {
  has_vdj: false,
  n_cells: 6,
  how_to: "No TCR/BCR contigs are attached. Add Cell Ranger's filtered_contig_annotations.csv here.",
  sample_choices: [
    { name: "donor", values: ["d1", "d2"] },
    { name: "__barcode_suffix__", label: "barcode suffix", values: ["*-1", "*-2"] },
  ],
};

const FULL: VdjSummary = {
  ...EMPTY,
  has_vdj: true,
  files: [],
  match: {
    cells_profiled: 6, cells_with_chain: 4, cells_with_contigs: 5, cells_without_contigs: 1, cells_not_profiled: 0,
    barcodes_without_cells: 2, barcodes_without_cells_usable: 1, ambiguous_barcodes: 0,
  },
  status: { paired: 3, multichain: 1, "no contig": 2 },
  receptor: {},
  pairing: {},
  loci: { TRA: 4, TRB: 4 },
  n_clonotypes: 2,
  n_cells_in_clonotypes: 3,
  n_expanded: 1,
  n_cells_expanded: 2,
  expansion: { "1 cell": { clonotypes: 1, cells: 1 }, "2-5 cells": { clonotypes: 1, cells: 2 } },
  same_v_gene: false,
  definition: "Clonotype = identical CDR3 ...",
  stale: false,
  stale_note: null,
  columns: ["vdj_status", "clonotype", "clone_size", "clone_expansion"],
};

const TABLE: ClonotypeTable = {
  group_by: "cell_type", group: null, n_clonotypes: 2, n_shown: 2, n_cells_in_clonotypes: 3,
  rows: [
    { id: "TCR1", n_cells: 2, n_in_group: null, expansion: "2-5 cells", receptor: "TCR alpha-beta",
      chains: [{ arm: "VJ", locus: "TRA", cdr3: "CAVA", v: "TRAV1-2", j: "TRAJ33" }, { arm: "VDJ", locus: "TRB", cdr3: "CASSA", v: "TRBV6-5", j: "TRBJ1" }],
      by_group: { "CD8 T": 2 }, top_group: "CD8 T", top_share: 1, indices: [0, 3] },
    { id: "TCR2", n_cells: 1, n_in_group: null, expansion: "1 cell", receptor: "TCR alpha-beta",
      chains: [{ arm: "VDJ", locus: "TRB", cdr3: "CASSB", v: null, j: null }], by_group: { "CD4 T": 1 }, top_group: "CD4 T",
      top_share: 1, indices: [5] },
  ],
};

const DIV: Diversity = {
  group_by: "donor", rarefy: false, depth: null, seed: 0, n_draws: 20, min_cells: 10, caption: "Over the cells with a clonotype.",
  groups: [{ name: "d1", n_cells: 3, n_with_clonotype: 12, richness: 10, shannon: 2.1, simpson: 0.8, clonality: 0.1, gini: 0.1, note: null }],
};
const OVERLAP: Overlap = {
  group_by: "donor", metric: "morisita_horn", groups: ["d1", "d2"], n_cells: [12, 11], n_clonotypes: [10, 9],
  matrix: [[1, 0.2], [0.2, 1]], shared: [[10, 1], [1, 9]], n_groups_total: 2, n_groups_shown: 2,
};
const USAGE: Usage = {
  group_by: "donor", chain: "TRB", gene: "v", count: "clonotypes", chains_present: { TRB: 4, TRA: 4 }, genes: ["TRBV6-5"],
  n_genes_total: 1, groups: ["d1"], n: [10], counts: [[10]], share: [[1]], n_groups_total: 1, n_groups_shown: 1,
};

function renderTab() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <ClonotypeSubtab groupByColumn="cell_type" />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
  refresh.mockReset().mockResolvedValue({});
  api.getClonotypes.mockResolvedValue(TABLE);
  api.getDiversity.mockResolvedValue(DIV);
  api.getOverlap.mockResolvedValue(OVERLAP);
  api.getUsage.mockResolvedValue(USAGE);
  useSelectionStore.setState({ selectedCellIndices: null });
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: {
      id: "d1", name: "d1", filename: "", n_cells: 6, n_genes: 10, status: "ready", available_embeddings: [], created_at: "",
      obs_columns: [
        { name: "cell_type", dtype: "category", n_unique: 2, values: ["CD4 T", "CD8 T"] },
        { name: "donor", dtype: "category", n_unique: 2, values: ["d1", "d2"] },
        { name: "vdj_status", dtype: "category", n_unique: 3, values: ["paired", "multichain", "no contig"] },
      ],
    } as DatasetInfo,
  });
});

describe("ClonotypeSubtab", () => {
  it("says how to attach contigs and asks for the sample of a multi-library dataset", async () => {
    api.getVdj.mockResolvedValue(EMPTY);
    renderTab();
    expect(await screen.findByTestId("vdj-empty")).toHaveTextContent("filtered_contig_annotations.csv");
    expect(screen.getByText(/come from several libraries/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Attach/ })).toBeDisabled();
  });

  it("attaches a file for the picked sample and refreshes the dataset", async () => {
    api.getVdj.mockResolvedValue(EMPTY);
    api.attachVdj.mockResolvedValue({
      attached: { name: "t.csv", sample_column: "donor", sample: "d2", dropped: {}, n_kept_chains: 8, n_contigs: 8, matched: 4,
        n_barcodes: 4, matched_core: 0, unmatched: 0, ambiguous: 0 },
      summary: FULL,
    });
    renderTab();
    await screen.findByTestId("vdj-empty");
    const file = new File(["barcode,chain,cdr3,productive\n"], "t.csv", { type: "text/csv" });
    fireEvent.change(screen.getByLabelText("Contig table file"), { target: { files: [file] } });
    fireEvent.change(screen.getByLabelText("Sample column"), { target: { value: "donor" } });
    fireEvent.change(screen.getByLabelText("Sample"), { target: { value: "d2" } });
    fireEvent.click(screen.getByRole("button", { name: /Attach/ }));
    await waitFor(() => expect(api.attachVdj).toHaveBeenCalledWith("d1", file, { sampleColumn: "donor", sample: "d2", replace: false }));
    await waitFor(() => expect(refresh).toHaveBeenCalledWith("d1", "clonotype"));
    expect(await screen.findByTestId("vdj-status")).toHaveTextContent("multichain");
  });

  it("shows the match, colours by expansion and a row selects its cells", async () => {
    api.getVdj.mockResolvedValue(FULL);
    renderTab();
    expect(await screen.findByText("4 of 6 cells have a usable TCR or BCR chain.")).toBeInTheDocument();
    expect(screen.getByTestId("vdj-clonotype-counts")).toHaveTextContent("2 clonotypes over 3 cells; 1 expanded");
    fireEvent.click(screen.getByRole("button", { name: "Expansion" }));
    expect(useSettingsStore.getState().colorBy).toBe("clone_expansion");
    const row = (await screen.findByText("TCR1")).closest("tr")!;
    expect(within(row).getByText("TRA CAVA / TRB CASSA")).toBeInTheDocument();
    expect(within(row).getByText("CD8 T (100%)")).toBeInTheDocument();
    fireEvent.click(row);
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([0, 3]);
    fireEvent.click(row);
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
  });

  it("draws diversity, overlap and usage for the server's default grouping", async () => {
    api.getVdj.mockResolvedValue(FULL);
    renderTab();
    expect(await screen.findByTestId("vdj-overlap")).toBeInTheDocument();
    await waitFor(() => expect(api.getOverlap).toHaveBeenCalledWith("d1", "donor", "morisita_horn"));
    const plots = await screen.findAllByTestId("plot");
    expect(plots.map((p) => p.textContent)).toEqual(["bar:d1", "heatmap:d1,d2", "heatmap:TRBV6-5"]);
    expect(screen.getByTestId("vdj-diversity-caption")).toHaveTextContent("n = 12 cells in d1.");
    fireEvent.click(screen.getByRole("button", { name: "Jaccard" }));
    await waitFor(() => expect(api.getOverlap).toHaveBeenCalledWith("d1", "donor", "jaccard"));
  });

  it("recomputes stale clone sizes and removes the contigs after a confirmation", async () => {
    api.getVdj.mockResolvedValue({ ...FULL, stale: true, stale_note: "Cells were removed after the contigs were attached." });
    api.redefineClonotypes.mockResolvedValue({ summary: FULL });
    api.removeVdj.mockResolvedValue({ removed_columns: ["clonotype"], summary: EMPTY });
    renderTab();
    fireEvent.click(await screen.findByRole("button", { name: "Recompute" }));
    await waitFor(() => expect(api.redefineClonotypes).toHaveBeenCalledWith("d1", false));
    fireEvent.click(screen.getByRole("button", { name: "Remove contigs" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(api.removeVdj).toHaveBeenCalledWith("d1"));
    expect(await screen.findByTestId("vdj-empty")).toBeInTheDocument();
  });
});

describe("ClonotypeRepertoire without a grouping column", () => {
  it("says why instead of asking the backend (an unprocessed import has no groups)", async () => {
    const { ClonotypeRepertoire } = await import("@/components/unified/ClonotypeRepertoire");
    const { QueryClient, QueryClientProvider } = await import("@tanstack/react-query");
    const { render, screen } = await import("@testing-library/react");
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(
      <QueryClientProvider client={new QueryClient()}>
        <ClonotypeRepertoire datasetId="d1" groupColumns={[]} groupByColumn="" />
      </QueryClientProvider>,
    );
    expect(screen.getByText(/no grouping column yet/)).toBeTruthy();
    expect(fetchSpy.mock.calls.filter((c) => String(c[0]).includes("/vdj/diversity"))).toHaveLength(0);
    fetchSpy.mockRestore();
  });
});
