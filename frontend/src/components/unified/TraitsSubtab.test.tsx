import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { GeneTraits, GwasResult, GwasStatus, GwasTraitRow } from "@/api/gwas";
import type { DatasetInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getGwasStatus: vi.fn(),
  getGwasClusters: vi.fn(),
  runGwasClusters: vi.fn(),
  getGeneTraits: vi.fn(),
  downloadGwas: vi.fn(),
}));
vi.mock("@/api/gwas", async (orig) => ({ ...(await orig<typeof import("@/api/gwas")>()), ...api }));

import { TraitsSubtab, sortTraitRows } from "./TraitsSubtab";
import { useDatasetStore } from "@/stores/datasetStore";

const CAVEAT =
  "Overlap between this cluster's marker genes and genes the GWAS Catalog maps near trait-associated variants. " +
  "Treat a hit as a lead for follow-up, not evidence that this cell type mediates the trait.";

function status(downloaded: boolean, extra: Partial<GwasStatus> = {}): GwasStatus {
  return {
    release: "2026-10-22", downloaded, raw_present: downloaded, auto_download: true, n_traits: downloaded ? 14154 : null,
    n_genes: null, n_loci: null, built_at: null, download_mb: 93, assoc_url: "", mappings_url: "",
    attribution: "NHGRI-EBI GWAS Catalog, release 2026-10-22 (ebi.ac.uk/gwas)", citation: "Sollis et al. 2023",
    licence: "CC0", caveat: CAVEAT, ...extra,
  };
}

const traitRow = (trait: string, p: number, extra: Partial<GwasTraitRow> = {}): GwasTraitRow => ({
  uri: `http://www.ebi.ac.uk/efo/${trait.replace(/ /g, "_")}`, trait, parent: "Immune system disorder",
  n_trait_genes: 247, n_overlap: 3, n_loci: 3, genes: ["CD40", "CXCR4", "IRF1"], dataset_genes: ["CD40", "CXCR4", "IRF1"],
  odds_ratio: 5.7, p, p_adj: p * 10, p_empirical: 0.002, n_draws: 10000, lead: p * 10 < 0.05, ...extra,
});

const RESULT: GwasResult = {
  result_id: "g1", column: "cluster", include_measurements: false, n_markers: 100, species: "human", mapping: null,
  n_background: 10747, n_measured: 14053, min_loci: 3, trait_size: [10, 500], n_draws: 1000, refine_draws: 10000,
  seed: 0, max_rows: 50, release: "2026-10-22", attribution: "NHGRI-EBI GWAS Catalog, release 2026-10-22 (ebi.ac.uk/gwas)",
  citation: "Sollis et al. 2023", caveat: CAVEAT, method: "", elapsed_seconds: 9.8, created_at: 0,
  groups: [
    { group: "Mk", n_markers: 36, n_query: 30, n_tested: 717, n_with_loci: 0, n_leads: 0, rows: [] },
    {
      group: "B", n_markers: 100, n_query: 78, n_tested: 717, n_with_loci: 64, n_leads: 1,
      rows: [
        traitRow("multiple sclerosis", 7e-5, { n_loci: 9, n_overlap: 9 }),
        traitRow("stroke", 0.02, { n_loci: 4, parent: "Cardiovascular disease" }),
      ],
    },
  ],
};

const GENE: GeneTraits = {
  gene: "CD40", ensembl: "ENSG00000101017", query: "CD40", via_ortholog: false, n_traits: 2, shown: 2,
  release: "2026-10-22", attribution: "", citation: "",
  traits: [
    { uri: "u1", trait: "rheumatoid arthritis", parent: "Immune system disorder", mlog: 12, p: 1e-12, n_loci: 1, pmids: [24390342] },
    { uri: "u2", trait: "multiple sclerosis", parent: "Immune system disorder", mlog: 9, p: 1e-9, n_loci: 1, pmids: [] },
  ],
};

function setDataset() {
  const ds: DatasetInfo = {
    id: "d1", name: "d1", filename: "", n_cells: 13836, n_genes: 14053, status: "ready", available_embeddings: [],
    obs_columns: [
      { name: "cluster", dtype: "category", n_unique: 13 },
      { name: "stim", dtype: "category", n_unique: 2 },
      { name: "n_genes", dtype: "int64", n_unique: 2000 },
    ],
    created_at: "",
  };
  useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: ds });
}

beforeEach(() => {
  for (const f of Object.values(api)) f.mockReset();
  api.getGwasClusters.mockResolvedValue(null);
  setDataset();
});

describe("TraitsSubtab", () => {
  it("offers the download when the catalog is not on disk and shows its progress", async () => {
    api.getGwasStatus.mockResolvedValue(status(false));
    let report: (p: unknown) => void = () => undefined;
    let finish: (s: GwasStatus) => void = () => undefined;
    api.downloadGwas.mockImplementation((onProgress: (p: unknown) => void) => {
      report = onProgress;
      return new Promise<GwasStatus>((r) => (finish = r));
    });
    render(<TraitsSubtab onGeneClick={vi.fn()} groupByColumn="cluster" setGroupByColumn={vi.fn()} />);
    expect(await screen.findByTestId("gwas-caveat")).toHaveTextContent("not evidence that this cell type mediates");
    expect(screen.getByTestId("gwas-citation")).toHaveTextContent("release 2026-10-22");
    expect(screen.getByTestId("gwas-citation")).toHaveTextContent("Sollis et al. 2023");
    expect(screen.queryByRole("button", { name: "Find trait links" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Download GWAS Catalog" }));
    act(() => report({ stage: "download", file: "x.zip", bytes: 20e6, total: 73e6 }));
    expect(await screen.findByTestId("gwas-progress")).toHaveTextContent("20 of 73 MB");
    await act(async () => finish(status(true)));
    expect(await screen.findByRole("button", { name: "Find trait links" })).toBeInTheDocument();
  });

  it("shows the server's message when the download fails", async () => {
    api.getGwasStatus.mockResolvedValue(status(false, { auto_download: false, raw_present: false }));
    api.downloadGwas.mockRejectedValue(new Error("The GWAS Catalog is not downloaded and GWAS_AUTO_DOWNLOAD is off."));
    render(<TraitsSubtab onGeneClick={vi.fn()} groupByColumn="cluster" setGroupByColumn={vi.fn()} />);
    expect(await screen.findByText(/Downloading is turned off on this server/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Download GWAS Catalog" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("GWAS_AUTO_DOWNLOAD is off");
  });

  it("runs on the group column, opens the group with leads, and lists traits with p-values", async () => {
    api.getGwasStatus.mockResolvedValue(status(true));
    api.runGwasClusters.mockResolvedValue(RESULT);
    render(<TraitsSubtab onGeneClick={vi.fn()} groupByColumn="cluster" setGroupByColumn={vi.fn()} />);
    const select = await screen.findByLabelText("Group by");
    expect(within(select).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "cluster (13 groups)", "stim (2 groups)",
    ]);
    fireEvent.click(screen.getByLabelText(/Include measurement traits/));
    fireEvent.click(screen.getByRole("button", { name: "Find trait links" }));
    await screen.findByText("multiple sclerosis");
    expect(api.runGwasClusters).toHaveBeenCalledWith("d1", { column: "cluster", include_measurements: true });
    expect(screen.getByLabelText("Group")).toHaveValue("B");
    expect(screen.getByText(/n = 2 groups; 10,747 background genes/)).toBeInTheDocument();
    expect(screen.getByText(/717 traits tested, 64 with overlap in 3\+ loci, 1 lead/)).toBeInTheDocument();
    expect(screen.getByTestId("cap-note")).toHaveTextContent("Top 2 of 64 by p");
    const first = screen.getAllByRole("row")[1]!;
    expect(within(first).getByRole("button", { name: "multiple sclerosis" })).toHaveClass("font-semibold");
    expect(first).toHaveTextContent("9 / 247");
    expect(first).toHaveTextContent("7.00e-5");
    expect(screen.getByText(/13 groups: about 13 s/)).toBeInTheDocument();
  });

  it("lists a trait's genes; a gene colours the plot and shows its traits with link-outs", async () => {
    api.getGwasStatus.mockResolvedValue(status(true));
    api.getGwasClusters.mockResolvedValue(RESULT);
    api.getGeneTraits.mockResolvedValue(GENE);
    const onGene = vi.fn();
    render(<TraitsSubtab onGeneClick={onGene} groupByColumn="cluster" setGroupByColumn={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "multiple sclerosis" }));
    const genes = screen.getByTestId("trait-genes");
    expect(genes).toHaveTextContent("9 overlap genes in 9 loci");
    fireEvent.click(within(genes).getByRole("button", { name: "CD40" }));
    expect(onGene).toHaveBeenCalledWith("CD40");
    await waitFor(() => expect(api.getGeneTraits).toHaveBeenCalledWith("CD40"));
    const card = await screen.findByTestId("gene-traits");
    await within(card).findByText("rheumatoid arthritis");
    expect(within(card).getByRole("link", { name: "24390342" })).toHaveAttribute(
      "href", "https://pubmed.ncbi.nlm.nih.gov/24390342/");
    expect(within(card).getByRole("link", { name: /Open Targets/ })).toHaveAttribute(
      "href", "https://platform.opentargets.org/target/ENSG00000101017");
    expect(within(card).getByRole("link", { name: /GWAS Catalog/ })).toHaveAttribute(
      "href", "https://www.ebi.ac.uk/gwas/genes/CD40");
    expect(within(card).getByRole("link", { name: /GTEx/ })).toHaveAttribute("href", "https://gtexportal.org/home/gene/CD40");
  });

  it("names mouse genes as the dataset does and says how they were mapped", async () => {
    api.getGwasStatus.mockResolvedValue(status(true));
    const mouse: GwasResult = {
      ...RESULT, species: "mouse", mapping: "mouse genes mapped to human with HCOP orthologs",
      groups: [{ ...RESULT.groups[1]!, rows: [traitRow("lupus", 1e-6, { genes: ["CD40"], dataset_genes: ["Cd40"] })] }],
    };
    api.getGwasClusters.mockResolvedValue(mouse);
    api.getGeneTraits.mockResolvedValue(null);
    const onGene = vi.fn();
    render(<TraitsSubtab onGeneClick={onGene} groupByColumn="cluster" setGroupByColumn={vi.fn()} />);
    expect(await screen.findByTestId("gwas-mapping")).toHaveTextContent("HCOP");
    fireEvent.click(screen.getByRole("button", { name: "lupus" }));
    fireEvent.click(screen.getByRole("button", { name: "Cd40 (CD40)" }));
    expect(onGene).toHaveBeenCalledWith("Cd40");
    expect(await screen.findByText(/maps no genome-wide significant association to CD40/)).toBeInTheDocument();
  });

  it("sorts rows", () => {
    const rows = [traitRow("a", 0.01, { n_loci: 3 }), traitRow("b", 0.001, { n_loci: 5 })];
    expect(sortTraitRows(rows, "p", "asc").map((r) => r.trait)).toEqual(["b", "a"]);
    expect(sortTraitRows(rows, "n_loci", "asc").map((r) => r.trait)).toEqual(["a", "b"]);
  });
});
