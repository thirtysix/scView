import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { DEGene, DEResponse } from "@/api/de";
import type { EnrichmentResult, GeneListEnrichmentResponse, GseaResponse } from "@/api/enrichment";

const { enrichGeneLists, runGsea } = vi.hoisted(() => ({ enrichGeneLists: vi.fn(), runGsea: vi.fn() }));
vi.mock("@/api/enrichment", () => ({ enrichGeneLists, runGsea }));
// The collection tree fetches MSigDB's hierarchy; not what is under test here.
vi.mock("@/components/panels/MSigDBCollectionTree", () => ({
  MSigDBCollectionTree: () => null,
  DEFAULT_MSIGDB_COLLECTIONS: ["h.all"],
}));

import { DEEnrichment } from "./DEEnrichment";
import { useViewStore } from "@/stores/viewStore";

const gene = (name: string, lfc: number, padj: number): DEGene => ({
  gene: name, logfoldchange: lfc, pval: padj / 10, pval_adj: padj, stat: lfc * 5,
});

const RESULT: DEResponse = {
  n_selected: 400, n_rest: 380, label: "STIM",
  genes: [
    gene("ISG15", 4.2, 1e-30),
    gene("IFIT1", 0.8, 1e-20),
    gene("CCL2", -1.5, 0.001),
    gene("FOS", -2, 0.3),
    gene("ACTB", 0.1, 0.9),
  ],
};

const term = (name: string, adj: number): EnrichmentResult => ({
  term: name, pvalue: adj / 10, adjusted_pvalue: adj, overlap_count: 2, gene_count: 40,
  genes: ["ISG15", "IFIT1"], collection: "h.all",
});

const RESPONSE: GeneListEnrichmentResponse = {
  background_n: 5,
  collections: ["h.all"],
  exclude_ribo_mito: true,
  source: "msigdb_local",
  up: { n_genes_requested: 1, n_genes_used: 1, n_excluded_ribo_mito: 0, n_terms: 1200, n_significant: 1100,
        results: [term("HALLMARK_INTERFERON_ALPHA_RESPONSE", 1e-6)] },
  down: { n_genes_requested: 1, n_genes_used: 1, n_excluded_ribo_mito: 0, n_terms: 1, n_significant: 0,
          results: [term("HALLMARK_INFLAMMATORY_RESPONSE", 0.2)] },
};

const GSEA: GseaResponse = {
  rank_metric: "wilcoxon_score",
  rank_metric_label: "Wilcoxon statistic (z-score)",
  n_genes_ranked: 5, n_excluded_ribo_mito: 0, n_sets_tested: 3, permutations: 1000,
  min_size: 15, max_size: 500, collections: ["h.all"], exclude_ribo_mito: true,
  up: { n_terms: 1, n_significant: 1, results: [{
    term: "HALLMARK_INTERFERON_GAMMA_RESPONSE", es: 0.7, nes: 2.05, pvalue: 0, fdr: 0, set_size: 192,
    leading_edge: ["ISG15", "IFIT1"], collection: "h.all",
  }] },
  down: { n_terms: 1, n_significant: 1, results: [{
    term: "HALLMARK_OXIDATIVE_PHOSPHORYLATION", es: -0.5, nes: -1.83, pvalue: 0, fdr: 0.002, set_size: 180,
    leading_edge: ["CCL2"], collection: "h.all",
  }] },
  seconds: 1.6, cached: false,
};

function setup(fcThreshold = 1, result = RESULT) {
  const props = {
    datasetId: "d1", result, labelA: "STIM", labelB: "CTRL", pThreshold: 0.05,
    onScoreGeneSet: vi.fn(), onGeneClick: vi.fn(),
  };
  const view = render(<DEEnrichment {...props} fcThreshold={fcThreshold} />);
  return { ...view, rerender: (fc: number, r: DEResponse = result) =>
    view.rerender(<DEEnrichment {...props} result={r} fcThreshold={fc} />) };
}

describe("DEEnrichment", () => {
  beforeEach(() => {
    enrichGeneLists.mockReset();
    runGsea.mockReset();
    useViewStore.setState({ pendingAsk: null });
  });

  it("sends the table's significant up and down genes with every tested gene as background", async () => {
    enrichGeneLists.mockResolvedValue(RESPONSE);
    setup();
    expect(screen.getByText("1 up · 1 down")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Enrich" }));
    await waitFor(() => expect(enrichGeneLists).toHaveBeenCalledTimes(1));
    expect(enrichGeneLists).toHaveBeenCalledWith("d1", {
      up: ["ISG15"],
      down: ["CCL2"],
      background: ["ISG15", "IFIT1", "CCL2", "FOS", "ACTB"],
      collections: ["h.all"],
      exclude_ribo_mito: true,
    });
  });

  it("states the cut-offs and background, and shows each direction on its own", async () => {
    enrichGeneLists.mockResolvedValue(RESPONSE);
    setup();
    fireEvent.click(screen.getByRole("button", { name: "Enrich" }));
    expect(await screen.findByText("HALLMARK_INTERFERON_ALPHA_RESPONSE")).toBeInTheDocument();
    expect(screen.getByText(/adj\. p ≤ 0\.05 and \|log2FC\| ≥ 1\.00: 1 higher in STIM, 1 higher in CTRL/))
      .toBeInTheDocument();
    expect(screen.getByText(/Background: the 5 genes tested/)).toBeInTheDocument();
    expect(screen.getByText(/The 1 most significant of 1,200 gene sets/)).toBeInTheDocument();
    // The count covers every significant term, not only the ones returned.
    expect(screen.getByRole("button", { name: /Higher in STIM · 1100 terms/ })).toBeInTheDocument();
    expect(screen.queryByText("HALLMARK_INFLAMMATORY_RESPONSE")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Higher in CTRL · 0 terms/ }));
    expect(screen.getByText("HALLMARK_INFLAMMATORY_RESPONSE")).toBeInTheDocument();
    expect(screen.queryByText("HALLMARK_INTERFERON_ALPHA_RESPONSE")).not.toBeInTheDocument();
  });

  it("says when the table's cut-off has moved since the run", async () => {
    enrichGeneLists.mockResolvedValue(RESPONSE);
    const { rerender } = setup();
    fireEvent.click(screen.getByRole("button", { name: "Enrich" }));
    await screen.findByText("HALLMARK_INTERFERON_ALPHA_RESPONSE");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    rerender(0.5);
    expect(screen.getByRole("status")).toHaveTextContent(
      "The table's cut-off is now |log2FC| ≥ 0.50; this enrichment used 1.00. Enrich again to match.",
    );
    expect(screen.getByText("2 up · 1 down")).toBeInTheDocument();
  });

  it("drops the enrichment when a new DE result arrives", async () => {
    enrichGeneLists.mockResolvedValue(RESPONSE);
    const { rerender } = setup();
    fireEvent.click(screen.getByRole("button", { name: "Enrich" }));
    await screen.findByText("HALLMARK_INTERFERON_ALPHA_RESPONSE");
    rerender(1, { ...RESULT, label: "other" });
    expect(screen.queryByText("HALLMARK_INTERFERON_ALPHA_RESPONSE")).not.toBeInTheDocument();
  });

  it("asks the co-pilot about a term with the comparison and direction", async () => {
    enrichGeneLists.mockResolvedValue(RESPONSE);
    setup();
    fireEvent.click(screen.getByRole("button", { name: "Enrich" }));
    await screen.findByText("HALLMARK_INTERFERON_ALPHA_RESPONSE");
    fireEvent.click(screen.getByTitle("Ask the co-pilot about this pathway"));
    const asked = useViewStore.getState().pendingAsk ?? "";
    expect(asked).toContain("comparison of STIM vs CTRL, the genes higher in STIM are enriched for");
    expect(asked).toContain('"HALLMARK_INTERFERON_ALPHA_RESPONSE"');
  });

  it("cannot run with no significant genes", () => {
    setup(10);
    expect(screen.getByText("0 up · 0 down")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enrich" })).toBeDisabled();
    // GSEA ranks every tested gene, so it needs no significant ones.
    expect(screen.getByRole("button", { name: "GSEA" })).toBeEnabled();
  });

  it("runs GSEA on every tested gene with its statistic and shows each direction", async () => {
    runGsea.mockResolvedValue(GSEA);
    setup();
    fireEvent.click(screen.getByRole("button", { name: "GSEA" }));
    await waitFor(() => expect(runGsea).toHaveBeenCalledTimes(1));
    expect(runGsea).toHaveBeenCalledWith("d1", {
      genes: RESULT.genes.map((g) => ({ gene: g.gene, logfoldchange: g.logfoldchange, pval: g.pval, score: g.stat })),
      collections: ["h.all"],
      exclude_ribo_mito: true,
    });
    const row = (await screen.findByText("HALLMARK_INTERFERON_GAMMA_RESPONSE")).closest("tr")!;
    expect(row).toHaveTextContent("+2.05");
    expect(row).toHaveTextContent("< 0.0010");
    expect(row).toHaveTextContent("2/192");
    expect(screen.getByText(/5 genes ranked by Wilcoxon statistic \(z-score\)/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Higher in CTRL · 1 set/ }));
    expect(screen.getByText("HALLMARK_OXIDATIVE_PHOSPHORYLATION")).toBeInTheDocument();
    expect(screen.queryByText("HALLMARK_INTERFERON_GAMMA_RESPONSE")).not.toBeInTheDocument();
  });

  it("names a result the server kept by its id instead of sending every gene back", async () => {
    runGsea.mockResolvedValue(GSEA);
    setup(1, { ...RESULT, result_id: "abc123" });
    fireEvent.click(screen.getByRole("button", { name: "GSEA" }));
    await waitFor(() => expect(runGsea).toHaveBeenCalledTimes(1));
    expect(runGsea).toHaveBeenCalledWith("d1", {
      de_result_id: "abc123", collections: ["h.all"], exclude_ribo_mito: true,
    });
  });

  it("asks the co-pilot about a GSEA term with its direction and leading edge", async () => {
    runGsea.mockResolvedValue(GSEA);
    setup();
    fireEvent.click(screen.getByRole("button", { name: "GSEA" }));
    await screen.findByText("HALLMARK_INTERFERON_GAMMA_RESPONSE");
    fireEvent.click(screen.getByTitle("Ask the co-pilot about this pathway"));
    const asked = useViewStore.getState().pendingAsk ?? "";
    expect(asked).toContain("enriched among the genes higher in STIM: NES 2.05");
    expect(asked).toContain("Leading-edge genes: ISG15, IFIT1");
  });

  it("drops the GSEA result when a new DE result arrives", async () => {
    runGsea.mockResolvedValue(GSEA);
    const { rerender } = setup();
    fireEvent.click(screen.getByRole("button", { name: "GSEA" }));
    await screen.findByText("HALLMARK_INTERFERON_GAMMA_RESPONSE");
    rerender(1, { ...RESULT, label: "other" });
    expect(screen.queryByText("HALLMARK_INTERFERON_GAMMA_RESPONSE")).not.toBeInTheDocument();
  });

  it("shows a GSEA error", async () => {
    runGsea.mockRejectedValue(new Error("API 400: Pick fewer collections."));
    setup();
    fireEvent.click(screen.getByRole("button", { name: "GSEA" }));
    expect(await screen.findByText(/GSEA: API 400: Pick fewer collections/)).toBeInTheDocument();
  });
});
