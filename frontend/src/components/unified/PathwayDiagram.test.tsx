import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { DiagramNode, PathwayDiagramModel, PathwayValues } from "@/api/pathways";
import type { EnrichmentResult, GseaTerm } from "@/api/enrichment";

const { getReactomeDiagram, getReactomeValues, listDEResults } = vi.hoisted(() => ({
  getReactomeDiagram: vi.fn(),
  getReactomeValues: vi.fn(),
  listDEResults: vi.fn(),
}));
vi.mock("@/api/pathways", async (orig) => ({
  ...(await orig<typeof import("@/api/pathways")>()),
  getReactomeDiagram,
  getReactomeValues,
}));
vi.mock("@/api/deOverlap", async (orig) => ({
  ...(await orig<typeof import("@/api/deOverlap")>()),
  listDEResults,
}));

import { PathwayDiagram, nodeFill } from "./PathwayDiagram";
import { PathwayDiagramPanel } from "./PathwayDiagramPanel";
import { EnrichmentTermsTable } from "./EnrichmentTermsTable";
import { GseaTermsTable } from "./GseaTermsTable";
import { diagramPalette, lfcScale } from "@/lib/pathwayDiagram";

const node = (id: number, cls: DiagramNode["class"], label: string, genes?: string[], extra: Partial<DiagramNode> = {}): DiagramNode => ({
  id, reactome_id: id * 10, class: cls, label, x: id * 100, y: 10, w: 80, h: 30, ...(genes ? { genes } : {}), ...extra,
});

const MODEL: PathwayDiagramModel = {
  st_id: "R-HSA-6783589", name: "Interleukin-6 family signaling", species: "human", requested: "R-HSA-1059683",
  release: 98, attribution: "Reactome (CC BY 4.0), release 98", licence: "CC BY 4.0",
  licence_url: "https://reactome.org/license", citation: "Milacic et al. 2024, Nucleic Acids Res (PMID 37941124)",
  url: "https://reactome.org/PathwayBrowser/#/R-HSA-1059683",
  bounds: { minX: 0, maxX: 800, minY: 0, maxY: 200 },
  compartments: [{ id: 80, label: "cytosol", x: 0, y: 0, w: 800, h: 200, text: [10, 180], inset: { x: 5, y: 5, w: 790, h: 190 } }],
  nodes: [
    node(1, "Protein", "STAT1 [cytosol]", ["STAT1"]),
    node(2, "Complex", "STAT1:IRF9", ["STAT1", "IRF9", "IRF1"]),
    node(3, "EntitySet", "IRF set", ["IRF9", "IRF1"]),
    node(4, "Chemical", "ATP"),
    node(5, "Gene", "SOCS1 gene", ["SOCS1"]),
    node(6, "ProcessNode", "Interleukin-6 signaling", undefined, { st_id: "R-HSA-1059683" }),
    node(7, "RNA", "MIR21 RNA", ["MIR21"]),
  ],
  reactions: [
    { id: 50, reactome_id: 5001, label: "binds", type: "association", segments: [[90, 50, 100, 50]],
      shape: { type: "BOX", empty: true, a: [84, 44], b: [96, 56] }, end: { type: "ARROW", empty: false, a: [96, 46], b: [100, 50], c: [96, 54] } },
    { id: 51, reactome_id: 5002, label: "other", type: "transition", segments: [[150, 90, 200, 90]],
      shape: { type: "CIRCLE", empty: false, c: [145, 90], r: 5 } },
  ],
  connectors: [{ node: 1, edge: 50, type: "INHIBITOR", segments: [[70, 30, 90, 50]], end: { type: "STOP", empty: false, a: [88, 48], b: [92, 52] } }],
  links: [],
  n_genes: 5, too_broad: false, suggestions: [],
  message: "R-HSA-1059683 has no diagram of its own: it is drawn inside Interleukin-6 family signaling (R-HSA-6783589), outlined.",
  highlight: { st_id: "R-HSA-1059683", name: "Interleukin-6 signaling", exact: true, outline: [[5, 5], [300, 5], [300, 100], [5, 100]], reactions: [50] },
};

const VALUES: PathwayValues = {
  st_id: "R-HSA-6783589", requested: "R-HSA-1059683", label: "CD14 Mono: STIM vs CTRL (Wilcoxon)", source: "de:r1",
  genes: { STAT1: { gene: "STAT1", lfc: 2, padj: 0.001 }, IRF1: { gene: "IRF1", lfc: -1, padj: 0.3 } },
  nodes: {
    "1": { value: 2, n_genes: 1, n_measured: 1, n_changed: 1 },
    "2": { value: 0.5, n_genes: 3, n_measured: 2, n_changed: 1 },
    "3": { value: -1, n_genes: 2, n_measured: 1, n_changed: 0 },
    "5": { value: null, n_genes: 1, n_measured: 0, n_changed: 0 },
    "7": { value: null, n_genes: 1, n_measured: 0, n_changed: 0 },
  },
  padj_cutoff: 0.05, has_padj: true, n_diagram_genes: 5, n_measured: 2, n_changed: 1,
};

const nodes = () => screen.getAllByTestId("pathway-node");
const nodeBy = (label: string) => nodes().find((g) => g.dataset.label === label)!;

describe("PathwayDiagram", () => {
  it("draws each class with its own shape", () => {
    render(<PathwayDiagram model={MODEL} values={VALUES} />);
    const shapeOf = (cls: string) => nodes().find((g) => g.dataset.class === cls)!.querySelector("[data-shape]")!.getAttribute("data-shape");
    expect(shapeOf("Protein")).toBe("rounded");
    expect(shapeOf("Complex")).toBe("octagon");
    expect(shapeOf("EntitySet")).toBe("double");
    expect(shapeOf("Chemical")).toBe("ellipse");
    expect(shapeOf("Gene")).toBe("gene");
    expect(shapeOf("RNA")).toBe("rna");
    expect(shapeOf("ProcessNode")).toBe("process");
    expect(screen.getAllByTestId("reaction")).toHaveLength(2);
    expect(screen.getAllByTestId("compartment")).toHaveLength(1);
    expect(document.querySelector('[data-connector="INHIBITOR"] line')).toBeTruthy();
  });

  it("colours nodes by fold change: diverging about 0, grey when not measured, plain without genes", () => {
    render(<PathwayDiagram model={MODEL} values={VALUES} />);
    const palette = diagramPalette("light");
    const up = nodeBy("STAT1 [cytosol]").dataset.fill!;
    const down = nodeBy("IRF set").dataset.fill!;
    const mid = nodeBy("STAT1:IRF9").dataset.fill!;
    expect(up).toMatch(/^rgb\(/);
    expect(up).not.toBe(down);
    // RdBu: positive red-ish, negative blue-ish
    const [ur, , ub] = up.slice(4, -1).split(",").map(Number);
    const [dr, , db] = down.slice(4, -1).split(",").map(Number);
    expect(ur! > ub!).toBe(true);
    expect(db! > dr!).toBe(true);
    expect(mid).not.toBe(up);
    expect(nodeBy("SOCS1 gene").dataset.fill).toBe(palette.notMeasured);
    expect(nodeBy("ATP").dataset.fill).toBe(palette.noGenes);
    // Not coloured: no values
    expect(nodeFill(MODEL.nodes[0]!, null, null, palette)).toBe(palette.noGenes);
    const scale = lfcScale(VALUES, "RdBu", { preset: "none" })!;
    expect(scale.min).toBe(-2);
    expect(scale.max).toBe(2);
    expect(scale.centre).toBe(0);
  });

  it("shows a node's genes, values and counts on hover", () => {
    render(<PathwayDiagram model={MODEL} values={VALUES} />);
    fireEvent.mouseEnter(nodeBy("STAT1:IRF9"), { clientX: 10, clientY: 10 });
    const tip = screen.getByTestId("pathway-tooltip");
    expect(tip.textContent).toContain("mean log FC 0.50");
    expect(tip.textContent).toContain("1 of 2 changed (adjusted p < 0.05); 2 of 3 genes measured");
    expect(within(tip).getByText("STAT1").parentElement!.textContent).toContain("+2.00");
    expect(within(tip).getByText("IRF9").parentElement!.textContent).toContain("not measured");
    fireEvent.mouseLeave(nodeBy("STAT1:IRF9"));
    expect(screen.queryByTestId("pathway-tooltip")).toBeNull();
  });

  it("outlines a pathway drawn in its ancestor and marks its reactions", () => {
    render(<PathwayDiagram model={MODEL} values={VALUES} />);
    expect(screen.getByTestId("pathway-highlight").getAttribute("points")).toBe("5,5 300,5 300,100 5,100");
    const marked = screen.getAllByTestId("reaction").filter((r) => r.dataset.highlight === "true");
    expect(marked).toHaveLength(1);
  });

  it("credits Reactome with the licence and release", () => {
    render(<PathwayDiagram model={MODEL} values={VALUES} />);
    const line = screen.getByTestId("pathway-attribution");
    expect(line.textContent).toContain("Reactome (CC BY 4.0), release 98");
    expect(within(line).getByText("CC BY 4.0").getAttribute("href")).toBe("https://reactome.org/license");
    expect(within(line).getByText("Reactome").getAttribute("href")).toBe(MODEL.url);
    expect(screen.getByTestId("pathway-legend").textContent).toContain("n = 2 of 5 diagram genes measured; 1 changed");
  });

  it("opens a sub-pathway box", () => {
    const onOpen = vi.fn();
    render(<PathwayDiagram model={MODEL} values={VALUES} onOpenPathway={onOpen} />);
    fireEvent.click(nodeBy("Interleukin-6 signaling"));
    expect(onOpen).toHaveBeenCalledWith("R-HSA-1059683");
  });

  it("zooms with the buttons and fits again", () => {
    render(<PathwayDiagram model={MODEL} values={VALUES} />);
    const svg = screen.getByTestId("pathway-svg");
    const fit = svg.getAttribute("viewBox");
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(svg.getAttribute("viewBox")).not.toBe(fit);
    fireEvent.click(screen.getByRole("button", { name: "Fit the diagram" }));
    expect(svg.getAttribute("viewBox")).toBe(fit);
  });
});

describe("PathwayDiagramPanel", () => {
  beforeEach(() => {
    getReactomeDiagram.mockReset().mockResolvedValue(MODEL);
    getReactomeValues.mockReset().mockResolvedValue(VALUES);
    listDEResults.mockReset().mockResolvedValue({ results: [
      { result_id: "r2", label: "B: STIM vs CTRL (Wilcoxon)", method: "wilcoxon", created_at: "", n_tested: 1, has_pvalues: true, rule: "padj", n_sig_up: 0, n_sig_down: 0 },
    ], keep: 8, padj_max: 0.05, lfc_min: 0.25, logreg_top_n: 50 });
  });

  it("colours by the result the enrichment came from, and by another on request", async () => {
    render(<PathwayDiagramPanel datasetId="d1"
      open={{ stId: "R-HSA-1059683", term: "REACTOME_INTERLEUKIN_6_SIGNALING", source: { kind: "markers", column: "cell_type", group: "CD14 Mono" } }}
      onClose={() => {}} />);
    await waitFor(() => expect(getReactomeValues).toHaveBeenCalled());
    expect(getReactomeDiagram).toHaveBeenCalledWith("R-HSA-1059683");
    expect(getReactomeValues).toHaveBeenLastCalledWith("d1", "R-HSA-1059683", { kind: "markers", column: "cell_type", group: "CD14 Mono" });
    expect(await screen.findByText(/no diagram of its own/)).toBeTruthy();
    const select = screen.getByLabelText("Colour by") as HTMLSelectElement;
    await waitFor(() => expect(select.options).toHaveLength(3));
    fireEvent.change(select, { target: { value: "de:r2" } });
    await waitFor(() => expect(getReactomeValues).toHaveBeenLastCalledWith("d1", "R-HSA-1059683", { kind: "de", id: "r2" }));
  });

  it("without a source of its own takes the newest DE result", async () => {
    render(<PathwayDiagramPanel datasetId="d1" open={{ stId: "R-HSA-1059683", term: "X", source: null }} onClose={() => {}} />);
    await waitFor(() => expect(getReactomeValues).toHaveBeenLastCalledWith("d1", "R-HSA-1059683", { kind: "de", id: "r2" }));
  });

  it("an overview lists its sub-pathways to open instead", async () => {
    getReactomeDiagram.mockImplementation(async (id: string) => id === "R-HSA-913531"
      ? { ...MODEL, st_id: id, requested: id, too_broad: true, highlight: null, nodes: [],
          message: "Interferon Signaling is an overview of 2 sub-pathways with no reactions of its own to colour.",
          suggestions: [{ st_id: "R-HSA-909733", name: "Interferon alpha/beta signaling" }, { st_id: "R-HSA-877300", name: "Interferon gamma signaling" }] }
      : { ...MODEL, st_id: id, requested: id, highlight: null, message: "" });
    render(<PathwayDiagramPanel datasetId="d1" open={{ stId: "R-HSA-913531", term: "IFN", source: null }} onClose={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: "Interferon gamma signaling" }));
    await waitFor(() => expect(getReactomeDiagram).toHaveBeenLastCalledWith("R-HSA-877300"));
    fireEvent.click(await screen.findByRole("button", { name: /Back/ }));
    await waitFor(() => expect(getReactomeDiagram).toHaveBeenLastCalledWith("R-HSA-913531"));
  });

  it("shows the server's message when the diagram cannot be had", async () => {
    getReactomeDiagram.mockRejectedValue(new Error("API error 503: The Reactome diagram is not downloaded yet"));
    render(<PathwayDiagramPanel datasetId="d1" open={{ stId: "R-HSA-1", term: "X", source: null }} onClose={() => {}} />);
    expect((await screen.findByRole("alert")).textContent).toContain("not downloaded yet");
  });
});

describe("Diagram buttons on enrichment rows", () => {
  const ora = (term: string, source_id: string, source_url = ""): EnrichmentResult => ({
    term, pvalue: 0.001, adjusted_pvalue: 0.01, overlap_count: 3, gene_count: 30, genes: ["STAT1"],
    collection: "c2.cp.reactome", source_id, source_url,
  });

  it("show only on Reactome rows; KEGG and WikiPathways rows link out", () => {
    const onDiagram = vi.fn();
    const rows = [
      ora("REACTOME_IL6", "R-HSA-1059683"),
      ora("KEGG_MEDICUS_X", "N00001", "https://www.kegg.jp/entry/N00001"),
      ora("WP_Y", "WP100", "https://www.wikipathways.org/pathways/WP100"),
      ora("HALLMARK_Z", ""),
    ];
    render(<EnrichmentTermsTable results={rows} sortField="adj_pval" sortDir="asc" onSort={() => {}}
      onTermClick={() => {}} onAsk={() => {}} scoringTerm={null} onDiagram={onDiagram} />);
    const buttons = screen.getAllByRole("button", { name: /Diagram/ });
    expect(buttons).toHaveLength(1);
    fireEvent.click(buttons[0]!);
    expect(onDiagram).toHaveBeenCalledWith(rows[0]);
    expect(screen.getByRole("link", { name: /Open in KEGG/ }).getAttribute("href")).toBe("https://www.kegg.jp/entry/N00001");
    expect(screen.getByRole("link", { name: /Open in WikiPathways/ }).getAttribute("rel")).toContain("noopener");
    expect(screen.getAllByRole("link")).toHaveLength(2);
  });

  it("GSEA rows too, and none without a handler", () => {
    const term: GseaTerm = { term: "REACTOME_IL6", es: 0.5, nes: 1.8, pvalue: 0.001, fdr: 0.01, set_size: 20,
      leading_edge: ["STAT1"], collection: "c2.cp.reactome", source_id: "R-MMU-1059683" };
    const onDiagram = vi.fn();
    const { rerender } = render(<GseaTermsTable results={[term]} permutations={1000} sortField="fdr" sortDir="asc"
      onSort={() => {}} onTermClick={() => {}} onAsk={() => {}} scoringTerm={null} onDiagram={onDiagram} />);
    fireEvent.click(screen.getByRole("button", { name: /Diagram/ }));
    expect(onDiagram).toHaveBeenCalledWith(term);
    rerender(<GseaTermsTable results={[term]} permutations={1000} sortField="fdr" sortDir="asc"
      onSort={() => {}} onTermClick={() => {}} onAsk={() => {}} scoringTerm={null} />);
    expect(screen.queryByRole("button", { name: /Diagram/ })).toBeNull();
  });
});
