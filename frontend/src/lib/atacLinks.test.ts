import { describe, expect, it } from "vitest";
import type { Coverage } from "@/api/atacFragments";
import type { LinkArc, LinksEstimate, LinksPage } from "@/api/atacLinks";
import { coverageFigure } from "./atacPlots";
import { arcColor, arcDepth, arcShapes, cellsText, countsText, estimateText, linksBody, parseGeneList } from "./atacLinks";

const arc = (peak: string, middle: number, score: number): LinkArc => ({
  peak, start: middle - 100, end: middle + 100, middle, tss: 5000, score, z: 3, p: 0.001, padj: 0.01, distance_to_tss: middle - 5000,
});

const cov: Coverage = {
  region: { chrom: "chr1", start: 0, end: 10000 },
  bins: 2, bin_size: 5000, x: [2500, 7500], group_by: null,
  tracks: [{ name: "All cells", n_cells: 5, n_fragments_in_window: 1, total_fragments: 1, values: [1, 2] }],
  ymax: 2, normalisation: "", notes: [], n_fragments: 1, max_window: 2000000,
  peaks: [{ start: 100, end: 200 }], genes: [], gene: null, genes_note: null, seconds: 0,
};

describe("peak-gene links text", () => {
  it("splits a pasted list on commas and white space, each gene once", () => {
    expect(parseGeneList(" LYZ, MS4A1\nCD3D;LYZ ")).toEqual(["LYZ", "MS4A1", "CD3D"]);
    expect(parseGeneList("  ")).toEqual([]);
  });

  it("sends a list only for a list and groups only with a column", () => {
    expect(linksBody("variable", "LYZ", "", [])).toEqual({ genes: "variable", cells_column: null, cells_value: null });
    expect(linksBody("list", "LYZ MS4A1", "celltype", ["B"])).toEqual({
      genes: "list", gene_list: ["LYZ", "MS4A1"], cells_column: "celltype", cells_value: ["B"],
    });
  });

  it("says a first genome download before the run", () => {
    const e: LinksEstimate = {
      genes: "variable", n_genes: 2000, n_asked: 2000, missing: [], n_cells: 11208, n_dataset_cells: 11208, n_peaks: 143887,
      estimate_seconds: 60, estimate_bytes: 1024 ** 3, estimate_disk_bytes: 800 * 1024 ** 2,
      genome: { genome: "GRCh38", source: "UCSC hg38", size: 835_393_456, downloaded: false },
    };
    expect(estimateText(e)).toContain("downloads the UCSC hg38 sequence (835 MB");
    expect(estimateText({ ...e, genome: { ...e.genome, downloaded: true } })).not.toContain("downloads");
    expect(estimateText(e)).toContain("2,000 genes, 11,208 cells");
    expect(estimateText(e)).not.toContain("rough");
    expect(estimateText({ ...e, n_cells: 300 })).toContain("rough for chosen groups");
  });

  it("counts what was left out", () => {
    const r = {
      counts: { n_links: 10, n_pairs_tested: 100, n_genes_tested: 5, n_links_padj: 7, n_genes_few_cells: 2, n_genes_few_peaks: 1,
                n_peaks_unknown: 51, n_genes_asked: 8, n_pairs_cis: 200, n_peaks_kept: 90, n_cells: 50 },
      params: { genes: "variable", cells_column: "celltype", cells_value: ["B", "T"] },
    } as unknown as LinksPage;
    const text = countsText(r);
    expect(text).toContain("10 links at p < 0.05 from 100 pairs");
    expect(text).toContain("3 genes not tested");
    expect(text).toContain("51 peaks left out");
    expect(cellsText(r)).toBe("celltype: B, T");
    expect(cellsText({ ...r, params: { genes: "variable" } } as LinksPage)).toBe("all cells");
  });
});

describe("arcs", () => {
  it("get deeper and darker with |score|, and the sign shows in colour and dash", () => {
    expect(arcDepth(0.6, 0.6)).toBeGreaterThan(arcDepth(0.1, 0.6));
    expect(arcColor(0.6, 0.6)).toBe("rgba(37,99,235,1.00)");
    expect(arcColor(-0.1, 0.6)).toMatch(/^rgba\(220,38,38,0\.\d+\)$/);
    const [weak, strong] = arcShapes([arc("up", 2000, 0.6), arc("down", 8000, -0.1)]);
    expect(weak!.dash).toBe("dash");
    expect(strong!.dash).toBe("solid");
    expect(strong!.apexY).toBeLessThan(weak!.apexY);
  });

  it("run from the peak's middle to the TSS along the row's top, the chosen one thicker", () => {
    const [only] = arcShapes([arc("up", 2000, 0.6)], "up");
    expect(only!.path.startsWith("M 2000 1 Q 3500 ")).toBe(true);
    expect(only!.path.endsWith(" 5000 1")).toBe(true);
    expect(only!.width).toBe(3);
  });

  it("add a row under the tracks with a hover point each, and an arc out of the window stays out", () => {
    const fig = coverageFigure(cov, () => "#000", {
      gene: "LYZ", arcs: [arc("up", 2000, 0.6), arc("down", 8000, -0.1), arc("far", 50_000, 0.3)], selectedPeak: "up",
    });
    const layout = fig.layout as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
    expect(layout.xaxis.anchor).toBe("y3");
    expect(layout.yaxis3.domain[0]).toBe(0);
    expect(layout.yaxis2.domain[0]).toBeGreaterThan(0);
    const paths = layout.shapes.filter((s: { type: string }) => s.type === "path");
    expect(paths).toHaveLength(2);
    expect(paths.map((s: { line: { dash: string } }) => s.line.dash)).toEqual(["dash", "solid"]);
    const hover = fig.data[fig.data.length - 1] as { yaxis: string; text: string[] };
    expect(hover.yaxis).toBe("y3");
    expect(hover.text[1]).toContain("LYZ ~ up");
    expect(layout.annotations.some((a: { text: string }) => a.text.includes("red dashed r < 0"))).toBe(true);
    const plain = coverageFigure(cov, () => "#000").layout as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
    expect(plain.xaxis.anchor).toBe("y2");
    expect(plain.yaxis2.domain[0]).toBe(0);
  });

  it("draw on a peak track alone when there are no tracks", () => {
    const fig = coverageFigure({ ...cov, tracks: [], x: [] }, () => "#000", { gene: "LYZ", arcs: [arc("up", 2000, 0.6)] });
    const layout = fig.layout as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
    expect(layout.xaxis.anchor).toBe("y2");
    expect(fig.data).toHaveLength(1);
    expect(layout.shapes.some((s: { type: string }) => s.type === "rect")).toBe(true);
  });
});
