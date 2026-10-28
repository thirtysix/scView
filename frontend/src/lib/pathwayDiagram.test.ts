import { describe, it, expect } from "vitest";
import { fitViewBox, isReactomeId, lfcScale, nodeSummary, outboundLabel, sourceKind, wrapLabel, zoomViewBox } from "./pathwayDiagram";
import type { PathwayValues } from "@/api/pathways";

describe("pathway diagram helpers", () => {
  it("tells sources apart by MSigDB's exactSource", () => {
    expect(sourceKind("R-HSA-1059683")).toBe("reactome");
    expect(isReactomeId("R-MMU-1059683")).toBe(true);
    expect(isReactomeId("R-BTA-1")).toBe(false);
    expect(sourceKind("WP100")).toBe("wikipathways");
    expect(sourceKind("N00001")).toBe("kegg");
    expect(sourceKind("hsa04010")).toBe("kegg");
    expect(sourceKind("")).toBeNull();
    expect(outboundLabel("N00001", "https://www.kegg.jp/entry/N00001")).toBe("Open in KEGG");
    expect(outboundLabel("WP100", "javascript:alert(1)")).toBeNull();
  });

  it("wraps labels into the box and cuts the last line", () => {
    expect(wrapLabel("STAT1 [cytosol]", 80, 30)).toEqual(["STAT1 [cytosol]"]);
    expect(wrapLabel("STAT1 [cytosol]", 50, 30)).toEqual(["STAT1", "[cytosol]"]);
    const many = wrapLabel("a very long complex name that will never fit in this little box", 60, 26);
    expect(many).toHaveLength(2);
    expect(many[1]!.endsWith("…")).toBe(true);
    expect(wrapLabel("IFNA1:IFNAR1:IFNAR2", 60, 40).join("")).toBe("IFNA1:IFNAR1:IFNAR2");
    for (const line of wrapLabel("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 40, 60)) expect(line.length).toBeLessThanOrEqual(7);
  });

  it("scales fold changes symmetric about 0 even when all are one sign", () => {
    const v = { nodes: { a: { value: 0.5, n_genes: 1, n_measured: 1, n_changed: 0 }, b: { value: 2, n_genes: 1, n_measured: 1, n_changed: 1 } } } as unknown as PathwayValues;
    const s = lfcScale(v, "PuOr", { preset: "none" })!;
    expect([s.min, s.max, s.centre, s.diverging, s.colormap]).toEqual([-2, 2, 0, true, "PuOr"]);
    expect(lfcScale(null, "RdBu", { preset: "none" })).toBeNull();
  });

  it("says how many of a node's genes changed", () => {
    expect(nodeSummary({ value: 1, n_genes: 3, n_measured: 2, n_changed: 1 }, 0.05)).toBe("1 of 2 changed (adjusted p < 0.05); 2 of 3 genes measured");
    expect(nodeSummary({ value: null, n_genes: 1, n_measured: 0, n_changed: 0 }, 0.05)).toBe("None of its 1 gene measured");
    expect(nodeSummary({ value: 1, n_genes: 2, n_measured: 2, n_changed: null }, 0.05)).toBe("2 of 2 genes measured; no adjusted p-values");
  });

  it("zooms about a point within limits", () => {
    const vb = fitViewBox({ minX: 0, maxX: 100, minY: 0, maxY: 50 }, 0);
    const z = zoomViewBox(vb, 0.5, 50, 25, { min: 10, max: 1000 });
    expect(z).toEqual({ x: 25, y: 12.5, w: 50, h: 25 });
    expect(zoomViewBox(vb, 0.01, 0, 0, { min: 10, max: 1000 }).w).toBe(10);
  });
});
