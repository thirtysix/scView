import { describe, expect, it } from "vitest";
import type { PpiNode, PpiResult } from "@/api/ppi";
import {
  COMMUNITY_OTHER,
  COMMUNITY_SERIES,
  communityColour,
  logfcColour,
  maxAbsLogfc,
  nodeRadius,
  ppiCsvRow,
  ppiInputNote,
  ppiPText,
  ppiSummary,
  sortNodes,
} from "./ppi";

const node = (gene: string, degree: number, extra: Partial<PpiNode> = {}): PpiNode => ({
  gene, name: gene, string_id: `9606.${gene}`, degree, betweenness: 0, community: degree ? 1 : null,
  logfc: null, x: degree ? 0 : null, y: degree ? 0 : null, hub: false, ...extra,
});

const RESULT = {
  n_proteins: 84, n_links: 61, expected_links: 22.4, p_value: 1 / 1001, n_draws: 1000,
  truncated: false, max_genes: 500, n_given: 90, n_unmapped: 0, unmapped: [],
} as unknown as PpiResult;

describe("ppi text", () => {
  it("says p below the resolution of the draws, else three decimals", () => {
    expect(ppiPText(1 / 1001, 1000)).toBe("p < 0.001");
    expect(ppiPText(0.0423, 1000)).toBe("p = 0.042");
    expect(ppiPText(null, 1000)).toBe("p n/a");
  });

  it("summarises proteins, interactions, expected and p", () => {
    expect(ppiSummary(RESULT)).toBe("84 proteins, 61 interactions (22 expected by chance), p < 0.001");
    expect(ppiSummary({ ...RESULT, n_links: 1, expected_links: 0.94 })).toBe(
      "84 proteins, 1 interaction (0.9 expected by chance), p < 0.001");
  });

  it("names the cap and the genes STRING does not know", () => {
    expect(ppiInputNote(RESULT)).toBeNull();
    const unmapped = Array.from({ length: 10 }, (_, i) => `X${i}`);
    expect(ppiInputNote({ ...RESULT, truncated: true, n_given: 700, n_unmapped: 10, unmapped })).toBe(
      "only the first 500 of 700 genes are used; 10 genes not in STRING (X0, X1, X2, X3, X4, X5, X6, X7 and 2 more).");
  });
});

describe("ppi colours and sizes", () => {
  it("colours the four largest communities and greys the rest", () => {
    expect(communityColour(1)).toBe(COMMUNITY_SERIES[0]);
    expect(communityColour(4)).toBe(COMMUNITY_SERIES[3]);
    expect(communityColour(5)).toBe(COMMUNITY_OTHER);
    expect(communityColour(null)).toBe(COMMUNITY_OTHER);
  });

  it("diverges about 0: red up, blue down, light at 0", () => {
    expect(logfcColour(2, 2)).toBe("rgb(202,0,32)");
    expect(logfcColour(-5, 2)).toBe("rgb(5,113,176)");
    expect(logfcColour(0, 2)).toBe("rgb(226,232,240)");
    expect(logfcColour(null, 2)).toBe(COMMUNITY_OTHER);
    expect(maxAbsLogfc([node("A", 1, { logfc: -3 }), node("B", 1, { logfc: 1 })])).toBe(3);
    expect(maxAbsLogfc([node("A", 1)])).toBe(0.5);
  });

  it("grows nodes with degree", () => {
    expect(nodeRadius(0)).toBe(3);
    expect(nodeRadius(16)).toBeGreaterThan(nodeRadius(4));
  });
});

describe("ppi table", () => {
  const nodes = [node("B", 3, { logfc: 1 }), node("A", 3), node("C", 9, { logfc: -2 }), node("D", 0)];

  it("sorts by degree, ties by name, nulls last", () => {
    expect(sortNodes(nodes, "degree", "desc").map((n) => n.gene)).toEqual(["C", "A", "B", "D"]);
    expect(sortNodes(nodes, "logfc", "desc").map((n) => n.gene)).toEqual(["B", "C", "A", "D"]);
    expect(sortNodes(nodes, "community", "asc").map((n) => n.gene)).toEqual(["C", "A", "B", "D"]);
    expect(sortNodes(nodes, "gene", "asc").map((n) => n.gene)).toEqual(["A", "B", "C", "D"]);
  });

  it("writes a CSV row per protein", () => {
    expect(ppiCsvRow(node("C", 9, { logfc: -2, hub: true, betweenness: 0.5 }))).toEqual(
      ["C", "C", "9606.C", "9", "0.5000", "1", "-2.000", "yes"]);
    expect(ppiCsvRow(node("D", 0))).toEqual(["D", "D", "9606.D", "0", "0.0000", "", "", ""]);
  });
});
