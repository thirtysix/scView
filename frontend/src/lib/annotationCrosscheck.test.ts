import { describe, expect, it } from "vitest";
import type { CrosscheckLabelRow, CrosscheckResult } from "@/api/annotation";
import {
  agreementSummary,
  callsText,
  crosscheckColumns,
  defaultReference,
  defaultSelection,
  diagonalOrder,
  pairMatrix,
  sankeyData,
  sortLabelTable,
  tableMatrix,
  validCrosscheckName,
} from "@/lib/annotationCrosscheck";

const col = (name: string, n_unique: number, dtype = "category") => ({ name, dtype, n_unique });

const dataset = {
  n_cells: 1000,
  active_clustering: "leiden",
  user_annotation_columns: [],
  obs_columns: [
    col("leiden", 10),
    col("cell_type_llm", 8),
    col("cell_type_ct", 12),
    col("Donor", 8),
    col("n_genes", 900, "float64"),
    col("is_doublet", 2, "bool"),
    col("scview_conf_cell_type_flag", 4),
  ],
};

describe("columns and defaults", () => {
  it("offers categorical groupings, not numbers, flags or confidence outputs", () => {
    const names = crosscheckColumns(dataset).map((c) => c.name);
    expect(names.sort()).toEqual(["Donor", "cell_type_ct", "cell_type_llm", "leiden"]);
  });

  it("preselects what annotation methods made plus the active clustering", () => {
    const available = crosscheckColumns(dataset);
    const info = { suggested: { cell_type_llm: "LLM", cell_type_ct: "CellTypist" }, active_clustering: "leiden" };
    const picked = defaultSelection(available, info);
    expect(new Set(picked)).toEqual(new Set(["cell_type_llm", "cell_type_ct", "leiden"]));
    expect(picked).not.toContain("Donor");
    // The reference is an annotation, not the clustering.
    expect(defaultReference(picked, available, info)).not.toBe("leiden");
  });

  it("tops a lone suggestion up to two columns and caps at the maximum", () => {
    const available = crosscheckColumns(dataset);
    expect(defaultSelection(available, { suggested: {}, active_clustering: "leiden" })).toHaveLength(2);
    const all = { suggested: Object.fromEntries(available.map((c) => [c.name, "LLM"])), active_clustering: null };
    expect(defaultSelection(available, all, 3)).toHaveLength(3);
  });

  it("falls back to the first column when every pick is a clustering", () => {
    const available = crosscheckColumns(dataset);
    expect(defaultReference(["leiden"], available, { suggested: { leiden: "clustering" } })).toBe("leiden");
  });
});

describe("diagonalOrder", () => {
  it("puts each other label under the reference label holding most of its cells", () => {
    const c = {
      rows: ["T", "B", "Mono"],
      cols: ["m1", "b1", "t2", "t1"],
      counts: { T: { t1: 50, t2: 20 }, B: { b1: 30, t2: 1 }, Mono: { m1: 40 } },
    };
    const { rows, cols } = diagonalOrder(c);
    expect(rows).toEqual(["T", "B", "Mono"]);
    // t1 (50) and t2 (20) under T, the bigger first; b1 under B; m1 under Mono.
    expect(cols).toEqual(["t1", "t2", "b1", "m1"]);
    // Each row's largest entry lies on the block diagonal: never left of an earlier row's.
    const z = tableMatrix(c.counts, rows, cols);
    const argmax = z.map((row) => row.indexOf(Math.max(...row)));
    expect(argmax).toEqual([...argmax].sort((a, b) => a - b));
  });

  it("keeps a label with no overlap at the end", () => {
    const { cols } = diagonalOrder({ rows: ["A"], cols: ["none", "a"], counts: { A: { a: 3 } } });
    expect(cols).toEqual(["a", "none"]);
  });
});

describe("matrices and the Sankey", () => {
  it("reads label-keyed tables in the given order with zeros for missing entries", () => {
    expect(tableMatrix({ A: { x: 2 } }, ["A", "B"], ["y", "x"])).toEqual([
      [0, 2],
      [0, 0],
    ]);
    expect(pairMatrix({ a: { a: 1, b: 0.5 }, b: { a: 0.5, b: 1 } }, ["b", "a"])).toEqual([
      [1, 0.5],
      [0.5, 1],
    ]);
  });

  it("links every non-zero cell from the reference side to the other side", () => {
    const s = sankeyData({ counts: { A: { x: 3, y: 1 }, B: { y: 4 } } }, ["A", "B"], ["x", "y"]);
    expect(s.labels).toEqual(["A", "B", "x", "y"]);
    const links = s.source.map((src, i) => `${s.labels[src]}>${s.labels[s.target[i]!]}:${s.value[i]}`);
    expect(links.sort()).toEqual(["A>x:3", "A>y:1", "B>y:4"]);
    // Totals out of the reference side equal the cells counted.
    expect(s.value.reduce((a, b) => a + b, 0)).toBe(8);
  });
});

const row = (n: number, mean: number | null, disagreeing: string[] = []): CrosscheckLabelRow => ({
  n_cells: n,
  mean_agreement: mean,
  disagreeing,
  annotators: {},
});

describe("sortLabelTable", () => {
  // Keys like "0" and "10" are listed by a browser in numeric order, whatever the reference's order.
  const labels = { "10": row(5, 0.2, ["x"]), "0": row(50, 0.9), "2": row(20, 0.2) };
  const order = ["2", "10", "0"];

  it("keeps the reference's order, not the object's key order", () => {
    expect(sortLabelTable(labels, order, "label", false).map((r) => r.label)).toEqual(["0", "10", "2"]);
    expect(sortLabelTable(labels, order, "mean_agreement", false).map((r) => r.label)).toEqual(["2", "10", "0"]);
  });

  it("sorts by cells and disagreement, ties in the reference's order", () => {
    expect(sortLabelTable(labels, order, "n_cells", true).map((r) => r.label)).toEqual(["0", "2", "10"]);
    expect(sortLabelTable(labels, order, "disagreeing", true).map((r) => r.label)).toEqual(["10", "2", "0"]);
  });
});

describe("text", () => {
  it("says what a column calls a label's cells and where the rest map", () => {
    const r = {
      label: "CD8 T",
      ...row(10, 0.2, ["ms"]),
      annotators: {
        ms: {
          agreement: 0.2,
          disagrees: true,
          calls: [
            { label: "NK cell", share: 0.8, maps_to: "NK" },
            { label: "CD8+ T cell", share: 0.2, maps_to: "CD8 T" },
          ],
        },
      },
    };
    expect(callsText(r, "ms")).toBe("NK cell (80.0%, → NK), CD8+ T cell (20.0%)");
    expect(callsText(r, "missing")).toBe("–");
  });

  const result = (over: Partial<CrosscheckResult>): CrosscheckResult =>
    ({
      reference: "cell_type",
      mean_agreement: 0.75,
      n_compared: 1000,
      exact_name_agreement: 0,
      n_unresolved: 0,
      comparisons: { other: { exact_name: { n_shared_labels: 0, shared_labels: [], agreement: 0 } } },
      ...over,
    }) as unknown as CrosscheckResult;

  it("keeps same-name agreement apart and explains a zero from disjoint names", () => {
    const text = agreementSummary(result({}));
    expect(text).toContain("75.0% of 1,000 cells");
    expect(text).toContain("share no label names");
    const shared = agreementSummary(
      result({
        exact_name_agreement: 0.4,
        n_unresolved: 12,
        comparisons: {
          other: { exact_name: { n_shared_labels: 2, shared_labels: ["B", "NK"], agreement: 0.4 } },
        } as unknown as CrosscheckResult["comparisons"],
      }),
    );
    expect(shared).toContain("Same label name: 40.0%");
    expect(shared).toContain("12 cells have no majority");
  });

  it("accepts the names the backend accepts", () => {
    expect(validCrosscheckName("crosscheck")).toBe(true);
    expect(validCrosscheckName("xc_2.v-1")).toBe(true);
    expect(validCrosscheckName("bad name")).toBe(false);
    expect(validCrosscheckName("_x")).toBe(false);
    expect(validCrosscheckName("a".repeat(61))).toBe(false);
  });
});
