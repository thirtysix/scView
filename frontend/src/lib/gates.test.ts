import { describe, it, expect } from "vitest";
import { decodeMask, type CellFilter } from "@/lib/cellFilter";
import { encodeMask, encodeSelection } from "@/lib/session";
import {
  bitsetAnd,
  bitsetAndNot,
  bitsetCount,
  bitsetFromIndices,
  bitsetFromMask,
  bitsetFromSelection,
  bitsetHas,
  bitsetIndices,
  bitsetOr,
  bitsetToSelection,
  bitsetToSet,
  combineName,
  defaultCombineParent,
  deleteGate,
  describeGate,
  evaluateGates,
  formatGatePct,
  fromSessionGate,
  fullBitset,
  gateCountsTable,
  gateNameError,
  gateTree,
  labelSlots,
  nextGateName,
  parseGateList,
  planLabelColumn,
  suggestGateColumn,
  toSessionGate,
  uniqueGateName,
  validateGates,
  type Bitset,
  type FilterCells,
  type Gate,
  type GateDef,
  type GateEval,
} from "@/lib/gates";

const N = 100;
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, k) => a + k);
const cellsGate = (id: string, name: string, cells: number[], parentId: string | null = null, n = N): Gate => ({
  id,
  name,
  parentId,
  def: { kind: "cells", selection: bitsetToSelection(bitsetFromIndices(cells, n)) },
  createdAt: "2026-10-21T00:00:00.000Z",
});
const filterDef: CellFilter = { op: "and", layer: "", items: [{ kind: "gene", gene: "ISG15", op: ">", value: 2 }] };
const filterGate = (id: string, name: string, parentId: string | null = null): Gate => ({
  id,
  name,
  parentId,
  def: { kind: "filter", filter: filterDef },
  createdAt: "",
});
const combo = (id: string, name: string, op: "and" | "or" | "not", operands: string[], parentId: string | null = null): Gate => ({
  id,
  name,
  parentId,
  def: { kind: "combine", op, operands },
  createdAt: "",
});
const noFilters = (): FilterCells => undefined;
const ok = (e: GateEval | undefined) => {
  if (e?.status !== "ok") throw new Error(`not ok: ${JSON.stringify(e)}`);
  return e;
};
const indicesOf = (b: Bitset) => Array.from(bitsetIndices(b));

describe("bitsets", () => {
  it("hold cell indices, ignoring ones out of range", () => {
    const b = bitsetFromIndices([0, 31, 32, 99, 100, -1, 2.5], N);
    expect(indicesOf(b)).toEqual([0, 31, 32, 99]);
    expect(bitsetCount(b)).toBe(4);
    expect(bitsetHas(b, 31)).toBe(true);
    expect(bitsetHas(b, 30)).toBe(false);
    expect([...bitsetToSet(b)]).toEqual([0, 31, 32, 99]);
  });

  it("and / or / and-not", () => {
    const a = bitsetFromIndices(range(0, 50), N);
    const b = bitsetFromIndices(range(40, 70), N);
    expect(bitsetCount(bitsetAnd(a, b))).toBe(10);
    expect(bitsetCount(bitsetOr(a, b))).toBe(70);
    expect(indicesOf(bitsetAndNot(b, a))).toEqual(range(50, 70));
  });

  it("a full set has exactly n cells, also when n is not a multiple of 32", () => {
    for (const n of [1, 31, 32, 33, 100, 1000]) expect(bitsetCount(fullBitset(n))).toBe(n);
    // the top bit of a word counts too
    expect(indicesOf(bitsetFromIndices([31, 63], 64))).toEqual([31, 63]);
  });

  it("read the backend's packed bitmask", () => {
    const cells = [0, 7, 8, 9, 63, 64, 99];
    const mask = encodeMask(cells, N);
    expect(indicesOf(bitsetFromMask(mask, N))).toEqual([...decodeMask(mask, N)]);
  });

  it("write a selection as session.ts encodeSelection does (indices or bitmask)", () => {
    const sparse = [3, 50];
    expect(bitsetToSelection(bitsetFromIndices(sparse, N))).toEqual(encodeSelection(sparse, N));
    const dense = range(0, 90);
    const sel = bitsetToSelection(bitsetFromIndices(dense, N));
    expect(sel).toEqual(encodeSelection(dense, N));
    expect("bitmask" in sel).toBe(true);
  });

  it("decode a saved selection, counting positions past the dataset's cells", () => {
    expect(bitsetFromSelection({ n_cells: 200, indices: [1, 150] }, N).ignored).toBe(1);
    const fromMask = bitsetFromSelection({ n_cells: 200, bitmask: encodeMask([5, 120, 180], 200) }, N);
    expect(indicesOf(fromMask.cells)).toEqual([5]);
    expect(fromMask.ignored).toBe(2);
  });
});

describe("evaluateGates", () => {
  it("keeps a child's cells inside its parent, with % of parent and % of total", () => {
    const gates = [cellsGate("mono", "Monocytes", range(0, 40)), cellsGate("isg", "ISG-high", range(30, 60), "mono")];
    const ev = evaluateGates(gates, N, noFilters);
    const mono = ok(ev.get("mono"));
    expect(mono.n).toBe(40);
    expect(mono.pctParent).toBeCloseTo(40);
    expect(mono.pctTotal).toBeCloseTo(40);
    const isg = ok(ev.get("isg"));
    expect(isg.n).toBe(10); // 30..39 only: the rest is outside Monocytes
    expect(isg.parentN).toBe(40);
    expect(isg.pctParent).toBeCloseTo(25);
    expect(isg.pctTotal).toBeCloseTo(10);
  });

  it("combines with AND, OR and NOT (the first minus the union of the others)", () => {
    const gates = [
      cellsGate("a", "A", range(0, 50)),
      cellsGate("b", "B", range(40, 70)),
      cellsGate("c", "C", range(0, 5)),
      combo("and", "A and B", "and", ["a", "b"]),
      combo("or", "A or B", "or", ["a", "b"]),
      combo("not", "A not (B or C)", "not", ["a", "b", "c"]),
    ];
    const ev = evaluateGates(gates, N, noFilters);
    expect(indicesOf(ok(ev.get("and")).cells)).toEqual(range(40, 50));
    expect(ok(ev.get("or")).n).toBe(70);
    expect(indicesOf(ok(ev.get("not")).cells)).toEqual(range(5, 40));
  });

  it("a combination inside a parent keeps only the parent's cells", () => {
    const gates = [
      cellsGate("mono", "Monocytes", range(0, 40)),
      cellsGate("isg", "ISG-high", range(30, 40), "mono"),
      combo("rest", "Monocytes not ISG-high", "not", ["mono", "isg"], "mono"),
    ];
    const rest = ok(evaluateGates(gates, N, noFilters).get("rest"));
    expect(rest.n).toBe(30);
    expect(rest.pctParent).toBeCloseTo(75);
  });

  it("an empty parent gives an empty child and no % of parent (not NaN)", () => {
    const gates = [cellsGate("e", "Empty", []), cellsGate("c", "Child", range(0, 10), "e")];
    const child = ok(evaluateGates(gates, N, noFilters).get("c"));
    expect(child.n).toBe(0);
    expect(child.pctParent).toBeNull();
    expect(formatGatePct(child.pctParent)).toBe("–");
  });

  it("waits for a filter gate's cells, and says why one failed, down the tree", () => {
    const gates = [
      cellsGate("mono", "Monocytes", range(0, 40)),
      filterGate("isg", "ISG-high", "mono"),
      combo("rest", "Monocytes not ISG-high", "not", ["mono", "isg"], "mono"),
    ];
    let ev = evaluateGates(gates, N, noFilters);
    expect(ev.get("isg")?.status).toBe("pending");
    expect(ev.get("rest")?.status).toBe("pending");
    ev = evaluateGates(gates, N, () => bitsetFromIndices(range(35, 80), N));
    expect(ok(ev.get("isg")).n).toBe(5);
    expect(ok(ev.get("rest")).n).toBe(35);
    ev = evaluateGates(gates, N, () => ({ error: "Gene 'ISG15' not found." }));
    const isg = ev.get("isg");
    expect(isg).toEqual({ status: "error", reason: "Gene 'ISG15' not found." });
    const rest = ev.get("rest");
    expect(rest?.status === "error" && rest.reason).toContain("ISG-high");
  });

  it("refuses cells saved on a different number of cells", () => {
    const gate = cellsGate("x", "X", [1, 2], null, 50);
    const ev = evaluateGates([gate], N, noFilters).get("x");
    expect(ev?.status).toBe("error");
  });

  it("is fast at 200k cells with a deep tree and combinations", () => {
    const n = 200_000;
    const gates: Gate[] = [];
    let parent: string | null = null;
    for (let k = 0; k < 20; k++) {
      const id = `g${k}`;
      const cells = range(k * 1000, n - k * 1000);
      gates.push({ ...cellsGate(id, `G${k}`, cells, parent, n) });
      parent = id;
    }
    for (let k = 0; k < 20; k++) gates.push(combo(`c${k}`, `C${k}`, k % 3 === 0 ? "and" : k % 3 === 1 ? "or" : "not", [`g${k}`, `g${19 - k}`]));
    evaluateGates(gates, n, noFilters); // decode once (cached per gate)
    const t0 = performance.now();
    const ev = evaluateGates(gates, n, noFilters);
    const ms = performance.now() - t0;
    expect(ok(ev.get("g19")).n).toBe(n - 2 * 19_000);
    expect(ms).toBeLessThan(250);
  });
});

describe("validateGates", () => {
  it("drops repeated ids, missing references and loops, and what depends on them", () => {
    const loopA = { ...cellsGate("la", "Loop A", [1]), parentId: "lb" };
    const loopB = { ...cellsGate("lb", "Loop B", [1]), parentId: "la" };
    const { gates, dropped } = validateGates([
      cellsGate("a", "A", [1, 2]),
      cellsGate("a", "A again", [3]),
      cellsGate("orphan", "Orphan", [1], "missing"),
      cellsGate("kid", "Kid of orphan", [1], "orphan"),
      combo("c", "Uses missing", "or", ["a", "nope"]),
      loopA,
      loopB,
      combo("self", "Self", "and", ["a", "self"]),
      combo("one", "One", "and", ["a"]),
    ]);
    expect(gates.map((g) => g.id)).toEqual(["a"]);
    expect(dropped.map((d) => d.name).sort()).toEqual(
      ["A again", "Kid of orphan", "Loop A", "Loop B", "One", "Orphan", "Self", "Uses missing"].sort(),
    );
  });

  it("catches a loop through a combination", () => {
    const { gates } = validateGates([cellsGate("p", "P", [1], "c"), combo("c", "C", "or", ["p", "q"]), cellsGate("q", "Q", [2])]);
    expect(gates.map((g) => g.id)).toEqual(["q"]);
  });
});

describe("tree helpers", () => {
  const gates = [
    cellsGate("t", "T cells", range(0, 50)),
    cellsGate("cd4", "CD4", range(0, 20), "t"),
    cellsGate("b", "B cells", range(60, 70)),
    cellsGate("cd8", "CD8", range(20, 40), "t"),
    cellsGate("naive", "Naive CD8", range(20, 25), "cd8"),
  ];

  it("lists parents before children, siblings in order", () => {
    expect(gateTree(gates).map((r) => [r.gate.name, r.depth, r.hasChildren])).toEqual([
      ["T cells", 0, true],
      ["CD4", 1, false],
      ["CD8", 1, true],
      ["Naive CD8", 2, false],
      ["B cells", 0, false],
    ]);
  });

  it("puts a NOT inside its first gate, AND / OR inside the nearest common parent", () => {
    expect(defaultCombineParent(gates, "not", ["cd8", "naive"])).toBe("cd8");
    expect(defaultCombineParent(gates, "or", ["cd4", "naive"])).toBe("t");
    expect(defaultCombineParent(gates, "and", ["cd4", "cd8"])).toBe("t");
    expect(defaultCombineParent(gates, "or", ["cd4", "b"])).toBeNull();
  });

  it("deletes a gate moving its children up, or with its subtree", () => {
    const ev = evaluateGates(gates, N, noFilters);
    const up = deleteGate(gates, "cd8", "reparent", ev);
    expect(up.removed).toEqual(["CD8"]);
    expect(up.gates.find((g) => g.id === "naive")?.parentId).toBe("t");
    const all = deleteGate(gates, "t", "subtree", ev);
    expect(all.removed).toEqual(["T cells", "CD4", "CD8", "Naive CD8"]);
    expect(all.gates.map((g) => g.id)).toEqual(["b"]);
  });

  it("a combination built from a deleted gate keeps the cells it had", () => {
    const withCombo = [...gates, combo("rest", "CD8 not naive", "not", ["cd8", "naive"], "cd8"), combo("x", "CD4 or B", "or", ["cd4", "b"])];
    const ev = evaluateGates(withCombo, N, noFilters);
    const res = deleteGate(withCombo, "naive", "subtree", ev);
    expect(res.removed).toEqual(["Naive CD8"]);
    expect(res.frozen).toEqual(["CD8 not naive"]);
    const frozen = res.gates.find((g) => g.id === "rest")!;
    expect(frozen.def.kind).toBe("cells");
    expect(ok(evaluateGates(res.gates, N, noFilters).get("rest")).n).toBe(15);
    // the untouched combination stays a combination
    expect(res.gates.find((g) => g.id === "x")?.def.kind).toBe("combine");
  });

  it("removes a combination whose cells are not known when its gate goes", () => {
    const withFilter = [cellsGate("a", "A", range(0, 10)), filterGate("f", "F"), combo("c", "A and F", "and", ["a", "f"])];
    const ev = evaluateGates(withFilter, N, noFilters); // F still pending
    const res = deleteGate(withFilter, "a", "subtree", ev);
    expect(res.removed).toEqual(["A", "A and F"]);
    expect(res.frozen).toEqual([]);
  });
});

describe("names", () => {
  const gates = [cellsGate("a", "Monocytes", [1])];
  it("checks gate names", () => {
    expect(gateNameError("", gates)).toBe("Name the gate.");
    expect(gateNameError("monocytes ", gates)).toMatch(/already exists/);
    expect(gateNameError("Monocytes", gates, "a")).toBeNull(); // renaming itself
    expect(gateNameError("unassigned", gates)).toMatch(/kept for cells in no gate/);
    expect(gateNameError("x".repeat(81), gates)).toMatch(/too long/);
    expect(gateNameError("bad​name", gates)).toMatch(/invisible/);
    expect(gateNameError("ISG-high", gates)).toBeNull();
  });

  it("makes names", () => {
    expect(nextGateName([cellsGate("a", "Gate 1", [1])])).toBe("Gate 2");
    expect(uniqueGateName("Monocytes", ["monocytes", "Monocytes 2"])).toBe("Monocytes 3");
    expect(combineName("not", ["Monocytes", "ISG-high"])).toBe("Monocytes not ISG-high");
    expect(combineName("not", ["A", "B", "C"])).toBe("A not (B or C)");
    expect(combineName("and", ["A", "B"])).toBe("A and B");
    expect(suggestGateColumn(["gates", "Gates_2"])).toBe("gates_3");
  });

  it("describes gates", () => {
    const all = [...gates, filterGate("f", "ISG-high", "a"), combo("c", "Mono not ISG", "not", ["a", "f"])];
    expect(describeGate(all[0]!, all)).toBe("1 selected cells, saved by position");
    expect(describeGate(all[1]!, all)).toBe("filter: ISG15 > 2");
    expect(describeGate(all[2]!, all)).toBe("Monocytes not ISG-high");
  });
});

describe("planLabelColumn", () => {
  it("labels each cell by its leaf gate, the deepest then the later one winning", () => {
    const gates = [
      cellsGate("t", "T", range(0, 50)),
      cellsGate("cd4", "CD4", range(0, 20), "t"),
      cellsGate("cd8", "CD8", range(15, 30), "t"), // 15..19 overlap CD4 (same depth: CD8 is later)
      cellsGate("b", "B", range(45, 60)), // 45..49 overlap T's cells, but T is not a leaf
      cellsGate("x", "Deep", range(10, 12), "cd4"),
    ];
    const ev = evaluateGates(gates, N, noFilters);
    const plan = planLabelColumn(gates, ev, N)!;
    const byLabel = Object.fromEntries(plan.groups.map((g) => [g.label, g.indices]));
    // CD4 has a gate inside it, so it is not a leaf; Deep (depth 2) is
    expect(Object.keys(byLabel)).toEqual(["Deep", "CD8", "B"]);
    expect(byLabel["Deep"]).toEqual([10, 11]);
    expect(byLabel["CD8"]).toEqual(range(15, 30));
    expect(byLabel["B"]).toEqual(range(45, 60));
    expect(plan.empty).toEqual([]);
  });

  it("reports overlap, unassigned cells and cells in a parent but no leaf", () => {
    const gates = [
      cellsGate("t", "T", range(0, 50)),
      cellsGate("cd4", "CD4", range(0, 20), "t"),
      cellsGate("cd8", "CD8", range(15, 30), "t"),
      cellsGate("deep", "Deep CD4", range(0, 5), "cd4"),
      cellsGate("b", "B", range(60, 70)),
    ];
    const plan = planLabelColumn(gates, evaluateGates(gates, N, noFilters), N)!;
    const byLabel = Object.fromEntries(plan.groups.map((g) => [g.label, g.indices]));
    expect(byLabel["CD8"]).toEqual(range(15, 30));
    expect(byLabel["Deep CD4"]).toEqual(range(0, 5));
    expect(byLabel["B"]).toEqual(range(60, 70));
    expect(plan.overlap).toBe(0); // CD4 is a parent, so only leaves count
    expect(plan.unassigned).toBe(N - 15 - 5 - 10);
    // T's cells 5..14 and 30..49 are in a parent gate but in no leaf
    expect(plan.inParentOnly).toBe(10 + 20);
  });

  it("deeper leaves win over shallower ones; a tie goes to the later leaf", () => {
    const gates = [
      cellsGate("a", "A", range(0, 30)),
      cellsGate("p", "P", range(0, 100)),
      cellsGate("deep", "Deep", range(20, 40), "p"),
      cellsGate("z", "Z", range(25, 35)),
    ];
    const plan = planLabelColumn(gates, evaluateGates(gates, N, noFilters), N)!;
    const byLabel = Object.fromEntries(plan.groups.map((g) => [g.label, g.indices]));
    expect(byLabel["Deep"]).toEqual(range(20, 40)); // depth 1 beats A and Z (depth 0)
    expect(byLabel["A"]).toEqual(range(0, 20));
    expect(byLabel["Z"]).toBeUndefined();
    expect(plan.empty).toEqual(["Z"]);
    expect(plan.overlap).toBe(15); // 20..34 are in two or three leaves
  });

  it("gives each labelled leaf the palette slot it takes in the column (Unassigned first)", () => {
    const gates = [
      cellsGate("p", "P", range(0, 50)),
      cellsGate("a", "A", range(0, 10), "p"),
      cellsGate("z", "Z", range(0, 10), "p"), // takes A's cells: A gets no label
      cellsGate("b", "B", range(60, 70)),
    ];
    const tree = gateTree(gates);
    const slots = labelSlots(tree, planLabelColumn(gates, evaluateGates(gates, N, noFilters), N));
    expect(Object.fromEntries(slots)).toEqual({ z: 1, b: 2 });
    // Still counting: leaves in list order.
    expect(Object.fromEntries(labelSlots(tree, null))).toEqual({ a: 1, z: 2, b: 3 });
  });

  it("needs every leaf counted", () => {
    const gates = [cellsGate("a", "A", range(0, 30)), filterGate("f", "F")];
    expect(planLabelColumn(gates, evaluateGates(gates, N, noFilters), N)).toBeNull();
  });
});

describe("gateCountsTable", () => {
  const gates = [cellsGate("mono", "Monocytes", range(0, 40)), cellsGate("isg", "ISG-high", range(30, 60), "mono")];
  const ev = evaluateGates(gates, N, noFilters);

  it("has gate, parent, definition, n, % parent and % total, all cells first", () => {
    const { headers, rows } = gateCountsTable(gates, ev, N);
    expect(headers).toEqual(["gate", "parent", "definition", "n", "pct_parent", "pct_total"]);
    expect(rows[0]).toEqual(["All cells", "", "every cell in the dataset", 100, 100, 100]);
    expect(rows[1]).toEqual(["Monocytes", "", "40 selected cells, saved by position", 40, 40, 40]);
    // saved as 30..59 (not trimmed to the parent here); its cells are the 10 inside Monocytes
    expect(rows[2]).toEqual(["ISG-high", "Monocytes", "30 selected cells, saved by position", 10, 25, 10]);
  });

  it("adds n and % of the parent's cells per sample, samples in natural order", () => {
    // donors: d1 for cells 0..49, d10 for 50..89, d2 for 90..98, cell 99 missing
    const values = range(0, N).map((i) => (i < 50 ? "d1" : i < 90 ? "d10" : i < 99 ? "d2" : null));
    const { headers, rows } = gateCountsTable(gates, ev, N, { column: "donor", values });
    expect(headers.slice(6)).toEqual(["n_d1", "pct_parent_d1", "n_d2", "pct_parent_d2", "n_d10", "pct_parent_d10"]);
    expect(rows[0]!.slice(6)).toEqual([50, 100, 9, 100, 40, 100]);
    // Monocytes = cells 0..39, all d1: 40 of the 50 d1 cells
    expect(rows[1]!.slice(6)).toEqual([40, 80, 0, 0, 0, 0]);
    // ISG-high = 30..39 inside Monocytes: 10 of Monocytes' 40 d1 cells; no d2 cells in the parent
    expect(rows[2]!.slice(6)).toEqual([10, 25, 0, "", 0, ""]);
  });

  it("leaves counts empty for a gate not counted", () => {
    const pending = [...gates, filterGate("f", "F")];
    const { rows } = gateCountsTable(pending, evaluateGates(pending, N, noFilters), N);
    expect(rows[3]).toEqual(["F", "", "filter: ISG15 > 2", "", "", ""]);
  });
});

describe("parseGateList", () => {
  it("reads what toSessionGate writes", () => {
    const gates = [
      cellsGate("a", "A", range(0, 90)),
      filterGate("f", "F", "a"),
      combo("c", "A not F", "not", ["a", "f"], "a"),
    ];
    const json = JSON.parse(JSON.stringify(gates.map(toSessionGate)));
    expect(json[1].parent_id).toBe("a");
    expect(parseGateList(json).map(fromSessionGate)).toEqual(gates.map((g) => ({ ...g })));
  });

  it("rejects malformed gates with the reason", () => {
    const good = toSessionGate(cellsGate("a", "A", [1]));
    expect(() => parseGateList({})).toThrow(/at most/);
    expect(() => parseGateList([{ ...good, id: "../x" }])).toThrow(/gates\[0\]\.id is not a gate id/);
    expect(() => parseGateList([{ ...good, name: "" }])).toThrow(/name/);
    expect(() => parseGateList([{ ...good, def: { kind: "lasso" } }])).toThrow(/kind/);
    expect(() => parseGateList([{ ...good, def: { kind: "combine", op: "xor", operands: ["a", "b"] } }])).toThrow(/op/);
    expect(() => parseGateList([{ ...good, def: { kind: "cells", selection: { n_cells: 100, bitmask: "AAAA" } } }])).toThrow(
      /bitmask/,
    );
    expect(() => parseGateList([{ ...good, def: { kind: "cells", selection: { n_cells: 10, indices: [10] } } }])).toThrow(
      /indices/,
    );
    expect(() => parseGateList([{ ...good, def: { kind: "filter", filter: { op: "and", items: [] } } }])).toThrow(/items/);
    const def: GateDef = { kind: "combine", op: "or", operands: ["a"] };
    expect(() => parseGateList([{ ...good, def }])).toThrow(/operands/);
  });
});
