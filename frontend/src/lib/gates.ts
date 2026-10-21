/**
 * Cell gates (the toolbar's Gates panel): named selections that nest like
 * flow-cytometry gates. A gate is drawn cells (a lasso or legend pick, saved by
 * position), a cell filter (evaluated by the backend, `POST /cells/filter`) or
 * an AND / OR / NOT of other gates. Its cells are its own cells that are also
 * in its parent gate, so a child is always drawn inside its parent and
 * "% of parent" means what it says.
 *
 * Everything here is pure; cells are bitsets (one bit per cell in 32-bit
 * words), so a tree of gates over 200k cells evaluates in a few milliseconds.
 */
import { describeFilter, parseCellFilter, type CellFilter } from "@/lib/cellFilter";
import { UNASSIGNED, LABEL_MAX } from "@/lib/labels";

export type CombineOp = "and" | "or" | "not";
export const COMBINE_OPS: readonly CombineOp[] = ["and", "or", "not"];

/** Cells by position, as a session file's selection holds them: indices or a packed bitmask. */
export type CellsSelection = { n_cells: number; indices: number[] } | { n_cells: number; bitmask: string };

export type GateDef =
  | { kind: "cells"; selection: CellsSelection }
  | { kind: "filter"; filter: CellFilter }
  /** "not": the first operand minus the union of the others */
  | { kind: "combine"; op: CombineOp; operands: string[] };

export interface Gate {
  id: string;
  name: string;
  /** The gate it is drawn inside; null for a top-level gate */
  parentId: string | null;
  def: GateDef;
  /** ISO timestamp */
  createdAt: string;
}

/** Gates kept per dataset (the backend's label column takes at most 100 labels). */
export const MAX_GATES = 200;
const MAX_CELLS = 50_000_000;
const ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
// Control, invisible format and line/paragraph separators (the rule lib/labels.ts applies to labels).
const INVISIBLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Zl}\p{Zp}]/u;

let counter = 0;
/** No crypto.randomUUID: it is missing outside secure contexts (http on a LAN address). */
export const newGateId = () =>
  `g-${Date.now().toString(36)}-${(counter++).toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

// ---------------------------------------------------------------------------
// Bitsets: cell i is bit i % 32 of word i >> 5
// ---------------------------------------------------------------------------

export interface Bitset {
  readonly n: number;
  readonly words: Uint32Array;
}

const wordCount = (n: number) => Math.ceil(n / 32);

export function emptyBitset(n: number): Bitset {
  return { n, words: new Uint32Array(wordCount(n)) };
}

/** Every cell. */
export function fullBitset(n: number): Bitset {
  const b = emptyBitset(n);
  b.words.fill(0xffffffff);
  clearTail(b);
  return b;
}

/** Bits at or past `n` in the last word are always clear. */
function clearTail(b: Bitset) {
  const extra = b.n & 31;
  if (extra && b.words.length) b.words[b.words.length - 1]! &= (1 << extra) - 1;
}

/** Indices outside 0..n-1 (or not whole numbers) are left out. */
export function bitsetFromIndices(indices: Iterable<number>, n: number): Bitset {
  const b = emptyBitset(n);
  for (const i of indices) {
    if (Number.isInteger(i) && i >= 0 && i < n) b.words[i >>> 5]! |= 1 << (i & 31);
  }
  return b;
}

/** A packed bitmask in the backend's layout (base64; cell i is bit i % 8 of byte i // 8). */
export function bitsetFromMask(base64: string, n: number): Bitset {
  const bytes = atob(base64);
  const b = emptyBitset(n);
  const nBytes = Math.min(bytes.length, Math.ceil(n / 8));
  for (let i = 0; i < nBytes; i++) {
    const byte = bytes.charCodeAt(i);
    if (byte) b.words[i >>> 2]! |= byte << ((i & 3) * 8);
  }
  clearTail(b);
  return b;
}

/** The cells of a saved selection that exist among `n` cells; `ignored` counts the rest. */
export function bitsetFromSelection(sel: CellsSelection, n: number): { cells: Bitset; ignored: number } {
  if ("indices" in sel) {
    const cells = bitsetFromIndices(sel.indices, n);
    return { cells, ignored: sel.indices.length - bitsetCount(cells) };
  }
  const all = bitsetFromMask(sel.bitmask, Math.max(n, sel.n_cells));
  const cells = bitsetFromMask(sel.bitmask, n);
  return { cells, ignored: bitsetCount(all) - bitsetCount(cells) };
}

function popcount(x: number): number {
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
}

export function bitsetCount(b: Bitset): number {
  let n = 0;
  for (let w = 0; w < b.words.length; w++) if (b.words[w]) n += popcount(b.words[w]!);
  return n;
}

function combineWords(a: Bitset, b: Bitset, f: (x: number, y: number) => number): Bitset {
  const out = emptyBitset(a.n);
  for (let w = 0; w < out.words.length; w++) out.words[w] = f(a.words[w]!, b.words[w] ?? 0);
  return out;
}

export const bitsetAnd = (a: Bitset, b: Bitset) => combineWords(a, b, (x, y) => x & y);
export const bitsetOr = (a: Bitset, b: Bitset) => combineWords(a, b, (x, y) => x | y);
/** Cells in `a` and not in `b`. */
export const bitsetAndNot = (a: Bitset, b: Bitset) => combineWords(a, b, (x, y) => x & ~y);

export function bitsetHas(b: Bitset, i: number): boolean {
  return i >= 0 && i < b.n && (b.words[i >>> 5]! & (1 << (i & 31))) !== 0;
}

/** Call `f` with each cell index, in order. */
export function forEachCell(b: Bitset, f: (i: number) => void) {
  for (let w = 0; w < b.words.length; w++) {
    let x = b.words[w]! | 0;
    while (x !== 0) {
      const low = x & -x;
      f(w * 32 + 31 - Math.clz32(low));
      x ^= low;
    }
  }
}

/** The cell indices, ascending. */
export function bitsetIndices(b: Bitset): Uint32Array {
  const out = new Uint32Array(bitsetCount(b));
  let k = 0;
  forEachCell(b, (i) => {
    out[k++] = i;
  });
  return out;
}

/** As the selection store holds cells. */
export function bitsetToSet(b: Bitset): Set<number> {
  const out = new Set<number>();
  forEachCell(b, (i) => out.add(i));
  return out;
}

/** Indices or a bitmask, whichever is shorter: the rule (and output) of session.ts `encodeSelection`. */
export function bitsetToSelection(b: Bitset): CellsSelection {
  const indices = bitsetIndices(b);
  let listChars = 0;
  for (let k = 0; k < indices.length; k++) listChars += String(indices[k]).length + 1;
  const nBytes = Math.ceil(b.n / 8);
  if (listChars <= Math.ceil(nBytes / 3) * 4) return { n_cells: b.n, indices: Array.from(indices) };
  const bytes = new Uint8Array(nBytes);
  for (let i = 0; i < nBytes; i++) bytes[i] = (b.words[i >>> 2]! >>> ((i & 3) * 8)) & 0xff;
  let text = "";
  for (let start = 0; start < bytes.length; start += 0x8000) {
    text += String.fromCharCode(...bytes.subarray(start, start + 0x8000));
  }
  return { n_cells: b.n, bitmask: btoa(text) };
}

// ---------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------

/** The gates a gate's cells depend on: its parent and, for a combination, its operands. */
export function gateDependencies(g: Gate): string[] {
  const deps = g.parentId ? [g.parentId] : [];
  return g.def.kind === "combine" ? [...deps, ...g.def.operands] : deps;
}

export interface TreeRow {
  gate: Gate;
  depth: number;
  hasChildren: boolean;
}

/** Depth-first, parents before children, siblings in the order they were made. */
export function gateTree(gates: readonly Gate[]): TreeRow[] {
  const ids = new Set(gates.map((g) => g.id));
  const children = new Map<string | null, Gate[]>();
  for (const g of gates) {
    const parent = g.parentId && ids.has(g.parentId) && g.parentId !== g.id ? g.parentId : null;
    children.set(parent, [...(children.get(parent) ?? []), g]);
  }
  const rows: TreeRow[] = [];
  const seen = new Set<string>();
  const walk = (parent: string | null, depth: number) => {
    for (const g of children.get(parent) ?? []) {
      if (seen.has(g.id)) continue;
      seen.add(g.id);
      rows.push({ gate: g, depth, hasChildren: (children.get(g.id)?.length ?? 0) > 0 });
      walk(g.id, depth + 1);
    }
  };
  walk(null, 0);
  // A parent loop never reaches the top: list such gates rather than lose them.
  for (const g of gates) if (!seen.has(g.id)) rows.push({ gate: g, depth: 0, hasChildren: false });
  return rows;
}

/** The parents above a gate, nearest first (never itself; stops at a loop). */
export function ancestors(gates: readonly Gate[], id: string): string[] {
  const byId = new Map(gates.map((g) => [g.id, g]));
  const out: string[] = [];
  let at = byId.get(id)?.parentId ?? null;
  while (at && byId.has(at) && at !== id && !out.includes(at)) {
    out.push(at);
    at = byId.get(at)!.parentId;
  }
  return out;
}

/** A gate and every gate drawn inside it, at any depth. */
export function subtree(gates: readonly Gate[], id: string): string[] {
  const out = [id];
  for (let k = 0; k < out.length; k++) {
    for (const g of gates) if (g.parentId === out[k] && !out.includes(g.id)) out.push(g.id);
  }
  return out;
}

/** Gates with no gate inside them. */
export function leafGates(gates: readonly Gate[]): Gate[] {
  const parents = new Set(gates.map((g) => g.parentId).filter(Boolean));
  return gates.filter((g) => !parents.has(g.id));
}

/** Combined gates that use `id` as an operand. */
export function dependents(gates: readonly Gate[], id: string): Gate[] {
  return gates.filter((g) => g.def.kind === "combine" && g.def.operands.includes(id));
}

export interface DroppedGate {
  name: string;
  reason: string;
}

/**
 * The usable gates, in order: a repeated id, a reference to a gate that is not
 * there, a combination with too few operands or a gate that depends on itself
 * is dropped, and so is everything that depends on a dropped gate.
 */
export function validateGates(gates: readonly Gate[]): { gates: Gate[]; dropped: DroppedGate[] } {
  const dropped: DroppedGate[] = [];
  const kept: Gate[] = [];
  const seen = new Set<string>();
  for (const g of gates) {
    if (seen.has(g.id)) {
      dropped.push({ name: g.name, reason: "another gate has the same id" });
      continue;
    }
    seen.add(g.id);
    if (g.def.kind === "combine") {
      const min = 2;
      if (g.def.operands.length < min) {
        dropped.push({ name: g.name, reason: `a combination needs at least ${min} gates` });
        continue;
      }
    }
    kept.push(g);
  }
  // Missing references, then loops; repeat while dropping one drops others.
  let current = kept;
  for (;;) {
    const ids = new Set(current.map((g) => g.id));
    const next = current.filter((g) => {
      const missing = gateDependencies(g).find((d) => !ids.has(d) || d === g.id);
      if (missing === undefined) return true;
      dropped.push({
        name: g.name,
        reason: missing === g.id ? "it refers to itself" : "it refers to a gate that is not there",
      });
      return false;
    });
    const loops = findLoops(next);
    const result = next.filter((g) => {
      if (!loops.has(g.id)) return true;
      dropped.push({ name: g.name, reason: "it depends on itself through other gates" });
      return false;
    });
    if (result.length === current.length) return { gates: result, dropped };
    current = result;
  }
}

/** Ids of gates on a dependency loop. */
function findLoops(gates: readonly Gate[]): Set<string> {
  const byId = new Map(gates.map((g) => [g.id, g]));
  const state = new Map<string, 1 | 2>(); // 1 visiting, 2 done
  const onLoop = new Set<string>();
  const visit = (id: string, path: string[]) => {
    const s = state.get(id);
    if (s === 2) return;
    if (s === 1) {
      for (const p of path.slice(path.indexOf(id))) onLoop.add(p);
      return;
    }
    state.set(id, 1);
    const g = byId.get(id);
    for (const d of g ? gateDependencies(g) : []) if (byId.has(d)) visit(d, [...path, id]);
    state.set(id, 2);
  };
  for (const g of gates) visit(g.id, []);
  return onLoop;
}

// ---------------------------------------------------------------------------
// Cells, counts and percentages
// ---------------------------------------------------------------------------

export type GateEval =
  | {
      status: "ok";
      cells: Bitset;
      n: number;
      /** Cells in the parent gate (all cells for a top-level gate) */
      parentN: number;
      /** n / parentN x 100; null when the parent has no cells */
      pctParent: number | null;
      /** n / all cells x 100 */
      pctTotal: number | null;
    }
  | { status: "pending" }
  | { status: "error"; reason: string };

/** and: cells in every part; or: in any; not: in the first and in none of the others. */
export function combineCells(op: CombineOp, parts: readonly Bitset[]): Bitset {
  const [first, ...rest] = parts;
  if (!first) throw new Error("nothing to combine");
  if (rest.length === 0) return first;
  if (op === "and") return rest.reduce(bitsetAnd, first);
  if (op === "or") return rest.reduce(bitsetOr, first);
  return bitsetAndNot(first, rest.slice(1).reduce(bitsetOr, rest[0]!));
}

/** A filter gate's cells as the backend found them; undefined while still being fetched. */
export type FilterCells = Bitset | { error: string } | undefined;

/** n / of x 100, or null for an empty denominator (never NaN). */
export function pctOf(n: number, of: number): number | null {
  return of > 0 ? (100 * n) / of : null;
}

// Decoded cell lists, per gate object (gates are replaced, never changed, so a
// changed gate is a new key).
const decoded = new WeakMap<Gate, Bitset>();

function savedCells(g: Gate & { def: { kind: "cells" } }, n: number): Bitset {
  const hit = decoded.get(g);
  if (hit && hit.n === n) return hit;
  const { cells } = bitsetFromSelection(g.def.selection, n);
  decoded.set(g, cells);
  return cells;
}

/**
 * Every gate's cells and counts. A gate's cells are its own cells (saved,
 * filtered or combined from other gates' cells) that are in its parent's;
 * each gate is computed once, after what it depends on.
 */
export function evaluateGates(
  gates: readonly Gate[],
  nCells: number,
  filterCells: (gate: Gate) => FilterCells,
): Map<string, GateEval> {
  const byId = new Map(gates.map((g) => [g.id, g]));
  const out = new Map<string, GateEval>();
  const visiting = new Set<string>();
  const nameOf = (id: string) => byId.get(id)?.name ?? "a deleted gate";

  const evaluate = (id: string): GateEval => {
    const done = out.get(id);
    if (done) return done;
    const g = byId.get(id);
    if (!g) return { status: "error", reason: "the gate is not there" };
    if (visiting.has(id)) return { status: "error", reason: "it depends on itself" };
    visiting.add(id);
    const result = compute(g);
    visiting.delete(id);
    out.set(id, result);
    return result;
  };

  const compute = (g: Gate): GateEval => {
    let own: Bitset;
    if (g.def.kind === "cells") {
      if (g.def.selection.n_cells !== nCells) {
        return {
          status: "error",
          reason: `it was saved on ${g.def.selection.n_cells.toLocaleString()} cells; the dataset has ${nCells.toLocaleString()}`,
        };
      }
      own = savedCells(g as Gate & { def: { kind: "cells" } }, nCells);
    } else if (g.def.kind === "filter") {
      const found = filterCells(g);
      if (found === undefined) return { status: "pending" };
      if ("error" in found) return { status: "error", reason: found.error };
      own = found;
    } else {
      const parts: Bitset[] = [];
      for (const op of g.def.operands) {
        const e = evaluate(op);
        if (e.status === "pending") return e;
        if (e.status === "error") return { status: "error", reason: `it uses “${nameOf(op)}”: ${e.reason}` };
        parts.push(e.cells);
      }
      if (parts.length === 0) return { status: "error", reason: "it combines no gates" };
      own = combineCells(g.def.op, parts);
    }

    let cells = own;
    let parentN = nCells;
    if (g.parentId) {
      const p = evaluate(g.parentId);
      if (p.status === "pending") return p;
      if (p.status === "error") return { status: "error", reason: `it is inside “${nameOf(g.parentId)}”: ${p.reason}` };
      cells = bitsetAnd(own, p.cells);
      parentN = p.n;
    }
    const n = bitsetCount(cells);
    return { status: "ok", cells, n, parentN, pctParent: pctOf(n, parentN), pctTotal: pctOf(n, nCells) };
  };

  for (const g of gates) evaluate(g.id);
  return out;
}

/** "12.3%", "0.04%", "100%"; an empty denominator reads "–". */
export function formatGatePct(pct: number | null): string {
  if (pct == null) return "–";
  if (pct === 0 || pct === 100) return `${pct}%`;
  return `${Number(pct.toPrecision(pct < 1 ? 2 : 3))}%`;
}

/** Cells of `cells` that are not in `parent`: what saving inside a parent leaves out. */
export function cellsOutside(cells: Bitset, parent: Bitset): number {
  return bitsetCount(bitsetAndNot(cells, parent));
}

// ---------------------------------------------------------------------------
// Names and descriptions
// ---------------------------------------------------------------------------

/** Why `name` can't name a gate among `gates` (other than `exceptId`), or null. */
export function gateNameError(name: string, gates: readonly Gate[], exceptId?: string): string | null {
  const v = name.trim();
  if (!v) return "Name the gate.";
  if (v.length > LABEL_MAX) return `Gate name is too long (max ${LABEL_MAX} characters).`;
  if (INVISIBLE.test(v)) return "Gate name must not contain control or invisible characters.";
  const lower = v.toLowerCase();
  if (lower === UNASSIGNED.toLowerCase()) return `“${UNASSIGNED}” is kept for cells in no gate.`;
  if (gates.some((g) => g.id !== exceptId && g.name.trim().toLowerCase() === lower)) {
    return `A gate named “${v}” already exists.`;
  }
  return null;
}

/** `base`, else `base 2`, `base 3`, ... not taken (ignoring case), within the length limit. */
export function uniqueGateName(base: string, taken: readonly string[]): string {
  const used = new Set(taken.map((t) => t.trim().toLowerCase()));
  const clean = base.trim().slice(0, LABEL_MAX) || "Gate";
  if (!used.has(clean.toLowerCase()) && clean.toLowerCase() !== UNASSIGNED.toLowerCase()) return clean;
  for (let i = 2; ; i++) {
    const suffix = ` ${i}`;
    const name = `${clean.slice(0, LABEL_MAX - suffix.length)}${suffix}`;
    if (!used.has(name.toLowerCase())) return name;
  }
}

/** "Gate 1", "Gate 2", ...: the first free one. */
export function nextGateName(gates: readonly Gate[]): string {
  const used = new Set(gates.map((g) => g.name.trim().toLowerCase()));
  for (let i = 1; ; i++) if (!used.has(`gate ${i}`)) return `Gate ${i}`;
}

/** "A and B", "A or B", "A not B", "A not (B or C)". */
export function combineName(op: CombineOp, names: readonly string[]): string {
  if (op === "not") {
    const [a = "", ...rest] = names;
    const others = rest.length > 1 ? `(${rest.join(" or ")})` : (rest[0] ?? "");
    return `${a} not ${others}`.trim();
  }
  return names.join(` ${op} `);
}

/** What a gate is, in words (the CSV's definition column and the row tooltip). */
export function describeGate(g: Gate, gates: readonly Gate[]): string {
  if (g.def.kind === "cells") {
    const sel = g.def.selection;
    const n = "indices" in sel ? sel.indices.length : bitsetCount(bitsetFromMask(sel.bitmask, sel.n_cells));
    return `${n.toLocaleString()} selected cells, saved by position`;
  }
  if (g.def.kind === "filter") return `filter: ${describeFilter(g.def.filter)}`;
  const names = g.def.operands.map((id) => gates.find((x) => x.id === id)?.name ?? "?");
  return combineName(g.def.op, names);
}

/**
 * Where a new combination goes by default: inside the first gate for "not"
 * (its cells are all in it), else inside the nearest gate every operand is in,
 * else at the top.
 */
export function defaultCombineParent(gates: readonly Gate[], op: CombineOp, operands: readonly string[]): string | null {
  if (operands.length === 0) return null;
  if (op === "not") return operands[0]!;
  const chains = operands.map((id) => ancestors(gates, id));
  return chains[0]!.find((a) => chains.every((c) => c.includes(a))) ?? null;
}

// ---------------------------------------------------------------------------
// Deleting
// ---------------------------------------------------------------------------

export type DeleteMode = "reparent" | "subtree";

export interface DeleteResult {
  gates: Gate[];
  /** Names of the gates removed */
  removed: string[];
  /** Combinations that used a removed gate and now keep the cells they had */
  frozen: string[];
}

/**
 * Delete a gate: with "reparent" its children move up to its parent, with
 * "subtree" they go too. A combination using a removed gate keeps the cells it
 * had (it becomes saved cells) when they are known, else it is removed as well.
 */
export function deleteGate(
  gates: readonly Gate[],
  id: string,
  mode: DeleteMode,
  evals: ReadonlyMap<string, GateEval>,
): DeleteResult {
  const byId = new Map(gates.map((g) => [g.id, g]));
  if (!byId.has(id)) return { gates: [...gates], removed: [], frozen: [] };
  const removed = new Set(mode === "subtree" ? subtree(gates, id) : [id]);
  const frozen = new Map<string, CellsSelection>();
  for (let changed = true; changed; ) {
    changed = false;
    for (const g of gates) {
      if (removed.has(g.id) || frozen.has(g.id) || g.def.kind !== "combine") continue;
      if (!g.def.operands.some((o) => removed.has(o))) continue;
      const e = evals.get(g.id);
      if (e?.status === "ok") frozen.set(g.id, bitsetToSelection(e.cells));
      else {
        removed.add(g.id);
        changed = true;
      }
    }
  }
  // A kept gate inside a removed one moves up to the nearest kept parent.
  const keptParent = (parentId: string | null): string | null => {
    let at = parentId;
    const seen = new Set<string>();
    while (at && removed.has(at) && !seen.has(at)) {
      seen.add(at);
      at = byId.get(at)?.parentId ?? null;
    }
    return at && !removed.has(at) ? at : null;
  };
  const next = gates
    .filter((g) => !removed.has(g.id))
    .map((g) => {
      const sel = frozen.get(g.id);
      const parentId = keptParent(g.parentId);
      if (!sel && parentId === g.parentId) return g;
      return { ...g, parentId, ...(sel ? { def: { kind: "cells" as const, selection: sel } } : {}) };
    });
  return {
    gates: next,
    removed: gates.filter((g) => removed.has(g.id)).map((g) => g.name),
    frozen: gates.filter((g) => frozen.has(g.id)).map((g) => g.name),
  };
}

// ---------------------------------------------------------------------------
// A label column from the leaf gates
// ---------------------------------------------------------------------------

export interface LabelGroup {
  gateId: string;
  label: string;
  indices: number[];
}

export interface LabelPlan {
  /** One per leaf gate that keeps cells, in tree order */
  groups: LabelGroup[];
  /** Cells in more than one leaf gate */
  overlap: number;
  /** Cells in no leaf gate (they become Unassigned) */
  unassigned: number;
  /** Of those, cells inside a gate with children but in none of its leaves */
  inParentOnly: number;
  /** Leaf gates left with no cells (all taken by a deeper or later gate, or empty) */
  empty: string[];
}

/**
 * One label per cell from the leaf gates: a cell in several takes the deepest
 * gate's name, the later gate in the list on a tie. Every leaf must be
 * evaluated ("ok"), else null.
 */
export function planLabelColumn(gates: readonly Gate[], evals: ReadonlyMap<string, GateEval>, nCells: number): LabelPlan | null {
  const rows = gateTree(gates);
  const leaves = rows.filter((r) => !r.hasChildren);
  if (leaves.length === 0) return null;
  const cellsOf = (id: string) => {
    const e = evals.get(id);
    return e?.status === "ok" ? e.cells : null;
  };
  if (leaves.some((r) => !cellsOf(r.gate.id))) return null;

  // Shallow first, list order within a depth: later assignments win.
  const order = leaves.map((r, k) => ({ r, k })).sort((a, b) => a.r.depth - b.r.depth || a.k - b.k);
  const owner = new Int32Array(nCells).fill(-1);
  const hits = new Uint8Array(nCells);
  for (const { r, k } of order) {
    forEachCell(cellsOf(r.gate.id)!, (i) => {
      owner[i] = k;
      if (hits[i]! < 2) hits[i]!++;
    });
  }
  const byLeaf: number[][] = leaves.map(() => []);
  let overlap = 0;
  let unassigned = 0;
  for (let i = 0; i < nCells; i++) {
    if (owner[i]! >= 0) byLeaf[owner[i]!]!.push(i);
    else unassigned++;
    if (hits[i]! > 1) overlap++;
  }
  let inParentOnly = 0;
  const parents = rows.filter((r) => r.hasChildren).map((r) => cellsOf(r.gate.id)).filter((c): c is Bitset => !!c);
  if (parents.length > 0) {
    const inParents = parents.slice(1).reduce(bitsetOr, parents[0]!);
    forEachCell(inParents, (i) => {
      if (owner[i]! < 0) inParentOnly++;
    });
  }
  const groups: LabelGroup[] = [];
  const empty: string[] = [];
  leaves.forEach((r, k) => {
    if (byLeaf[k]!.length > 0) groups.push({ gateId: r.gate.id, label: r.gate.name.trim(), indices: byLeaf[k]! });
    else empty.push(r.gate.name);
  });
  return { groups, overlap, unassigned, inParentOnly, empty };
}

/**
 * The palette slot each gate takes in a label column made from the gates: the
 * column starts with Unassigned (slot 0) and adds the leaf gates' labels in
 * list order, so the colours match what "Label column…" will draw. Gates with
 * gates inside them, and leaves left with no cells of their own, get none.
 * Without a plan (cells still being counted) leaves are numbered in order.
 */
export function labelSlots(tree: readonly TreeRow[], plan: LabelPlan | null): Map<string, number> {
  const slots = new Map<string, number>();
  if (plan) plan.groups.forEach((g, k) => slots.set(g.gateId, k + 1));
  else tree.filter((r) => !r.hasChildren).forEach((r, k) => slots.set(r.gate.id, k + 1));
  return slots;
}

/** A free name for the gates' label column: gates, gates_2, ... */
export function suggestGateColumn(existing: readonly string[]): string {
  const taken = new Set(existing.map((c) => c.toLowerCase()));
  for (let i = 1; ; i++) {
    const name = i === 1 ? "gates" : `gates_${i}`;
    if (!taken.has(name)) return name;
  }
}

// ---------------------------------------------------------------------------
// Counts as a table (the CSV export)
// ---------------------------------------------------------------------------

export interface SampleValues {
  column: string;
  /** The column's value per cell (null: missing) */
  values: ArrayLike<string | null>;
}

const pctCell = (pct: number | null) => (pct == null ? "" : Number(pct.toFixed(3)));

/**
 * gate, parent, definition, n, % of parent, % of total, then (with a sample
 * column) n and % of the parent gate's cells for every sample. The first row
 * is all cells. Gates not evaluated have empty counts.
 */
export function gateCountsTable(
  gates: readonly Gate[],
  evals: ReadonlyMap<string, GateEval>,
  nCells: number,
  sample?: SampleValues | null,
): { headers: string[]; rows: (string | number)[][] } {
  const headers = ["gate", "parent", "definition", "n", "pct_parent", "pct_total"];
  let samples: string[] = [];
  let code: Int32Array | null = null;
  if (sample) {
    const index = new Map<string, number>();
    code = new Int32Array(nCells).fill(-1);
    for (let i = 0; i < nCells; i++) {
      const v = sample.values[i];
      if (v == null) continue;
      let c = index.get(v);
      if (c === undefined) {
        c = index.size;
        index.set(v, c);
      }
      code[i] = c;
    }
    samples = [...index.keys()];
    // Natural order (d2 before d10), as the legend sorts names.
    const sorted = [...samples].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    const remap = new Int32Array(samples.length);
    sorted.forEach((s, k) => (remap[index.get(s)!] = k));
    for (let i = 0; i < nCells; i++) if (code[i]! >= 0) code[i] = remap[code[i]!]!;
    samples = sorted;
    for (const s of samples) headers.push(`n_${s}`, `pct_parent_${s}`);
  }

  const perSample = (cells: Bitset | null): number[] => {
    const counts = new Array<number>(samples.length).fill(0);
    if (!code) return counts;
    if (!cells) {
      for (let i = 0; i < nCells; i++) if (code[i]! >= 0) counts[code[i]!]!++;
    } else {
      forEachCell(cells, (i) => {
        if (code![i]! >= 0) counts[code![i]!]!++;
      });
    }
    return counts;
  };
  const allSample = perSample(null);
  const sampleCells = (counts: number[], parent: number[]) =>
    counts.flatMap((n, k) => [n, pctCell(pctOf(n, parent[k]!))]);

  const rows: (string | number)[][] = [
    ["All cells", "", "every cell in the dataset", nCells, 100, 100, ...sampleCells(allSample, allSample)],
  ];
  const cache = new Map<string, number[]>();
  const countsOf = (id: string | null): number[] | null => {
    if (!id) return allSample;
    if (cache.has(id)) return cache.get(id)!;
    const e = evals.get(id);
    const c = e?.status === "ok" ? perSample(e.cells) : null;
    if (c) cache.set(id, c);
    return c;
  };
  for (const { gate } of gateTree(gates)) {
    const e = evals.get(gate.id);
    const parent = gate.parentId ? (gates.find((g) => g.id === gate.parentId)?.name ?? "") : "";
    const base = [gate.name, parent, describeGate(gate, gates)];
    if (e?.status !== "ok") {
      rows.push([...base, "", "", "", ...samples.flatMap(() => ["", ""])]);
      continue;
    }
    const counts = countsOf(gate.id)!;
    const parentCounts = countsOf(gate.parentId) ?? counts.map(() => 0);
    rows.push([...base, e.n, pctCell(e.pctParent), pctCell(e.pctTotal), ...sampleCells(counts, parentCounts)]);
  }
  return { headers, rows };
}

// ---------------------------------------------------------------------------
// Reading gates from untrusted JSON (a session file, localStorage)
// ---------------------------------------------------------------------------

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && !Array.isArray(x);

function text(x: unknown, where: string, max: number): string {
  if (typeof x !== "string" || x.trim() === "" || x.length > max) {
    throw new Error(`${where} must be non-empty text of at most ${max} characters`);
  }
  return x;
}

function gateId(x: unknown, where: string): string {
  if (typeof x !== "string" || !ID_RE.test(x)) throw new Error(`${where} is not a gate id`);
  return x;
}

function parseCellsSelection(x: unknown, where: string): CellsSelection {
  if (!isObj(x)) throw new Error(`${where} must be an object`);
  const n = x.n_cells;
  if (typeof n !== "number" || !Number.isInteger(n) || n < 1 || n > MAX_CELLS) {
    throw new Error(`${where}.n_cells must be a whole number from 1 to ${MAX_CELLS}`);
  }
  if (Object.prototype.hasOwnProperty.call(x, "indices")) {
    const indices = x.indices;
    if (!Array.isArray(indices) || indices.length > n) throw new Error(`${where}.indices must list at most ${n} cells`);
    for (const i of indices) {
      if (typeof i !== "number" || !Number.isInteger(i) || i < 0 || i >= n) {
        throw new Error(`${where}.indices must hold whole numbers from 0 to ${n - 1}`);
      }
    }
    return { n_cells: n, indices: indices as number[] };
  }
  const bitmask = x.bitmask;
  const nBytes = Math.ceil(n / 8);
  if (typeof bitmask !== "string" || bitmask.length !== Math.ceil(nBytes / 3) * 4 || !/^[A-Za-z0-9+/]*={0,2}$/.test(bitmask)) {
    throw new Error(`${where}.bitmask must be base64 of ${nBytes} bytes`);
  }
  let bytes = "";
  try {
    bytes = atob(bitmask);
  } catch {
    throw new Error(`${where}.bitmask is not valid base64`);
  }
  if (bytes.length !== nBytes) throw new Error(`${where}.bitmask must be base64 of ${nBytes} bytes`);
  return { n_cells: n, bitmask };
}

function parseDef(x: unknown, where: string): GateDef {
  if (!isObj(x)) throw new Error(`${where} must be an object`);
  if (x.kind === "cells") return { kind: "cells", selection: parseCellsSelection(x.selection, `${where}.selection`) };
  if (x.kind === "filter") return { kind: "filter", filter: parseCellFilter(x.filter, `${where}.filter`) };
  if (x.kind === "combine") {
    if (!COMBINE_OPS.includes(x.op as CombineOp)) throw new Error(`${where}.op must be "and", "or" or "not"`);
    const operands = x.operands;
    if (!Array.isArray(operands) || operands.length < 2 || operands.length > MAX_GATES) {
      throw new Error(`${where}.operands must list 2 to ${MAX_GATES} gates`);
    }
    return { kind: "combine", op: x.op as CombineOp, operands: operands.map((o, i) => gateId(o, `${where}.operands[${i}]`)) };
  }
  throw new Error(`${where}.kind must be "cells", "filter" or "combine"`);
}

/** A gate as session files and localStorage keep it (snake_case, like the rest of a session file). */
export interface SessionGate {
  id: string;
  name: string;
  parent_id: string | null;
  def: GateDef;
  created_at?: string;
}

export function toSessionGate(g: Gate): SessionGate {
  return { id: g.id, name: g.name, parent_id: g.parentId, def: g.def, ...(g.createdAt ? { created_at: g.createdAt } : {}) };
}

export function fromSessionGate(s: SessionGate): Gate {
  return { id: s.id, name: s.name, parentId: s.parent_id, def: s.def, createdAt: s.created_at ?? "" };
}

/**
 * Gates from untrusted JSON (a session file's `gates`, localStorage), every
 * field checked; throws an Error saying what is wrong. References between
 * gates are checked by `validateGates`.
 */
export function parseGateList(x: unknown, where = "gates"): SessionGate[] {
  if (!Array.isArray(x) || x.length > MAX_GATES) throw new Error(`${where} must hold at most ${MAX_GATES} gates`);
  return x.map((item, i) => {
    const at = `${where}[${i}]`;
    if (!isObj(item)) throw new Error(`${at} must be an object`);
    const parentId = item.parent_id ?? null;
    const gate: SessionGate = {
      id: gateId(item.id, `${at}.id`),
      name: text(item.name, `${at}.name`, LABEL_MAX).trim(),
      parent_id: parentId === null ? null : gateId(parentId, `${at}.parent_id`),
      def: parseDef(item.def, `${at}.def`),
    };
    if (item.created_at !== undefined) gate.created_at = text(item.created_at, `${at}.created_at`, 64);
    return gate;
  });
}
