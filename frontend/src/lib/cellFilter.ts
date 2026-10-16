/**
 * Cell filters: conditions on genes and metadata joined by AND / OR, with one
 * level of grouping (`CD14 > 1 AND (stim = STIM OR cell_type in {B, NK})`).
 * The backend evaluates them (`POST /cells/filter`, core/cell_filter.py) and
 * the matching cells become the selection, as a lasso's would.
 */
import { prettyObsLabel } from "@/lib/formatting";

export type FilterOp = "and" | "or";
export type GeneOp = ">" | ">=" | "<" | "<=";
export const GENE_OPS: readonly GeneOp[] = [">", ">=", "<", "<="];

/** A gene's expression on the filter's layer compared with a value. */
export interface GeneCondition {
  kind: "gene";
  gene: string;
  op: GeneOp;
  /** null while the value box is empty (an incomplete condition) */
  value: number | null;
}

/** A categorical column's value in a set. */
export interface CategoryCondition {
  kind: "category";
  column: string;
  values: string[];
}

/** A numeric column within [min, max]; null leaves that end open. */
export interface RangeCondition {
  kind: "range";
  column: string;
  min: number | null;
  max: number | null;
}

export type Condition = GeneCondition | CategoryCondition | RangeCondition;

export interface ConditionGroup {
  kind: "group";
  op: FilterOp;
  conditions: Condition[];
}

export type FilterItem = Condition | ConditionGroup;

export interface CellFilter {
  op: FilterOp;
  items: FilterItem[];
  /** Expression layer the gene conditions read ("" = the dataset's default) */
  layer: string;
}

// The backend's limits (core/cell_filter.py)
export const MAX_ITEMS = 20;
export const MAX_CONDITIONS = 40;
export const MAX_VALUES = 200;
export const MAX_NAME = 256;

/** A condition with everything filled in. */
export function isConditionComplete(c: Condition): boolean {
  switch (c.kind) {
    case "gene":
      return c.gene.trim() !== "" && c.value != null && Number.isFinite(c.value);
    case "category":
      return c.column !== "" && c.values.length > 0;
    case "range":
      return (
        c.column !== "" &&
        (c.min != null || c.max != null) &&
        (c.min == null || c.max == null || c.min <= c.max)
      );
  }
}

/**
 * The filter without its incomplete conditions (and the groups left empty), or
 * null when nothing is left: what the count and "Select" use, so the sentence
 * shown always says exactly what was applied.
 */
export function pruneFilter(filter: CellFilter): CellFilter | null {
  const items: FilterItem[] = [];
  for (const item of filter.items) {
    if (item.kind === "group") {
      const conditions = item.conditions.filter(isConditionComplete);
      if (conditions.length > 0) items.push({ ...item, conditions });
    } else if (isConditionComplete(item)) {
      items.push(item);
    }
  }
  return items.length > 0 ? { ...filter, items } : null;
}

/** Every condition of the filter, groups flattened. */
export function filterConditions(filter: CellFilter): Condition[] {
  return filter.items.flatMap((i) => (i.kind === "group" ? i.conditions : [i]));
}

const num = (v: number) => (Number.isInteger(v) ? String(v) : String(+v.toPrecision(6)));
const OP_TEXT: Record<GeneOp, string> = { ">": ">", ">=": "≥", "<": "<", "<=": "≤" };

export function describeCondition(c: Condition): string {
  switch (c.kind) {
    case "gene":
      return `${c.gene} ${OP_TEXT[c.op]} ${c.value == null ? "?" : num(c.value)}`;
    case "category": {
      const col = prettyObsLabel(c.column);
      return c.values.length === 1 ? `${col} = ${c.values[0]}` : `${col} in {${c.values.join(", ")}}`;
    }
    case "range": {
      const col = prettyObsLabel(c.column);
      if (c.min != null && c.max != null) return `${num(c.min)} ≤ ${col} ≤ ${num(c.max)}`;
      if (c.min != null) return `${col} ≥ ${num(c.min)}`;
      return `${col} ≤ ${num(c.max ?? 0)}`;
    }
  }
}

/** The filter as a sentence: `CD14 > 1 AND (stim = STIM OR cell_type in {B, NK})`. */
export function describeFilter(filter: CellFilter): string {
  const join = (op: FilterOp) => ` ${op.toUpperCase()} `;
  return filter.items
    .map((item) => {
      if (item.kind !== "group") return describeCondition(item);
      const inner = item.conditions.map(describeCondition).join(join(item.op));
      return item.conditions.length > 1 && filter.items.length > 1 ? `(${inner})` : inner;
    })
    .join(join(filter.op));
}

/**
 * The set bits of a packed bitmask (base64; cell i is bit i % 8, least
 * significant first, of byte i // 8), as cell indices below `nCells`.
 */
export function decodeMask(base64: string, nCells: number): Set<number> {
  const bytes = atob(base64);
  const out = new Set<number>();
  const n = Math.min(nCells, bytes.length * 8);
  for (let b = 0; b * 8 < n; b++) {
    const byte = bytes.charCodeAt(b);
    if (byte === 0) continue;
    for (let bit = 0; bit < 8; bit++) {
      const i = b * 8 + bit;
      if (i < n && byte & (1 << bit)) out.add(i);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Checking a filter read from a file (a saved session)
// ---------------------------------------------------------------------------

const isObj = (x: unknown): x is Record<string, unknown> =>
  typeof x === "object" && x !== null && !Array.isArray(x);

function name(x: unknown, what: string): string {
  if (typeof x !== "string" || x.length === 0 || x.length > MAX_NAME) {
    throw new Error(`${what} must be a non-empty text of at most ${MAX_NAME} characters`);
  }
  return x;
}

function finiteOrNull(x: unknown, what: string): number | null {
  if (x == null) return null;
  if (typeof x !== "number" || !Number.isFinite(x)) throw new Error(`${what} must be a number`);
  return x;
}

function op(x: unknown, what: string): FilterOp {
  if (x !== "and" && x !== "or") throw new Error(`${what} must be "and" or "or"`);
  return x;
}

function parseCondition(x: unknown, where: string): Condition {
  if (!isObj(x)) throw new Error(`${where} must be an object`);
  switch (x.kind) {
    case "gene": {
      if (!GENE_OPS.includes(x.op as GeneOp)) throw new Error(`${where}.op must be one of ${GENE_OPS.join(" ")}`);
      const value = finiteOrNull(x.value, `${where}.value`);
      if (value == null) throw new Error(`${where}.value must be a number`);
      return { kind: "gene", gene: name(x.gene, `${where}.gene`), op: x.op as GeneOp, value };
    }
    case "category": {
      const values = x.values;
      if (!Array.isArray(values) || values.length === 0 || values.length > MAX_VALUES) {
        throw new Error(`${where}.values must list 1 to ${MAX_VALUES} values`);
      }
      return {
        kind: "category",
        column: name(x.column, `${where}.column`),
        values: values.map((v, i) => name(v, `${where}.values[${i}]`)),
      };
    }
    case "range": {
      const c: RangeCondition = {
        kind: "range",
        column: name(x.column, `${where}.column`),
        min: finiteOrNull(x.min, `${where}.min`),
        max: finiteOrNull(x.max, `${where}.max`),
      };
      if (!isConditionComplete(c)) throw new Error(`${where} needs a minimum or a maximum, the minimum not above the maximum`);
      return c;
    }
    default:
      throw new Error(`${where}.kind must be "gene", "category" or "range"`);
  }
}

/** A filter from untrusted JSON: every field checked, limits as on the backend. Throws on anything else. */
export function parseCellFilter(x: unknown, where = "filter"): CellFilter {
  if (!isObj(x)) throw new Error(`${where} must be an object`);
  const items = x.items;
  if (!Array.isArray(items) || items.length === 0 || items.length > MAX_ITEMS) {
    throw new Error(`${where}.items must hold 1 to ${MAX_ITEMS} conditions or groups`);
  }
  const layer = x.layer ?? "";
  if (typeof layer !== "string" || layer.length > MAX_NAME) throw new Error(`${where}.layer must be text`);
  const filter: CellFilter = {
    op: op(x.op, `${where}.op`),
    layer,
    items: items.map((item, i): FilterItem => {
      const at = `${where}.items[${i}]`;
      if (!isObj(item) || item.kind !== "group") return parseCondition(item, at);
      const conditions = item.conditions;
      if (!Array.isArray(conditions) || conditions.length === 0 || conditions.length > MAX_ITEMS) {
        throw new Error(`${at}.conditions must hold 1 to ${MAX_ITEMS} conditions`);
      }
      return {
        kind: "group",
        op: op(item.op, `${at}.op`),
        conditions: conditions.map((c, j) => parseCondition(c, `${at}.conditions[${j}]`)),
      };
    }),
  };
  if (filterConditions(filter).length > MAX_CONDITIONS) {
    throw new Error(`${where} has more than ${MAX_CONDITIONS} conditions`);
  }
  return filter;
}
