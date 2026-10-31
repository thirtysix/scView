/**
 * The session file: a small JSON description of the workspace ("Save session"
 * downloads it, "Open session" restores it; lib/sessionRestore.ts does both).
 * The app's own reload restore (hooks/useSessionRestore.ts) keeps a slice of
 * the same format in localStorage.
 *
 * The file is user-supplied, so `parseSession` checks every field it reads:
 * an unknown format or version, or a field of the wrong type, rejects the file;
 * fields it does not know are ignored. Cell indices are checked against the
 * dataset when the session is restored, never trusted.
 */
import { parseCellFilter, decodeMask, type CellFilter } from "@/lib/cellFilter";
import type { LabelChoice } from "@/lib/centroidLabels";
import {
  CATEGORICAL_PALETTES,
  COLORMAP_STOPS,
  DIVERGING_COLORMAP_STOPS,
  type CategoricalPaletteName,
  type ColormapName,
  type DivergingColormapName,
} from "@/lib/colors";
import type { ClipSetting, SignedColormapMode } from "@/lib/colorScale";
import { PANEL_IDS, UNIFIED_SUBTABS, type PanelId, type UnifiedSubtab } from "@/lib/constants";
import { exportFilename } from "@/lib/exportFilename";
import { parseGateList, type SessionGate } from "@/lib/gates";
import type { DEMode, DETest } from "@/stores/deSettingsStore";
import type { CellMethod } from "@/lib/deMethods";

export const SESSION_FORMAT = "scview-session";
export const SESSION_VERSION = 1;
/** Larger files are refused before they are read. */
export const SESSION_MAX_BYTES = 20 * 1024 * 1024;
const MAX_CELLS = 50_000_000;
const MAX_TEXT = 256;
const MAX_GENES = 5000;
const MAX_GENE_LISTS = 200;

/** The main plot's camera: deck.gl view state fields, in embedding coordinates. */
export interface Camera {
  target: number[];
  zoom: number;
  rotationOrbit?: number;
  rotationX?: number;
}

/** The camera fields of a deck.gl view state (null without a target and zoom). */
export function cameraFromViewState(vs: Record<string, unknown>): Camera | null {
  const target = vs.target;
  const zoom = vs.zoom;
  if (!Array.isArray(target) || !target.every((t) => typeof t === "number" && Number.isFinite(t))) return null;
  if (typeof zoom !== "number" || !Number.isFinite(zoom)) return null;
  const cam: Camera = { target: target as number[], zoom };
  for (const key of ["rotationOrbit", "rotationX"] as const) {
    const v = vs[key];
    if (typeof v === "number" && Number.isFinite(v)) cam[key] = v;
  }
  return cam;
}

export type SessionOverlay =
  // layer: a protein (CITE-seq) is read from the protein layer
  | { type: "expression"; gene: string; layer?: string }
  // method: how the set was scored; absent (and in older sessions) = score_genes
  | { type: "score"; name: string; genes: string[]; method?: "score_genes" | "aucell" | "ssgsea" | "gsva" }
  // A TF or pathway's per-cell activity (Activity subtab), computed again on open.
  | { type: "activity"; network: "collectri" | "progeny"; source: string }
  // Plot reliability (neighbour preservation) of an embedding, computed again on open;
  // dims: the PCA pair plotted, when not the first two.
  | { type: "reliability"; embedding: string; dims?: [number, number] }
  // A gene program's usage share (Programs subtab), read again from the saved run on open.
  | { type: "program"; index: number }
  // One gene's peak-based ATAC activity (Accessibility subtab), read again on open.
  | { type: "atac_activity"; gene: string };

/** A selection: the filter it came from, else its cells (whichever is shorter). */
export type SessionSelection =
  | { filter: CellFilter }
  | { n_cells: number; indices: number[] }
  | { n_cells: number; bitmask: string };

export interface SessionDataset {
  id: string;
  name?: string;
  n_cells?: number | null;
  /** Hash of the cell barcodes in order (GET /datasets/{id}): equal = same cells */
  cells_fingerprint?: string | null;
}

export interface SessionView {
  panel?: PanelId;
  subtab?: UnifiedSubtab;
  embedding?: string;
  color_by?: string;
  /** The colouring was picked by the user, not the dataset's default */
  color_by_chosen?: boolean;
  expression_layer?: string;
  group_by?: string;
  split_by?: string | null;
  show_context?: boolean;
  compare_columns?: string[] | null;
  labels?: LabelChoice | null;
  camera?: Camera | null;
  highlight?: { column: string; value: string } | null;
  overlay?: SessionOverlay | null;
  /** This view's colour-range clipping over the default (lib/colorScale.ts) */
  color_clip?: ClipSetting | null;
}

export interface SessionDE {
  mode?: DEMode;
  split_by?: string;
  group_a?: string;
  group_b?: string;
  within_column?: string;
  within_value?: string;
  test?: DETest;
  replicate_key?: string;
  fc_threshold?: number;
  min_pct?: number;
  /** The cell-level test: wilcoxon | t-test | logreg. */
  cell_method?: CellMethod;
}

export interface SessionColors {
  categorical_palette?: CategoricalPaletteName;
  continuous_colormap?: ColormapName;
  diverging_colormap?: DivergingColormapName;
  signed_colormap_mode?: SignedColormapMode;
  /** The default clipping of the colour range */
  clip?: ClipSetting;
  point_size?: number;
  opacity?: number;
  background?: "white" | "dark";
  max_rendered_cells?: number;
}

/** The Communication tab's run settings (stores/communicationStore.ts); the result is run again. */
export interface SessionCommunication {
  groupby?: string;
  subset_column?: string;
  subset_value?: string;
  expr_prop?: number;
}

/** One of the user's gene lists (stores/geneListsStore.ts); lists are not tied to the dataset. */
export interface SessionGeneList {
  name: string;
  genes: string[];
  source?: string;
}

export interface SessionFile {
  format: typeof SESSION_FORMAT;
  version: typeof SESSION_VERSION;
  saved_at?: string;
  dataset?: SessionDataset;
  view?: SessionView;
  selection?: SessionSelection | null;
  de?: SessionDE;
  colors?: SessionColors;
  gene_lists?: SessionGeneList[];
  communication?: SessionCommunication;
  /** The dataset's cell gates (lib/gates.ts); gates saved by cell position need the same cells */
  gates?: SessionGate[];
}

/** A session file that can't be used, with the reason in plain words. */
export class SessionError extends Error {}

// ---------------------------------------------------------------------------
// Cells: indices or a bitmask, whichever is shorter
// ---------------------------------------------------------------------------

const maskBytes = (nCells: number) => Math.ceil(nCells / 8);

/** Packed bitmask (base64), the backend's layout: cell i is bit i % 8 of byte i // 8. */
export function encodeMask(indices: Iterable<number>, nCells: number): string {
  const bytes = new Uint8Array(maskBytes(nCells));
  for (const i of indices) {
    if (Number.isInteger(i) && i >= 0 && i < nCells) bytes[i >> 3]! |= 1 << (i & 7);
  }
  let text = "";
  for (let start = 0; start < bytes.length; start += 0x8000) {
    text += String.fromCharCode(...bytes.subarray(start, start + 0x8000));
  }
  return btoa(text);
}

export function encodeSelection(indices: Iterable<number>, nCells: number): SessionSelection {
  const sorted = [...indices].filter((i) => Number.isInteger(i) && i >= 0 && i < nCells).sort((a, b) => a - b);
  const listChars = sorted.reduce((n, i) => n + String(i).length + 1, 0);
  const maskChars = Math.ceil(maskBytes(nCells) / 3) * 4;
  return listChars <= maskChars
    ? { n_cells: nCells, indices: sorted }
    : { n_cells: nCells, bitmask: encodeMask(sorted, nCells) };
}

/**
 * The saved cells that exist in a dataset of `nCells` cells; `ignored` counts
 * the indices at or beyond it (never used).
 */
export function decodeSelection(
  sel: Exclude<SessionSelection, { filter: CellFilter }>,
  nCells: number,
): { cells: Set<number>; ignored: number } {
  if ("indices" in sel) {
    const cells = new Set<number>();
    let ignored = 0;
    for (const i of sel.indices) {
      if (i < nCells) cells.add(i);
      else ignored++;
    }
    return { cells, ignored };
  }
  const all = decodeMask(sel.bitmask, sel.n_cells);
  const cells = new Set<number>();
  for (const i of all) if (i < nCells) cells.add(i);
  return { cells, ignored: all.size - cells.size };
}

// ---------------------------------------------------------------------------
// Checking a file
// ---------------------------------------------------------------------------

type Obj = Record<string, unknown>;
const isObj = (x: unknown): x is Obj => typeof x === "object" && x !== null && !Array.isArray(x);
const own = (o: object, key: string) => Object.prototype.hasOwnProperty.call(o, key);

function fail(where: string, what: string): never {
  throw new SessionError(`${where} ${what}.`);
}

function text(x: unknown, where: string, { empty = true, max = MAX_TEXT } = {}): string {
  if (typeof x !== "string" || x.length > max || (!empty && x === "")) {
    fail(where, `must be ${empty ? "" : "non-empty "}text of at most ${max} characters`);
  }
  return x;
}

function number(x: unknown, where: string, min: number, max: number, integer = false): number {
  if (typeof x !== "number" || !Number.isFinite(x) || x < min || x > max || (integer && !Number.isInteger(x))) {
    fail(where, `must be ${integer ? "a whole number" : "a number"} from ${min} to ${max}`);
  }
  return x;
}

function oneOf<T extends string>(x: unknown, values: readonly T[], where: string): T {
  if (typeof x !== "string" || !values.includes(x as T)) fail(where, `must be one of ${values.join(", ")}`);
  return x as T;
}

function bool(x: unknown, where: string): boolean {
  if (typeof x !== "boolean") fail(where, "must be true or false");
  return x;
}

function texts(x: unknown, where: string, min: number, max: number): string[] {
  if (!Array.isArray(x) || x.length < min || x.length > max) fail(where, `must list ${min} to ${max} names`);
  return x.map((v, i) => text(v, `${where}[${i}]`, { empty: false }));
}

/** Copy the fields of `o` that are present, each through its checker. */
function fields<T>(o: Obj, where: string, checks: { [K in keyof T]-?: (x: unknown, at: string) => T[K] }): T {
  const out: Partial<T> = {};
  for (const key of Object.keys(checks) as (keyof T & string)[]) {
    if (own(o, key) && o[key] !== undefined) out[key] = checks[key](o[key], `${where}.${key}`);
  }
  return out as T;
}

const nullable =
  <T>(check: (x: unknown, at: string) => T) =>
  (x: unknown, at: string): T | null =>
    x === null ? null : check(x, at);

function object(x: unknown, where: string): Obj {
  if (!isObj(x)) fail(where, "must be an object");
  return x;
}

function parseCamera(x: unknown, where: string): Camera {
  const o = object(x, where);
  const target = o.target;
  if (!Array.isArray(target) || target.length < 2 || target.length > 3) fail(`${where}.target`, "must list 2 or 3 numbers");
  return {
    target: target.map((v, i) => number(v, `${where}.target[${i}]`, -1e12, 1e12)),
    zoom: number(o.zoom, `${where}.zoom`, -100, 100),
    ...fields<Pick<Camera, "rotationOrbit" | "rotationX">>(o, where, {
      rotationOrbit: (v, at) => number(v, at, -1e6, 1e6),
      rotationX: (v, at) => number(v, at, -1e6, 1e6),
    }),
  };
}

function parseOverlay(x: unknown, where: string): SessionOverlay {
  const o = object(x, where);
  if (o.type === "expression") {
    const gene = text(o.gene, `${where}.gene`, { empty: false });
    return o.layer === undefined ? { type: "expression", gene }
      : { type: "expression", gene, layer: text(o.layer, `${where}.layer`, { empty: false }) };
  }
  if (o.type === "score") {
    const out: Extract<SessionOverlay, { type: "score" }> = {
      type: "score",
      name: text(o.name, `${where}.name`, { empty: false, max: 1024 }),
      genes: texts(o.genes, `${where}.genes`, 1, MAX_GENES),
    };
    if (o.method !== undefined) out.method = oneOf(o.method, ["score_genes", "aucell", "ssgsea", "gsva"] as const, `${where}.method`);
    return out;
  }
  if (o.type === "activity") {
    return {
      type: "activity",
      network: oneOf(o.network, ["collectri", "progeny"] as const, `${where}.network`),
      source: text(o.source, `${where}.source`, { empty: false, max: 256 }),
    };
  }
  if (o.type === "reliability") {
    const embedding = text(o.embedding, `${where}.embedding`, { empty: false });
    if (o.dims === undefined) return { type: "reliability", embedding };
    const d = o.dims;
    if (!Array.isArray(d) || d.length !== 2) fail(`${where}.dims`, "must list 2 column numbers");
    const dims: [number, number] = [
      number(d[0], `${where}.dims[0]`, 0, 10_000, true),
      number(d[1], `${where}.dims[1]`, 0, 10_000, true),
    ];
    return { type: "reliability", embedding, dims };
  }
  if (o.type === "program") {
    return { type: "program", index: number(o.index, `${where}.index`, 1, 1000, true) };
  }
  if (o.type === "atac_activity") {
    return { type: "atac_activity", gene: text(o.gene, `${where}.gene`, { empty: false }) };
  }
  fail(`${where}.type`, 'must be "expression", "score", "activity", "reliability", "program" or "atac_activity"');
}

const CLIP_PRESETS = ["none", "p1", "p2", "p5", "manual"] as const;

function parseClip(x: unknown, where: string): ClipSetting {
  const o = object(x, where);
  const preset = oneOf(o.preset, CLIP_PRESETS, `${where}.preset`);
  const out: ClipSetting = { preset };
  if (preset === "manual") {
    if (o.min !== undefined && o.min !== null) out.min = number(o.min, `${where}.min`, -1e12, 1e12);
    if (o.max !== undefined && o.max !== null) out.max = number(o.max, `${where}.max`, -1e12, 1e12);
  }
  return out;
}

function parseSelection(x: unknown, where: string): SessionSelection {
  const o = object(x, where);
  if (own(o, "filter")) {
    try {
      return { filter: parseCellFilter(o.filter, `${where}.filter`) };
    } catch (e) {
      throw new SessionError(`${e instanceof Error ? e.message : String(e)}.`);
    }
  }
  const nCells = number(o.n_cells, `${where}.n_cells`, 1, MAX_CELLS, true);
  if (own(o, "indices")) {
    const indices = o.indices;
    if (!Array.isArray(indices) || indices.length > nCells) fail(`${where}.indices`, `must list at most ${nCells} cells`);
    for (const i of indices) {
      if (typeof i !== "number" || !Number.isInteger(i) || i < 0 || i >= MAX_CELLS) {
        fail(`${where}.indices`, "must hold whole numbers from 0");
      }
    }
    return { n_cells: nCells, indices: indices as number[] };
  }
  const bitmask = o.bitmask;
  const chars = Math.ceil(maskBytes(nCells) / 3) * 4;
  if (typeof bitmask !== "string" || bitmask.length !== chars || !/^[A-Za-z0-9+/]*={0,2}$/.test(bitmask)) {
    fail(`${where}.bitmask`, `must be base64 of ${maskBytes(nCells)} bytes`);
  }
  let bytes = "";
  try {
    bytes = atob(bitmask);
  } catch {
    fail(`${where}.bitmask`, "is not valid base64");
  }
  if (bytes.length !== maskBytes(nCells)) fail(`${where}.bitmask`, `must be base64 of ${maskBytes(nCells)} bytes`);
  return { n_cells: nCells, bitmask };
}

/** A session from untrusted JSON (a file or localStorage). Throws SessionError. */
export function parseSession(x: unknown): SessionFile {
  if (!isObj(x) || x.format !== SESSION_FORMAT) {
    throw new SessionError("This is not an scView session file.");
  }
  if (x.version !== SESSION_VERSION) {
    const v = typeof x.version === "number" ? x.version : String(x.version);
    throw new SessionError(
      typeof x.version === "number" && x.version > SESSION_VERSION
        ? `This session was saved by a newer scView (format version ${v}); this one reads version ${SESSION_VERSION}.`
        : `Unknown session format version ${v}.`,
    );
  }
  const session: SessionFile = { format: SESSION_FORMAT, version: SESSION_VERSION };
  if (x.saved_at !== undefined) session.saved_at = text(x.saved_at, "saved_at", { max: 64 });

  if (x.dataset !== undefined) {
    const d = object(x.dataset, "dataset");
    const id = text(d.id, "dataset.id", { empty: false, max: 64 });
    // Ids go into request paths: the backend's id rule, nothing else.
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(id)) fail("dataset.id", "is not a dataset id");
    session.dataset = {
      id,
      ...fields<Omit<SessionDataset, "id">>(d, "dataset", {
        name: (v, at) => text(v, at, { max: 1024 }),
        n_cells: nullable((v, at) => number(v, at, 0, MAX_CELLS, true)),
        cells_fingerprint: nullable((v, at) => text(v, at, { max: 128 })),
      }),
    };
  }

  if (x.view !== undefined) {
    session.view = fields<SessionView>(object(x.view, "view"), "view", {
      panel: (v, at) => oneOf(v, PANEL_IDS, at),
      subtab: (v, at) => oneOf(v, UNIFIED_SUBTABS, at),
      embedding: (v, at) => text(v, at),
      color_by: (v, at) => text(v, at),
      color_by_chosen: bool,
      expression_layer: (v, at) => text(v, at),
      group_by: (v, at) => text(v, at),
      split_by: nullable((v, at) => text(v, at, { empty: false })),
      show_context: bool,
      compare_columns: nullable((v, at) => texts(v, at, 2, 4)),
      labels: nullable((v, at) => {
        const o = object(v, at);
        return { key: text(o.key, `${at}.key`, { max: 4 * MAX_TEXT + 16 }), show: bool(o.show, `${at}.show`) };
      }),
      camera: nullable(parseCamera),
      highlight: nullable((v, at) => {
        const o = object(v, at);
        return {
          column: text(o.column, `${at}.column`, { empty: false }),
          value: text(o.value, `${at}.value`, { max: 1024 }),
        };
      }),
      overlay: nullable(parseOverlay),
      color_clip: nullable(parseClip),
    });
  }

  if (x.selection !== undefined) session.selection = nullable(parseSelection)(x.selection, "selection");

  if (x.de !== undefined) {
    session.de = fields<SessionDE>(object(x.de, "de"), "de", {
      mode: (v, at) => oneOf(v, ["selection", "groups"] as const, at),
      split_by: (v, at) => text(v, at),
      group_a: (v, at) => text(v, at, { max: 1024 }),
      group_b: (v, at) => text(v, at, { max: 1024 }),
      within_column: (v, at) => text(v, at),
      within_value: (v, at) => text(v, at, { max: 1024 }),
      test: (v, at) => oneOf(v, ["wilcoxon", "pseudobulk"] as const, at),
      replicate_key: (v, at) => text(v, at),
      fc_threshold: (v, at) => number(v, at, 0, 20),
      min_pct: (v, at) => number(v, at, 0, 1),
      cell_method: (v, at) => oneOf(v, ["wilcoxon", "t-test", "logreg"] as const, at),
    });
  }

  if (x.colors !== undefined) {
    session.colors = fields<SessionColors>(object(x.colors, "colors"), "colors", {
      categorical_palette: (v, at) =>
        oneOf(v, Object.keys(CATEGORICAL_PALETTES) as CategoricalPaletteName[], at),
      continuous_colormap: (v, at) => oneOf(v, Object.keys(COLORMAP_STOPS) as ColormapName[], at),
      diverging_colormap: (v, at) => oneOf(v, Object.keys(DIVERGING_COLORMAP_STOPS) as DivergingColormapName[], at),
      signed_colormap_mode: (v, at) => oneOf(v, ["auto", "sequential", "diverging"] as const, at),
      clip: parseClip,
      point_size: (v, at) => number(v, at, 0.5, 20),
      opacity: (v, at) => number(v, at, 0.05, 1),
      background: (v, at) => oneOf(v, ["white", "dark"] as const, at),
      max_rendered_cells: (v, at) => number(v, at, 1000, MAX_CELLS, true),
    });
  }

  if (x.communication !== undefined) {
    session.communication = fields<SessionCommunication>(object(x.communication, "communication"), "communication", {
      groupby: (v, at) => text(v, at),
      subset_column: (v, at) => text(v, at),
      subset_value: (v, at) => text(v, at, { max: 1024 }),
      expr_prop: (v, at) => number(v, at, 0, 1),
    });
  }

  if (x.gene_lists !== undefined) {
    const lists = x.gene_lists;
    if (!Array.isArray(lists) || lists.length > MAX_GENE_LISTS) {
      fail("gene_lists", `must hold at most ${MAX_GENE_LISTS} gene lists`);
    }
    session.gene_lists = lists.map((v, i) => {
      const at = `gene_lists[${i}]`;
      const o = object(v, at);
      return {
        name: text(o.name, `${at}.name`, { empty: false, max: 200 }),
        genes: texts(o.genes, `${at}.genes`, 0, MAX_GENES),
        ...fields<Pick<SessionGeneList, "source">>(o, at, { source: (v2, w) => text(v2, w, { max: 300 }) }),
      };
    });
  }

  if (x.gates !== undefined) {
    try {
      session.gates = parseGateList(x.gates, "gates");
    } catch (e) {
      throw new SessionError(`${e instanceof Error ? e.message : String(e)}.`);
    }
  }
  return session;
}

/** Read a session file the user picked: size first, then JSON, then every field. */
export async function readSessionFile(file: Blob): Promise<SessionFile> {
  if (file.size > SESSION_MAX_BYTES) {
    throw new SessionError(`The file is too large for a session (${Math.round(file.size / 1048576)} MB).`);
  }
  let json: unknown;
  try {
    json = JSON.parse(await file.text());
  } catch {
    throw new SessionError("This file is not JSON, so not an scView session file.");
  }
  return parseSession(json);
}

/** The file's text: indented, with lists of numbers (cell indices) on one line. */
export function sessionJson(session: SessionFile): string {
  return JSON.stringify(session, null, 2).replace(/\[\s+(-?[\d.e+-]+(?:,\s+-?[\d.e+-]+)*)\s+\]/g, (_m, list: string) =>
    `[${list.replace(/,\s+/g, ", ")}]`,
  );
}

/** `scview_<dataset>_session_<yyyy-mm-dd>.json` */
export function sessionFilename(datasetName: string | null | undefined, when: Date): string {
  const day = when.toISOString().slice(0, 10);
  return `${exportFilename(datasetName, `session_${day}`)}.json`;
}

// ---------------------------------------------------------------------------
// The reload restore's old localStorage snapshot (before the session format)
// ---------------------------------------------------------------------------

/** A stored snapshot, either a (partial) session or the older flat shape. */
export function sessionFromStorage(raw: unknown): SessionFile | null {
  if (!isObj(raw)) return null;
  // The flat shape kept what it could and skipped the rest, field by field.
  const known = (v: unknown, values: readonly string[]) => typeof v === "string" && values.includes(v);
  const candidate =
    raw.format === undefined
      ? {
          format: SESSION_FORMAT,
          version: SESSION_VERSION,
          ...(typeof raw.datasetId === "string" && raw.datasetId ? { dataset: { id: raw.datasetId } } : {}),
          view: {
            ...(known(raw.activePanel, PANEL_IDS) ? { panel: raw.activePanel } : {}),
            ...(typeof raw.colorBy === "string" && raw.colorBy ? { color_by: raw.colorBy, color_by_chosen: true } : {}),
            ...(typeof raw.embedding === "string" && raw.embedding ? { embedding: raw.embedding } : {}),
          },
          colors: {
            ...(known(raw.categoricalPalette, Object.keys(CATEGORICAL_PALETTES))
              ? { categorical_palette: raw.categoricalPalette }
              : {}),
            ...(known(raw.continuousColormap, Object.keys(COLORMAP_STOPS))
              ? { continuous_colormap: raw.continuousColormap }
              : {}),
          },
        }
      : raw;
  try {
    return parseSession(candidate);
  } catch {
    return null;
  }
}
