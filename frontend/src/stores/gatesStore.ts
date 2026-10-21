import { create } from "zustand";
import type { DatasetInfo } from "@/api/types";
import {
  bitsetToSet,
  deleteGate as deleteFromTree,
  fromSessionGate,
  gateNameError,
  MAX_GATES,
  newGateId,
  parseGateList,
  toSessionGate,
  validateGates,
  type Bitset,
  type DeleteMode,
  type DeleteResult,
  type DroppedGate,
  type Gate,
  type GateDef,
  type GateEval,
} from "@/lib/gates";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";

/**
 * Saved cell gates per dataset (lib/gates.ts, the toolbar's Gates panel).
 * Gates saved by cell position mean something only on the cells they were
 * drawn on, so each dataset's gates carry its `cells_fingerprint` and cell
 * count; when the open dataset's cells differ (it was filtered since), its
 * gates are stale and are not applied. Kept in localStorage (size-capped, a
 * blocked or full storage leaves them working for this page) and in session
 * files (lib/sessionRestore.ts).
 */
export interface GateRecord {
  /** The dataset's cells when its gates were saved */
  fingerprint: string | null;
  nCells: number;
  gates: Gate[];
  /** ISO timestamp of the last change: the most recent are kept first when storage runs short */
  updatedAt: string;
}

/** A filter gate's cells from the backend, or why they could not be found. */
export type FilterResult = Bitset | { error: string };

export interface GateDraft {
  name: string;
  parentId: string | null;
  def: GateDef;
}

export interface MergeResult {
  added: number;
  replaced: number;
  /** Stale gates (drawn on other cells) the session's replaced */
  replacedStale: number;
  dropped: DroppedGate[];
}

interface GatesState {
  records: Record<string, GateRecord>;
  /** The gate clicked last: new gates go inside it (this page only) */
  active: { datasetId: string; gateId: string } | null;
  /** The cells the selection was set to from a gate, to tell whether it still shows that gate */
  shown: { datasetId: string; gateId: string; cells: Set<number> } | null;
  /** Filter gates' cells (the filter alone, before the parent), by dataset then gate (this page only) */
  filterCells: Record<string, Record<string, FilterResult>>;
  /** The toolbar's Gates strip is open (kept here: a tab switch remounts the toolbar) */
  panelOpen: boolean;
  setPanelOpen: (open: boolean) => void;
  /** A new gate; returns its id, or null when the dataset's gates are stale or MAX_GATES are kept.
   *  `cells`: a filter gate's cells, already known from the selection it came from. */
  addGate: (dataset: DatasetInfo, draft: GateDraft, cells?: Bitset) => string | null;
  renameGate: (datasetId: string, id: string, name: string) => void;
  deleteGate: (datasetId: string, id: string, mode: DeleteMode, evals: ReadonlyMap<string, GateEval>) => DeleteResult | null;
  setActive: (datasetId: string, gateId: string | null) => void;
  /** Forget a dataset's gates (the stale ones, or all) */
  discard: (datasetId: string) => void;
  /** A session's gates into the dataset's: same id = the session's version, others kept */
  mergeGates: (dataset: DatasetInfo, gates: Gate[]) => MergeResult;
  setFilterCells: (datasetId: string, gateId: string, result: FilterResult) => void;
  clearFilterCells: (datasetId: string) => void;
}

const now = () => new Date().toISOString();

/** Why a dataset's saved gates do not fit its cells now, or null when they do. */
export function staleReason(record: GateRecord, dataset: Pick<DatasetInfo, "n_cells" | "cells_fingerprint">): string | null {
  const n = dataset.n_cells ?? 0;
  if (record.nCells !== n) {
    return `They were drawn on ${record.nCells.toLocaleString()} cells and the dataset now has ${n.toLocaleString()} (it was filtered since), so they would point at other cells.`;
  }
  if (record.fingerprint && dataset.cells_fingerprint && record.fingerprint !== dataset.cells_fingerprint) {
    return "The dataset has the same number of cells as when they were drawn, but not the same cells, so they would point at other cells.";
  }
  return null;
}

const recordFor = (dataset: DatasetInfo, gates: Gate[]): GateRecord => ({
  fingerprint: dataset.cells_fingerprint ?? null,
  nCells: dataset.n_cells ?? 0,
  gates,
  updatedAt: now(),
});

// Requests for a dataset's filter cells started before a clear are ignored.
const generations = new Map<string, number>();
export const filterGeneration = (datasetId: string) => generations.get(datasetId) ?? 0;

export const useGatesStore = create<GatesState>((set, get) => {
  const usable = (dataset: DatasetInfo): GateRecord | null => {
    const r = get().records[dataset.id];
    if (!r) return recordFor(dataset, []);
    return staleReason(r, dataset) ? null : r;
  };
  const putGates = (datasetId: string, record: GateRecord, gates: Gate[]) =>
    set((s) => ({ records: { ...s.records, [datasetId]: { ...record, gates, updatedAt: now() } } }));

  return {
    records: {},
    active: null,
    shown: null,
    filterCells: {},
    panelOpen: false,
    setPanelOpen: (open) => set({ panelOpen: open }),
    addGate: (dataset, draft, cells) => {
      const record = usable(dataset);
      if (!record || record.gates.length >= MAX_GATES) return null;
      if (gateNameError(draft.name, record.gates)) return null;
      const gate: Gate = { id: newGateId(), name: draft.name.trim(), parentId: draft.parentId, def: draft.def, createdAt: now() };
      putGates(dataset.id, record, [...record.gates, gate]);
      if (cells && draft.def.kind === "filter") get().setFilterCells(dataset.id, gate.id, cells);
      return gate.id;
    },
    renameGate: (datasetId, id, name) => {
      const record = get().records[datasetId];
      if (!record || gateNameError(name, record.gates, id)) return;
      putGates(datasetId, record, record.gates.map((g) => (g.id === id ? { ...g, name: name.trim() } : g)));
    },
    deleteGate: (datasetId, id, mode, evals) => {
      const record = get().records[datasetId];
      if (!record) return null;
      const result = deleteFromTree(record.gates, id, mode, evals);
      const left = new Set(result.gates.map((g) => g.id));
      putGates(datasetId, record, result.gates);
      set((s) => ({
        active: s.active && s.active.datasetId === datasetId && !left.has(s.active.gateId) ? null : s.active,
        shown: s.shown && s.shown.datasetId === datasetId && !left.has(s.shown.gateId) ? null : s.shown,
      }));
      return result;
    },
    setActive: (datasetId, gateId) => set({ active: gateId ? { datasetId, gateId } : null }),
    discard: (datasetId) =>
      set((s) => {
        const records = { ...s.records };
        delete records[datasetId];
        return {
          records,
          active: s.active?.datasetId === datasetId ? null : s.active,
          shown: s.shown?.datasetId === datasetId ? null : s.shown,
        };
      }),
    mergeGates: (dataset, incoming) => {
      const current = get().records[dataset.id];
      const stale = current ? staleReason(current, dataset) : null;
      const base = current && !stale ? current.gates : [];
      const ids = new Set(base.map((g) => g.id));
      const byId = new Map(incoming.map((g) => [g.id, g]));
      const merged = [...base.map((g) => byId.get(g.id) ?? g), ...incoming.filter((g) => !ids.has(g.id))];
      const { gates, dropped } = validateGates(merged.slice(0, MAX_GATES));
      if (merged.length > MAX_GATES) {
        for (const g of merged.slice(MAX_GATES)) dropped.push({ name: g.name, reason: `at most ${MAX_GATES} gates are kept` });
      }
      const record = current && !stale ? current : recordFor(dataset, []);
      putGates(dataset.id, record, gates);
      get().clearFilterCells(dataset.id);
      return {
        added: incoming.filter((g) => !ids.has(g.id)).length,
        replaced: incoming.filter((g) => ids.has(g.id)).length,
        replacedStale: stale && current ? current.gates.length : 0,
        dropped,
      };
    },
    setFilterCells: (datasetId, gateId, result) =>
      set((s) => ({ filterCells: { ...s.filterCells, [datasetId]: { ...s.filterCells[datasetId], [gateId]: result } } })),
    clearFilterCells: (datasetId) => {
      generations.set(datasetId, filterGeneration(datasetId) + 1);
      set((s) => {
        if (!s.filterCells[datasetId]) return s;
        const filterCells = { ...s.filterCells };
        delete filterCells[datasetId];
        return { filterCells };
      });
    },
  };
});

/**
 * Make a gate the selection (its cells, so DE "selection vs rest", "Label
 * selection…" and the selection histogram work on it) and the active gate.
 * A top-level filter gate keeps its filter as the selection's, as a filter's
 * own "Select" would. Clicking the gate that is already shown clears it.
 */
export function selectGate(datasetId: string, gate: Gate, ev: GateEval | undefined) {
  if (ev?.status !== "ok" || ev.n === 0) return;
  const selection = useSelectionStore.getState();
  const { active, shown } = useGatesStore.getState();
  if (
    active?.datasetId === datasetId &&
    active.gateId === gate.id &&
    shown?.gateId === gate.id &&
    shown.cells === selection.selectedCellIndices
  ) {
    selection.clearSelection();
    return;
  }
  const cells = bitsetToSet(ev.cells);
  selection.setHighlight(null);
  selection.setSelection(cells, gate.def.kind === "filter" && !gate.parentId ? gate.def.filter : null);
  useGatesStore.setState({ active: { datasetId, gateId: gate.id }, shown: { datasetId, gateId: gate.id, cells } });
}

// ---------------------------------------------------------------------------
// localStorage: every access in try/catch, at most GATES_MAX_CHARS
// ---------------------------------------------------------------------------

export const GATES_STORAGE_KEY = "scview.gates.v1";
/** About 2 MB of localStorage's usual 5 MB: a 200k-cell gate saved by position takes ~33 kB. */
export const GATES_MAX_CHARS = 2_000_000;

export interface GatesStorageStatus {
  /** The last save failed: storage blocked or full */
  blocked: boolean;
  /** Datasets whose gates were left out of storage to stay under GATES_MAX_CHARS */
  notKept: string[];
}

export const useGatesStorageStatus = create<GatesStorageStatus>(() => ({ blocked: false, notKept: [] }));

const DATASET_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

/** The stored text for `records`: the most recently changed first, while under `maxChars`. */
export function serializeGates(
  records: Record<string, GateRecord>,
  maxChars = GATES_MAX_CHARS,
): { json: string; notKept: string[] } {
  const head = '{"version":1,"datasets":{';
  let size = head.length + 2;
  const parts: string[] = [];
  const notKept: string[] = [];
  const entries = Object.entries(records)
    .filter(([, r]) => r.gates.length > 0)
    .sort(([, a], [, b]) => b.updatedAt.localeCompare(a.updatedAt));
  for (const [id, r] of entries) {
    const part = `${JSON.stringify(id)}:${JSON.stringify({
      fingerprint: r.fingerprint,
      n_cells: r.nCells,
      updated_at: r.updatedAt,
      gates: r.gates.map(toSessionGate),
    })}`;
    if (size + part.length + 1 > maxChars) {
      notKept.push(id);
      continue;
    }
    size += part.length + 1;
    parts.push(part);
  }
  return { json: `${head}${parts.join(",")}}}`, notKept };
}

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && !Array.isArray(x);

/** Stored records checked field by field: a malformed dataset entry is dropped, never trusted. */
export function readStoredGates(raw: string | null): Record<string, GateRecord> {
  const out: Record<string, GateRecord> = {};
  if (!raw) return out;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return out;
  }
  if (!isObj(parsed) || parsed.version !== 1 || !isObj(parsed.datasets)) return out;
  for (const [id, entry] of Object.entries(parsed.datasets)) {
    if (!DATASET_ID_RE.test(id) || !isObj(entry)) continue;
    const { fingerprint, n_cells: nCells, updated_at: updatedAt } = entry;
    if (fingerprint !== null && (typeof fingerprint !== "string" || fingerprint.length > 128)) continue;
    if (typeof nCells !== "number" || !Number.isInteger(nCells) || nCells < 0) continue;
    try {
      const { gates } = validateGates(parseGateList(entry.gates).map(fromSessionGate));
      if (gates.length === 0) continue;
      out[id] = { fingerprint, nCells, gates, updatedAt: typeof updatedAt === "string" ? updatedAt : "" };
    } catch {
      /* a malformed entry: drop it */
    }
  }
  return out;
}

let loading = false;

function saveGates(records: Record<string, GateRecord>) {
  if (loading) return;
  const { json, notKept } = serializeGates(records);
  let blocked = false;
  try {
    if (Object.values(records).some((r) => r.gates.length > 0)) window.localStorage.setItem(GATES_STORAGE_KEY, json);
    else window.localStorage.removeItem(GATES_STORAGE_KEY);
  } catch {
    blocked = true;
  }
  const prev = useGatesStorageStatus.getState();
  if (prev.blocked !== blocked || prev.notKept.join("\u0000") !== notKept.join("\u0000")) {
    useGatesStorageStatus.setState({ blocked, notKept });
  }
}

/** Read the stored gates into the store (on start; tests call it again). */
export function loadGatesFromStorage() {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(GATES_STORAGE_KEY);
  } catch {
    raw = null;
  }
  loading = true;
  try {
    useGatesStore.setState({ records: readStoredGates(raw), active: null, shown: null, filterCells: {} });
  } finally {
    loading = false;
  }
}

useGatesStore.subscribe((s, prev) => {
  if (s.records !== prev.records) saveGates(s.records);
});

// A cleared selection leaves no gate active (new gates then go at the top).
useSelectionStore.subscribe((s, prev) => {
  if (prev.selectedCellIndices && !s.selectedCellIndices) {
    const g = useGatesStore.getState();
    if (g.active || g.shown) useGatesStore.setState({ active: null, shown: null });
  }
});

// The open dataset reloaded (a label, a renamed category, a new clustering):
// filter gates may match other cells now, so they are asked again.
useDatasetStore.subscribe((s, prev) => {
  const d = s.currentDataset;
  if (d && prev.currentDataset && d !== prev.currentDataset && d.id === prev.currentDataset.id) {
    useGatesStore.getState().clearFilterCells(d.id);
  }
});

loadGatesFromStorage();
