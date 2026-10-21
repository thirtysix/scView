import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import type { DatasetInfo } from "@/api/types";
import type { CellFilter } from "@/lib/cellFilter";
import { bitsetFromIndices, bitsetToSelection, evaluateGates, type Gate } from "@/lib/gates";
import { useDatasetStore } from "@/stores/datasetStore";
import {
  GATES_STORAGE_KEY,
  loadGatesFromStorage,
  readStoredGates,
  selectGate,
  serializeGates,
  staleReason,
  useGatesStorageStatus,
  useGatesStore,
  type GateRecord,
} from "@/stores/gatesStore";
import { useSelectionStore } from "@/stores/selectionStore";

const ds = (extra: Partial<DatasetInfo> = {}): DatasetInfo => ({
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 100,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [],
  cells_fingerprint: "fp-1",
  created_at: "",
  ...extra,
});
const cellsSel = (cells: number[], n: number) => bitsetToSelection(bitsetFromIndices(cells, n));
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, k) => a + k);
const cellsDef = (cells: number[], n = 100) => ({ kind: "cells" as const, selection: cellsSel(cells, n) });
const filter: CellFilter = { op: "and", layer: "", items: [{ kind: "gene", gene: "ISG15", op: ">", value: 2 }] };

function reset() {
  localStorage.clear();
  loadGatesFromStorage();
  useGatesStorageStatus.setState({ blocked: false, notKept: [] });
  useSelectionStore.setState({ selectedCellIndices: null, selectionFilter: null, highlightedGroup: null });
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
}

describe("gatesStore", () => {
  beforeEach(reset);
  afterEach(() => vi.restoreAllMocks());

  it("adds gates under the dataset with its cells' fingerprint, and keeps them in localStorage", () => {
    const store = useGatesStore.getState();
    const mono = store.addGate(ds(), { name: "Monocytes", parentId: null, def: cellsDef(range(0, 40)) })!;
    store.addGate(ds(), { name: "ISG-high", parentId: mono, def: { kind: "filter", filter } });
    const record = useGatesStore.getState().records.kang!;
    expect(record.fingerprint).toBe("fp-1");
    expect(record.nCells).toBe(100);
    expect(record.gates.map((g) => [g.name, g.parentId])).toEqual([
      ["Monocytes", null],
      ["ISG-high", mono],
    ]);
    const stored = JSON.parse(localStorage.getItem(GATES_STORAGE_KEY)!);
    expect(stored.version).toBe(1);
    expect(stored.datasets.kang.gates[1].parent_id).toBe(mono);

    // A reload reads them back (loading replaces the store's records without writing).
    loadGatesFromStorage();
    expect(localStorage.getItem(GATES_STORAGE_KEY)).not.toBeNull();
    expect(useGatesStore.getState().records.kang!.gates.map((g) => g.name)).toEqual(["Monocytes", "ISG-high"]);
  });

  it("refuses a duplicate name and an empty one", () => {
    const store = useGatesStore.getState();
    expect(store.addGate(ds(), { name: "A", parentId: null, def: cellsDef([1]) })).not.toBeNull();
    expect(store.addGate(ds(), { name: " a ", parentId: null, def: cellsDef([2]) })).toBeNull();
    expect(store.addGate(ds(), { name: "", parentId: null, def: cellsDef([2]) })).toBeNull();
  });

  it("marks gates stale when the dataset's cells changed, and does not add to them", () => {
    const store = useGatesStore.getState();
    store.addGate(ds(), { name: "A", parentId: null, def: cellsDef([1]) });
    const record = useGatesStore.getState().records.kang!;
    expect(staleReason(record, ds())).toBeNull();
    expect(staleReason(record, ds({ n_cells: 90 }))).toMatch(/drawn on 100 cells and the dataset now has 90/);
    expect(staleReason(record, ds({ cells_fingerprint: "fp-2" }))).toMatch(/not the same cells/);
    expect(store.addGate(ds({ n_cells: 90 }), { name: "B", parentId: null, def: cellsDef([1], 90) })).toBeNull();
    store.discard("kang");
    expect(useGatesStore.getState().records.kang).toBeUndefined();
    expect(localStorage.getItem(GATES_STORAGE_KEY)).toBeNull();
  });

  it("keeps working when localStorage is blocked, and says the gates are not kept", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("QuotaExceededError");
    });
    const id = useGatesStore.getState().addGate(ds(), { name: "A", parentId: null, def: cellsDef([1]) });
    expect(id).not.toBeNull();
    expect(useGatesStore.getState().records.kang!.gates).toHaveLength(1);
    expect(useGatesStorageStatus.getState().blocked).toBe(true);
  });

  it("starts empty when localStorage can't be read or holds junk", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("SecurityError");
    });
    loadGatesFromStorage();
    expect(useGatesStore.getState().records).toEqual({});
    vi.restoreAllMocks();
    expect(readStoredGates("{not json")).toEqual({});
    expect(readStoredGates(JSON.stringify({ version: 2, datasets: {} }))).toEqual({});
  });

  it("drops malformed stored entries field by field, keeping the good ones", () => {
    const good = { id: "g1", name: "A", parent_id: null, def: cellsDef([1, 2]) };
    const raw = JSON.stringify({
      version: 1,
      datasets: {
        kang: { fingerprint: "fp-1", n_cells: 100, updated_at: "t", gates: [good, { ...good, id: "g2", parent_id: "gone" }] },
        "../evil": { fingerprint: null, n_cells: 100, gates: [good] },
        bad: { fingerprint: 5, n_cells: 100, gates: [good] },
        junk: { fingerprint: null, n_cells: 100, gates: [{ id: "x" }] },
      },
    });
    const records = readStoredGates(raw);
    expect(Object.keys(records)).toEqual(["kang"]);
    expect(records.kang!.gates.map((g) => g.id)).toEqual(["g1"]); // g2's parent is not there
  });

  it("caps the stored size, keeping the most recently changed datasets", () => {
    const gate: Gate = { id: "g1", name: "A", parentId: null, def: cellsDef(range(0, 100)), createdAt: "" };
    const records: Record<string, GateRecord> = {
      old: { fingerprint: null, nCells: 100, gates: [gate], updatedAt: "2026-01-01" },
      recent: { fingerprint: null, nCells: 100, gates: [gate], updatedAt: "2026-10-01" },
    };
    const one = serializeGates({ recent: records.recent! }).json.length;
    const { json, notKept } = serializeGates(records, one + 10);
    expect(notKept).toEqual(["old"]);
    expect(Object.keys(JSON.parse(json).datasets)).toEqual(["recent"]);
    expect(serializeGates(records, 50).notKept).toEqual(["recent", "old"]);
  });

  it("clears the active gate when the selection is cleared", () => {
    const store = useGatesStore.getState();
    const id = store.addGate(ds(), { name: "A", parentId: null, def: cellsDef(range(0, 10)) })!;
    const ev = evaluateGates(useGatesStore.getState().records.kang!.gates, 100, () => undefined);
    selectGate("kang", useGatesStore.getState().records.kang!.gates[0]!, ev.get(id));
    expect(useGatesStore.getState().active).toEqual({ datasetId: "kang", gateId: id });
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual(range(0, 10));
    useSelectionStore.getState().clearSelection();
    expect(useGatesStore.getState().active).toBeNull();
  });

  it("selecting a gate twice clears it; a top-level filter gate keeps its filter", () => {
    const store = useGatesStore.getState();
    const id = store.addGate(ds(), { name: "ISG", parentId: null, def: { kind: "filter", filter } }, bitsetFromIndices([1, 2, 3], 100))!;
    const gates = useGatesStore.getState().records.kang!.gates;
    const ev = evaluateGates(gates, 100, (g) => useGatesStore.getState().filterCells.kang?.[g.id]);
    useSelectionStore.getState().setHighlight({ column: "cell_type", value: "B" });
    selectGate("kang", gates[0]!, ev.get(id));
    expect(useSelectionStore.getState().selectionFilter).toEqual(filter);
    expect(useSelectionStore.getState().highlightedGroup).toBeNull();
    selectGate("kang", gates[0]!, ev.get(id));
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
  });

  it("deleting the active gate leaves none active", () => {
    const store = useGatesStore.getState();
    const id = store.addGate(ds(), { name: "A", parentId: null, def: cellsDef(range(0, 10)) })!;
    store.setActive("kang", id);
    store.deleteGate("kang", id, "subtree", new Map());
    expect(useGatesStore.getState().active).toBeNull();
    expect(useGatesStore.getState().records.kang!.gates).toEqual([]);
  });

  it("merges a session's gates: same id replaced, others kept; stale ones replaced", () => {
    const store = useGatesStore.getState();
    const a = store.addGate(ds(), { name: "A", parentId: null, def: cellsDef([1]) })!;
    store.addGate(ds(), { name: "B", parentId: null, def: cellsDef([2]) });
    const incoming: Gate[] = [
      { id: a, name: "A renamed", parentId: null, def: cellsDef([1, 5]), createdAt: "" },
      { id: "new1", name: "C", parentId: a, def: { kind: "filter", filter }, createdAt: "" },
    ];
    const res = store.mergeGates(ds(), incoming);
    expect(res).toMatchObject({ added: 1, replaced: 1, replacedStale: 0, dropped: [] });
    expect(useGatesStore.getState().records.kang!.gates.map((g) => g.name)).toEqual(["A renamed", "B", "C"]);

    // On other cells now: the local gates are stale and the session's replace them.
    const res2 = store.mergeGates(ds({ n_cells: 80, cells_fingerprint: "fp-2" }), [incoming[1]!].map((g) => ({ ...g, parentId: null })));
    expect(res2.replacedStale).toBe(3);
    const record = useGatesStore.getState().records.kang!;
    expect(record.nCells).toBe(80);
    expect(record.gates.map((g) => g.name)).toEqual(["C"]);
  });

  it("asks filter gates again when the open dataset reloads", () => {
    useDatasetStore.setState({ currentDataset: ds(), currentDatasetId: "kang" });
    useGatesStore.getState().setFilterCells("kang", "g1", bitsetFromIndices([1], 100));
    useDatasetStore.getState().setCurrentDataset(ds({ name: "Kang (refreshed)" }));
    expect(useGatesStore.getState().filterCells.kang).toBeUndefined();
  });
});
