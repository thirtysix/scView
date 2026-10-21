import { describe, it, expect, vi, beforeEach } from "vitest";

const api = vi.hoisted(() => ({ getDataset: vi.fn(), cellsMatching: vi.fn(), apiFetch: vi.fn() }));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
vi.mock("@/api/cells", () => ({ cellsMatching: api.cellsMatching, filterCells: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch: api.apiFetch }));

import type { DatasetInfo } from "@/api/types";
import type { CellFilter } from "@/lib/cellFilter";
import { bitsetFromIndices, bitsetToSelection } from "@/lib/gates";
import { parseSession, sessionJson, SessionError, SESSION_FORMAT, SESSION_VERSION } from "@/lib/session";
import { captureSession, restoreSession } from "@/lib/sessionRestore";
import { useDatasetStore } from "@/stores/datasetStore";
import { loadGatesFromStorage, useGatesStore } from "@/stores/gatesStore";
import { useSelectionStore } from "@/stores/selectionStore";

const kang = (extra: Partial<DatasetInfo> = {}): DatasetInfo => ({
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 100,
  n_genes: 50,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "CD14 Mono"] }],
  cells_fingerprint: "fp-kang",
  created_at: "",
  ...extra,
});
const cellsSel = (cells: number[], n: number) => bitsetToSelection(bitsetFromIndices(cells, n));
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, k) => a + k);
const filter: CellFilter = { op: "and", layer: "", items: [{ kind: "gene", gene: "ISG15", op: ">", value: 2 }] };

/** Monocytes (lasso) > ISG-high (filter), and "Monocytes not ISG-high" inside Monocytes. */
function buildGates() {
  useDatasetStore.getState().setCurrentDataset(kang());
  const store = useGatesStore.getState();
  const mono = store.addGate(kang(), {
    name: "Monocytes",
    parentId: null,
    def: { kind: "cells", selection: cellsSel(range(0, 40), 100) },
  })!;
  const isg = store.addGate(kang(), { name: "ISG-high", parentId: mono, def: { kind: "filter", filter } })!;
  store.addGate(kang(), { name: "Monocytes not ISG-high", parentId: mono, def: { kind: "combine", op: "not", operands: [mono, isg] } });
  return { mono, isg };
}

function resetAll() {
  localStorage.clear();
  loadGatesFromStorage();
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  useSelectionStore.setState({ selectedCellIndices: null, selectionFilter: null, highlightedGroup: null });
}

beforeEach(() => {
  vi.clearAllMocks();
  resetAll();
  api.getDataset.mockResolvedValue(kang());
  api.apiFetch.mockResolvedValue({ results: [] });
});

describe("gates in session files", () => {
  it("an old session file without gates still loads, and restores no gates", async () => {
    const old = { format: SESSION_FORMAT, version: SESSION_VERSION, dataset: { id: "kang", n_cells: 100 } };
    const session = parseSession(old);
    expect(session.gates).toBeUndefined();
    const report = await restoreSession(session);
    expect(report.notes).toEqual([]);
    expect(useGatesStore.getState().records.kang).toBeUndefined();
  });

  it("a malformed gates field rejects the file with the reason", () => {
    const bad = { format: SESSION_FORMAT, version: SESSION_VERSION, gates: [{ id: "a", name: "A", def: { kind: "nope" } }] };
    expect(() => parseSession(bad)).toThrow(SessionError);
    expect(() => parseSession(bad)).toThrow(/gates\[0\]\.def\.kind/);
  });

  it("saves the tree and opens it again on the same cells", async () => {
    const { mono, isg } = buildGates();
    const file = sessionJson(captureSession(new Date("2026-10-21T20:00:00Z")));
    const session = parseSession(JSON.parse(file));
    expect(session.gates?.map((g) => [g.name, g.parent_id])).toEqual([
      ["Monocytes", null],
      ["ISG-high", mono],
      ["Monocytes not ISG-high", mono],
    ]);
    expect(session.gates?.[2]?.def).toEqual({ kind: "combine", op: "not", operands: [mono, isg] });

    resetAll();
    const report = await restoreSession(session);
    expect(report.notes).toEqual([]);
    const record = useGatesStore.getState().records.kang!;
    expect(record.fingerprint).toBe("fp-kang");
    expect(record.gates.map((g) => [g.name, g.parentId])).toEqual([
      ["Monocytes", null],
      ["ISG-high", mono],
      ["Monocytes not ISG-high", mono],
    ]);
  });

  it("does not save stale gates (drawn on other cells)", () => {
    buildGates();
    useDatasetStore.getState().setCurrentDataset(null);
    useDatasetStore.getState().setCurrentDataset(kang({ n_cells: 90, cells_fingerprint: "fp-filtered" }));
    expect(captureSession().gates).toBeUndefined();
  });

  it("on other cells brings back filter gates only, and says which gates were left out", async () => {
    buildGates();
    const session = parseSession(JSON.parse(sessionJson(captureSession())));
    resetAll();
    api.getDataset.mockResolvedValue(kang({ n_cells: 90, cells_fingerprint: "fp-filtered" }));
    const report = await restoreSession(session);
    // ISG-high is inside a lasso gate and the combination uses it: all three need the cells.
    expect(report.notes.find((n) => n.startsWith("Gates"))).toBe(
      "Gates “Monocytes”, “ISG-high”, “Monocytes not ISG-high”: not restored, the cells differ and they were saved by cell position (or inside or built from such a gate).",
    );
    expect(useGatesStore.getState().records.kang?.gates ?? []).toEqual([]);

    // A top-level filter gate is portable.
    resetAll();
    const portable = {
      ...session,
      gates: [{ id: "f1", name: "ISG-high anywhere", parent_id: null, def: { kind: "filter" as const, filter } }],
    };
    const report2 = await restoreSession(portable);
    expect(report2.notes.filter((n) => n.startsWith("Gates"))).toEqual([]);
    const record = useGatesStore.getState().records.kang!;
    expect(record.nCells).toBe(90);
    expect(record.gates.map((g) => g.name)).toEqual(["ISG-high anywhere"]);
  });

  it("notes gates that refer to gates not in the file", async () => {
    const session = parseSession({
      format: SESSION_FORMAT,
      version: SESSION_VERSION,
      dataset: { id: "kang", n_cells: 100, cells_fingerprint: "fp-kang" },
      gates: [
        { id: "a", name: "A", parent_id: null, def: { kind: "cells", selection: { n_cells: 100, indices: [1, 2] } } },
        { id: "b", name: "B", parent_id: "missing", def: { kind: "cells", selection: { n_cells: 100, indices: [1] } } },
      ],
    });
    const report = await restoreSession(session);
    expect(report.notes).toContain("Gates not restored: “B” (it refers to a gate that is not there).");
    expect(useGatesStore.getState().records.kang!.gates.map((g) => g.name)).toEqual(["A"]);
  });
});
