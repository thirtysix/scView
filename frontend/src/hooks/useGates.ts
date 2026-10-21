import { useEffect, useMemo } from "react";
import { filterCells as fetchFilterCells } from "@/api/cells";
import type { DatasetInfo } from "@/api/types";
import { bitsetFromMask, emptyBitset, evaluateGates, gateTree, type Gate, type GateEval, type TreeRow } from "@/lib/gates";
import { filterGeneration, staleReason, useGatesStore, type GateRecord } from "@/stores/gatesStore";

const NO_GATES: Gate[] = [];

export interface GatesView {
  record: GateRecord | undefined;
  /** The gates in use: none while they are stale */
  gates: Gate[];
  tree: TreeRow[];
  /** Why the saved gates are not applied, or null */
  stale: string | null;
  evals: Map<string, GateEval>;
}

/** The open dataset's gates, evaluated (each gate once per change of gates or filter cells). */
export function useGates(dataset: DatasetInfo | null): GatesView {
  const id = dataset?.id ?? "";
  const record = useGatesStore((s) => (id ? s.records[id] : undefined));
  const filterCells = useGatesStore((s) => (id ? s.filterCells[id] : undefined));
  const nCells = dataset?.n_cells ?? 0;
  const fingerprint = dataset?.cells_fingerprint;
  const stale = record ? staleReason(record, { n_cells: nCells, cells_fingerprint: fingerprint }) : null;
  const gates = !record || stale ? NO_GATES : record.gates;
  const tree = useMemo(() => gateTree(gates), [gates]);
  const evals = useMemo(() => evaluateGates(gates, nCells, (g) => filterCells?.[g.id]), [gates, nCells, filterCells]);
  return { record, gates, tree, stale, evals };
}

const inFlight = new Set<string>();

/** Ask the backend for the cells of every filter gate not known yet, while mounted. */
export function useResolveFilterGates(dataset: DatasetInfo | null, gates: Gate[]) {
  const id = dataset?.id ?? "";
  const known = useGatesStore((s) => (id ? s.filterCells[id] : undefined));
  const nCells = dataset?.n_cells ?? 0;
  useEffect(() => {
    if (!id || nCells <= 0) return;
    for (const g of gates) {
      if (g.def.kind !== "filter" || known?.[g.id]) continue;
      const gen = filterGeneration(id);
      const key = `${id}\u0000${g.id}\u0000${gen}`;
      if (inFlight.has(key)) continue;
      inFlight.add(key);
      const filter = g.def.filter;
      fetchFilterCells(id, filter, true)
        .then((res) => {
          if (filterGeneration(id) !== gen) return;
          const result =
            res.n_cells !== nCells
              ? { error: `the dataset has ${res.n_cells.toLocaleString()} cells now; reopen it` }
              : res.mask
                ? bitsetFromMask(res.mask, nCells)
                : emptyBitset(nCells);
          useGatesStore.getState().setFilterCells(id, g.id, result);
        })
        .catch((e: unknown) => {
          if (filterGeneration(id) !== gen) return;
          useGatesStore.getState().setFilterCells(id, g.id, { error: e instanceof Error ? e.message : String(e) });
        })
        .finally(() => inFlight.delete(key));
    }
  }, [id, nCells, gates, known]);
}
