import { useEffect, useMemo, useRef, useState } from "react";
import { CopyPlus, Loader2 } from "lucide-react";
import { getDataset, listDatasets } from "@/api/datasets";
import { makeSubset, type SubsetResult, type SubsetSelector } from "@/api/subset";
import type { DatasetInfo } from "@/api/types";
import { Help } from "@/components/common/Help";
import { useDialogFocus } from "@/hooks/useDialogFocus";
import type { CellFilter } from "@/lib/cellFilter";
import { formatNumber } from "@/lib/formatting";
import { encodeMask } from "@/lib/session";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

/** At least this many cells (backend core/subset.py MIN_CELLS). */
export const SUBSET_MIN_CELLS = 50;

/** The request's selector: the filter a selection came from, else the cells themselves. */
export function subsetSelector(
  indices: Set<number>,
  filter: CellFilter | null,
  nCells: number,
): SubsetSelector {
  return filter ? { filter } : { mask: encodeMask(indices, nCells) };
}

/** The name the server would give (core/subset.py default_name), shown to edit. */
export function defaultSubsetName(parent: string, n: number, byFilter: boolean): string {
  const name = `${parent} - ${n.toLocaleString("en-US")} ${byFilter ? "filtered" : "selected"} cells`;
  return name.length <= 120 ? name : `${name.slice(0, 119)}…`;
}

/**
 * Toolbar control: "New dataset…" while cells are selected (a lasso, a legend group, a
 * filter). The dialog says what the new dataset holds and makes it; the parent is unchanged.
 */
export function SubsetSelectionControls() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const selected = useSelectionStore((s) => s.selectedCellIndices);
  const filter = useSelectionStore((s) => s.selectionFilter);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const hasSelection = !!selected && selected.size > 0;

  useEffect(() => {
    if (!hasSelection) setOpen(false);
  }, [hasSelection]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  if (!dataset || !hasSelection || !selected) return null;

  return (
    <div ref={ref} className="relative flex items-center">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Make a new dataset of the selected cells"
        className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
          open ? "bg-orange-50 text-orange-700 ring-1 ring-orange-300" : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        <CopyPlus className="h-3.5 w-3.5" />
        New dataset…
      </button>
      {open && (
        <SubsetForm dataset={dataset} indices={selected} filter={filter} onClose={() => setOpen(false)} />
      )}
    </div>
  );
}

interface SubsetFormProps {
  dataset: DatasetInfo;
  indices: Set<number>;
  filter: CellFilter | null;
  onClose: () => void;
}

function SubsetForm({ dataset, indices, filter, onClose }: SubsetFormProps) {
  const dialogRef = useRef<HTMLFormElement>(null);
  useDialogFocus(dialogRef, onClose);
  const n = indices.size;
  const selector = useMemo(
    () => subsetSelector(indices, filter, dataset.n_cells ?? 0),
    [indices, filter, dataset.n_cells],
  );
  const [name, setName] = useState(() => defaultSubsetName(dataset.name, n, !!filter));
  const [plan, setPlan] = useState<SubsetResult | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [made, setMade] = useState<SubsetResult | null>(null);
  const tooFew = n < SUBSET_MIN_CELLS;

  useEffect(() => {
    if (tooFew) return;
    const ctrl = new AbortController();
    setPlan(null);
    setPlanError(null);
    makeSubset(dataset.id, selector, { dryRun: true, signal: ctrl.signal })
      .then((r) => {
        if (!ctrl.signal.aborted) setPlan(r);
      })
      .catch((e: unknown) => {
        if (!ctrl.signal.aborted) setPlanError(e instanceof Error ? e.message : "Could not check the selection.");
      });
    return () => ctrl.abort();
  }, [dataset.id, selector, tooFew]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy || tooFew || planError || !name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await makeSubset(dataset.id, selector, { name: name.trim() });
      setMade(res);
      listDatasets()
        .then(useDatasetStore.getState().setAvailableDatasets)
        .catch(() => undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The new dataset could not be made.");
    } finally {
      setBusy(false);
    }
  };

  const openMade = async () => {
    if (!made?.dataset_id) return;
    try {
      const ds = await getDataset(made.dataset_id);
      useSelectionStore.getState().clearSelection();
      useDatasetStore.getState().setCurrentDataset(ds);
      if (ds.available_embeddings.includes("X_umap_parent")) useSettingsStore.getState().setEmbedding("X_umap_parent");
      useViewStore.getState().setPanel("assessment");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open it.");
    }
  };

  return (
    <form
      ref={dialogRef}
      role="dialog"
      aria-label="New dataset from the selection"
      onSubmit={(e) => void submit(e)}
      className="absolute left-0 top-full z-50 mt-1 w-80 space-y-2.5 rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg"
    >
      <div className="font-semibold text-slate-700">
        <Help id="plot.new-dataset" side="below">
          New dataset from {n.toLocaleString("en-US")} selected cell{n === 1 ? "" : "s"}
        </Help>
      </div>

      {made ? (
        <div className="space-y-2" data-testid="subset-made">
          <p className="rounded bg-emerald-50 px-2 py-1.5 text-[11px] text-emerald-800">
            Made &ldquo;{made.name}&rdquo;: {made.n_cells.toLocaleString("en-US")} cells
            {made.n_genes ? ` x ${made.n_genes.toLocaleString("en-US")} genes` : ""}. It is in the dataset list, ready to
            process; this dataset is unchanged.
          </p>
          {error && (
            <p role="alert" className="rounded bg-red-50 px-2 py-1 text-[11px] text-red-700">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <button type="button" onClick={onClose} className="rounded px-2 py-1 text-xs text-slate-500 hover:bg-slate-100">
              Close
            </button>
            <button
              type="button"
              onClick={() => void openMade()}
              className="rounded bg-blue-500 px-3 py-1 text-xs font-medium text-white hover:bg-blue-600"
            >
              Open it to process
            </button>
          </div>
        </div>
      ) : (
        <>
          {filter && <p className="text-[11px] text-slate-500">Chosen by the filter the selection came from.</p>}
          <label className="block space-y-1">
            <span className="text-[11px] font-medium text-slate-500">Name</span>
            <input
              value={name}
              autoFocus
              maxLength={120}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
            />
          </label>

          {tooFew && (
            <p role="alert" className="rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
              A new dataset needs at least {SUBSET_MIN_CELLS} cells; {formatNumber(n)} are selected.
            </p>
          )}
          {!tooFew && !plan && !planError && (
            <p className="flex items-center gap-1 text-[11px] text-slate-400">
              <Loader2 className="h-3 w-3 animate-spin" /> Checking what it would hold…
            </p>
          )}
          {planError && (
            <p role="alert" className="rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
              {planError}
            </p>
          )}
          {plan && (
            <div className="space-y-1.5 text-[11px]" data-testid="subset-plan">
              <div>
                <span className="font-medium text-slate-600">Carried: </span>
                <span className="text-slate-500">
                  {["the cells' counts", ...plan.carried].join("; ")}
                </span>
              </div>
              <div>
                <span className="font-medium text-slate-600">Not carried: </span>
                <span className="text-slate-500">{plan.not_carried.join("; ")}</span>
              </div>
            </div>
          )}
          {error && (
            <p role="alert" className="rounded bg-red-50 px-2 py-1 text-[11px] text-red-700">
              {error}
            </p>
          )}

          <div className="flex justify-end gap-2 pt-1">
            <button type="button" onClick={onClose} className="rounded px-2 py-1 text-xs text-slate-500 hover:bg-slate-100">
              Cancel
            </button>
            <button
              type="submit"
              disabled={busy || tooFew || !!planError || !plan || !name.trim()}
              className="flex items-center gap-1 rounded bg-blue-500 px-3 py-1 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-60"
            >
              {busy && <Loader2 className="h-3 w-3 animate-spin" />}
              Make dataset
            </button>
          </div>
        </>
      )}
    </form>
  );
}
