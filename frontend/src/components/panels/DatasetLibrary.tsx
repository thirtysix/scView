import { useCallback, useEffect, useState } from "react";
import { Loader2, Trash2 } from "lucide-react";

import { deleteDataset, getDataset, listDatasets, pruneDatasets } from "@/api/datasets";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { formatNumber } from "@/lib/formatting";

/** The "Your datasets" library: reopen / select / delete / bulk-delete / clean up. */
export function DatasetLibrary() {
  const { availableDatasets, setAvailableDatasets, currentDataset, setCurrentDataset } =
    useDatasetStore();
  const setPanel = useViewStore((s) => s.setPanel);

  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [cleaning, setCleaning] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);

  const toggleSelected = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const refreshList = useCallback(
    () => listDatasets().then(setAvailableDatasets).catch(() => {}),
    [setAvailableDatasets]
  );

  // Populate previous datasets on mount so they can be reopened without re-uploading.
  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  const handleDelete = useCallback(
    async (id: string, name: string) => {
      if (!window.confirm(`Delete "${name}"? This removes its files permanently.`)) return;
      setDeletingId(id);
      try {
        await deleteDataset(id);
        if (currentDataset?.id === id) setCurrentDataset(null);
        await refreshList();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to delete dataset.");
      } finally {
        setDeletingId(null);
      }
    },
    [currentDataset, setCurrentDataset, refreshList]
  );

  const handleCleanup = useCallback(async () => {
    setCleaning(true);
    setError(null);
    try {
      const { count } = await pruneDatasets();
      await refreshList();
      setError(count > 0 ? null : "No unavailable datasets to clean up.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Cleanup failed.");
    } finally {
      setCleaning(false);
    }
  }, [refreshList]);

  const handleDeleteSelected = useCallback(async () => {
    const ids = [...selectedIds];
    if (ids.length === 0) return;
    if (
      !window.confirm(
        `Delete ${ids.length} dataset${ids.length > 1 ? "s" : ""}? This removes their files permanently.`
      )
    )
      return;
    setBulkDeleting(true);
    setError(null);
    try {
      const results = await Promise.allSettled(ids.map((id) => deleteDataset(id)));
      if (currentDataset && selectedIds.has(currentDataset.id)) setCurrentDataset(null);
      setSelectedIds(new Set());
      await refreshList();
      const failed = results.filter((r) => r.status === "rejected").length;
      if (failed > 0) setError(`${failed} of ${ids.length} could not be deleted.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Bulk delete failed.");
    } finally {
      setBulkDeleting(false);
    }
  }, [selectedIds, currentDataset, setCurrentDataset, refreshList]);

  if (availableDatasets.length === 0) {
    return error ? (
      <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>
    ) : null;
  }

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <input
            type="checkbox"
            title="Select all"
            checked={selectedIds.size === availableDatasets.length}
            ref={(el) => {
              if (el)
                el.indeterminate =
                  selectedIds.size > 0 && selectedIds.size < availableDatasets.length;
            }}
            onChange={(e) =>
              setSelectedIds(
                e.target.checked ? new Set(availableDatasets.map((d) => d.id)) : new Set()
              )
            }
            className="h-4 w-4 cursor-pointer"
          />
          <h3 className="text-sm font-semibold text-slate-700">
            Your datasets
            {selectedIds.size > 0 && (
              <span className="ml-1 font-normal text-slate-400">
                ({selectedIds.size} selected)
              </span>
            )}
          </h3>
        </div>
        <div className="flex items-center gap-2">
          {selectedIds.size > 0 && (
            <button
              onClick={handleDeleteSelected}
              disabled={bulkDeleting}
              className="flex items-center gap-1.5 rounded-lg bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-700 disabled:opacity-50"
            >
              {bulkDeleting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Trash2 className="h-3.5 w-3.5" />
              )}
              Delete selected ({selectedIds.size})
            </button>
          )}
          <button
            onClick={handleCleanup}
            disabled={cleaning}
            title="Remove datasets whose files are missing or failed to convert"
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-500 hover:bg-slate-50 hover:text-slate-700 disabled:opacity-50"
          >
            {cleaning ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Trash2 className="h-3.5 w-3.5" />
            )}
            Clean up unavailable
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-2 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>
      )}

      <div className="space-y-2">
        {availableDatasets.map((ds) => (
          <div
            key={ds.id}
            className={`flex items-center justify-between rounded-lg border bg-white p-3 ${
              selectedIds.has(ds.id) ? "border-primary/50 bg-primary/5" : "border-slate-200"
            }`}
          >
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={selectedIds.has(ds.id)}
                onChange={() => toggleSelected(ds.id)}
                title="Select"
                className="h-4 w-4 cursor-pointer"
              />
              <button
                onClick={async () => {
                  const dataset = await getDataset(ds.id);
                  setCurrentDataset(dataset);
                  if (dataset.status === "ready") setPanel("unified");
                }}
                className="text-left"
              >
                <p className="text-sm font-medium text-slate-900">{ds.name}</p>
                <p className="text-xs text-slate-500">
                  {ds.n_cells != null && `${formatNumber(ds.n_cells)} cells`}
                  {ds.n_genes != null && ` · ${formatNumber(ds.n_genes)} genes`}
                </p>
              </button>
            </div>
            <button
              onClick={() => handleDelete(ds.id, ds.name)}
              disabled={deletingId === ds.id}
              title="Delete dataset"
              className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-red-500 disabled:opacity-50"
            >
              {deletingId === ds.id ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4" />
              )}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
