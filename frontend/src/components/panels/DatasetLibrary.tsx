import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Pencil, Trash2 } from "lucide-react";

import {
  deleteDataset,
  getDataset,
  listDatasets,
  openFailureMessage,
  pruneDatasets,
  renameDataset,
} from "@/api/datasets";
import { DatasetDownloadMenu } from "@/components/common/DatasetDownloadMenu";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { formatDate, formatNumber } from "@/lib/formatting";

/** Same limit the backend enforces (DATASET_NAME_MAX). */
const NAME_MAX = 120;

/** The "Your datasets" library: reopen / select / rename / delete / bulk-delete / clean up. */
export function DatasetLibrary() {
  const { availableDatasets, setAvailableDatasets, currentDataset, setCurrentDataset } =
    useDatasetStore();
  const setPanel = useViewStore((s) => s.setPanel);

  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [cleaning, setCleaning] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  // Enter and the blur that follows both commit; the ref makes the second a no-op.
  const renamingRef = useRef<string | null>(null);

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

  const startRename = useCallback((id: string, name: string) => {
    renamingRef.current = id;
    setDraftName(name);
    setRenamingId(id);
  }, []);

  const cancelRename = useCallback(() => {
    renamingRef.current = null;
    setRenamingId(null);
  }, []);

  const commitRename = useCallback(
    async (id: string, oldName: string) => {
      if (renamingRef.current !== id) return;
      renamingRef.current = null;
      setRenamingId(null);
      const name = draftName.trim();
      if (!name || name === oldName) return;
      setError(null);
      try {
        const res = await renameDataset(id, name);
        // Read the store fresh: the list may have been refreshed while the request ran.
        const { availableDatasets: list, currentDataset: current } = useDatasetStore.getState();
        setAvailableDatasets(list.map((d) => (d.id === id ? { ...d, name: res.name } : d)));
        // Same id, so this updates the header without resetting the view.
        if (current?.id === id) setCurrentDataset({ ...current, name: res.name });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Rename failed.");
      }
    },
    [draftName, setAvailableDatasets, setCurrentDataset]
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
            <div className="flex min-w-0 items-center gap-3">
              <input
                type="checkbox"
                checked={selectedIds.has(ds.id)}
                onChange={() => toggleSelected(ds.id)}
                title="Select"
                className="h-4 w-4 cursor-pointer"
              />
              {renamingId === ds.id ? (
                <div className="min-w-0 flex-1">
                  <input
                    autoFocus
                    value={draftName}
                    maxLength={NAME_MAX}
                    aria-label="Dataset name"
                    onChange={(e) => setDraftName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") void commitRename(ds.id, ds.name);
                      else if (e.key === "Escape") cancelRename();
                    }}
                    onBlur={() => void commitRename(ds.id, ds.name)}
                    className="w-full min-w-[16rem] rounded border border-primary px-1.5 py-0.5 text-sm focus:outline-none"
                  />
                  <p className="mt-0.5 text-xs text-slate-400">Enter to save, Esc to cancel</p>
                </div>
              ) : (
                <div className="group flex min-w-0 items-start gap-1.5">
                  <button
                    onClick={async () => {
                      setError(null);
                      try {
                        const dataset = await getDataset(ds.id);
                        setCurrentDataset(dataset);
                        if (dataset.status === "ready") setPanel("unified");
                      } catch (e) {
                        // A 507's detail names the run that holds the memory (or the limit).
                        setError(openFailureMessage(e));
                      }
                    }}
                    className="min-w-0 text-left"
                  >
                    <p className="break-words text-sm font-medium text-slate-900">{ds.name}</p>
                    <p className="text-xs text-slate-500">
                      {[
                        ds.n_cells != null && `${formatNumber(ds.n_cells)} cells`,
                        ds.n_genes != null && `${formatNumber(ds.n_genes)} genes`,
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                      {formatDate(ds.created_at) && (
                        <span title={`Added ${new Date(ds.created_at).toLocaleString()}`}>
                          {ds.n_cells != null || ds.n_genes != null ? " · " : ""}
                          added {formatDate(ds.created_at)}
                        </span>
                      )}
                    </p>
                  </button>
                  <button
                    onClick={() => startRename(ds.id, ds.name)}
                    title="Rename dataset"
                    aria-label={`Rename ${ds.name}`}
                    className="mt-0.5 flex-shrink-0 rounded p-0.5 text-slate-300 opacity-0 transition-opacity hover:text-primary focus:opacity-100 group-hover:opacity-100"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}
            </div>
            <div className="flex flex-shrink-0 items-center gap-1">
              {ds.status === "ready" && <DatasetDownloadMenu datasetId={ds.id} compact />}
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
          </div>
        ))}
      </div>
    </div>
  );
}
