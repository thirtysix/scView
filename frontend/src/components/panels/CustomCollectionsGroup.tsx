import { useState, useEffect, useCallback, useRef } from "react";
import { Upload, Pencil, Trash2, Loader2, X } from "lucide-react";
import {
  listCustomCollections,
  uploadCustomCollection,
  renameCustomCollection,
  deleteCustomCollection,
  describeCoverage,
  collectionOptionLabel,
  pruneDeletedCustom,
  GeneSetUploadError,
  type CollectionChoice,
  type CustomCollection,
} from "@/api/genesets";
import { useCustomGeneSetsStore } from "@/stores/customGeneSetsStore";

const NAME_MAX = 120;
const ACCEPT = ".gmt,.csv,.tsv,.txt";

/** <option>s for a collection dropdown, custom collections in their own "Custom" group. */
export function CollectionOptions({ collections }: { collections: CollectionChoice[] }) {
  const msigdb = collections.filter((c) => c.category !== "Custom");
  const custom = collections.filter((c) => c.category === "Custom");
  const option = (c: CollectionChoice) => (
    <option key={c.id} value={c.id}>
      {collectionOptionLabel(c)}
    </option>
  );
  if (custom.length === 0) return <>{msigdb.map(option)}</>;
  return (
    <>
      {msigdb.length > 0 && <optgroup label="MSigDB">{msigdb.map(option)}</optgroup>}
      <optgroup label="Custom">{custom.map(option)}</optgroup>
    </>
  );
}

interface CustomCollectionsGroupProps {
  /** With a dataset, each collection says how many of its genes the dataset measures. */
  datasetId?: string | null;
  /** With a selection the rows are checkboxes (the enrichment tree); without, a plain list. */
  selected?: Set<string>;
  onChange?: (next: Set<string>) => void;
  /** Told the current collections after every fetch. */
  onLoaded?: (collections: CustomCollection[]) => void;
  compact?: boolean;
}

/**
 * The "Custom" group: the user's own gene-set collections, with upload, rename
 * and delete. A collection deleted here (or elsewhere) is dropped from the
 * selection, so an enrichment run never asks for one that no longer exists.
 */
export function CustomCollectionsGroup({
  datasetId,
  selected,
  onChange,
  onLoaded,
  compact = false,
}: CustomCollectionsGroupProps) {
  const version = useCustomGeneSetsStore((s) => s.version);
  const bump = useCustomGeneSetsStore((s) => s.bump);

  const [collections, setCollections] = useState<CustomCollection[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [notice, setNotice] = useState<string | null>(null);

  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const renamingRef = useRef<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Read through refs so a fetch never acts on a stale selection or callback.
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const onLoadedRef = useRef(onLoaded);
  onLoadedRef.current = onLoaded;

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    listCustomCollections(datasetId)
      .then((list) => {
        if (cancelled) return;
        setCollections(list);
        onLoadedRef.current?.(list);
        const sel = selectedRef.current;
        if (sel && onChangeRef.current) {
          const pruned = pruneDeletedCustom(sel, list.map((c) => c.id));
          if (pruned.size !== sel.size) onChangeRef.current(pruned);
        }
      })
      .catch(() => {
        if (!cancelled) setCollections([]);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, version]);

  const showError = useCallback((e: unknown, fallback: string) => {
    setNotice(null);
    if (e instanceof GeneSetUploadError) {
      setError(e.message);
      setProblems(e.problems);
    } else {
      setError(e instanceof Error ? e.message : fallback);
      setProblems([]);
    }
  }, []);

  const clearMessages = useCallback(() => {
    setError(null);
    setProblems([]);
    setNotice(null);
  }, []);

  const pickFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      clearMessages();
      setPendingFile(file);
      setUploadName(file.name.replace(/\.[^.]+$/, ""));
    },
    [clearMessages],
  );

  const cancelUpload = useCallback(() => {
    setPendingFile(null);
    if (fileRef.current) fileRef.current.value = "";
  }, []);

  const handleUpload = useCallback(async () => {
    if (!pendingFile) return;
    setIsUploading(true);
    clearMessages();
    try {
      const res = await uploadCustomCollection(pendingFile, uploadName.trim());
      setNotice(
        [`Added "${res.name}" (${res.n_sets} sets).`, ...(res.warnings ?? [])].join(" "),
      );
      cancelUpload();
      // Ready to use straight away in the enrichment tree.
      const sel = selectedRef.current;
      if (sel && onChangeRef.current) onChangeRef.current(new Set([...sel, res.id]));
      bump();
    } catch (e) {
      showError(e, "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }, [pendingFile, uploadName, clearMessages, cancelUpload, bump, showError]);

  const startRename = useCallback((c: CustomCollection) => {
    renamingRef.current = c.id;
    setDraftName(c.name);
    setRenamingId(c.id);
  }, []);

  const cancelRename = useCallback(() => {
    renamingRef.current = null;
    setRenamingId(null);
  }, []);

  const commitRename = useCallback(
    async (c: CustomCollection) => {
      if (renamingRef.current !== c.id) return; // Enter already committed; this is its blur
      renamingRef.current = null;
      setRenamingId(null);
      const name = draftName.trim();
      if (!name || name === c.name) return;
      clearMessages();
      try {
        await renameCustomCollection(c.id, name);
        bump();
      } catch (e) {
        showError(e, "Rename failed.");
      }
    },
    [draftName, clearMessages, bump, showError],
  );

  const handleDelete = useCallback(
    async (c: CustomCollection) => {
      if (
        !window.confirm(
          `Delete the custom collection "${c.name}"? Enrichment results already computed with it ` +
            "are kept and will show it as deleted.",
        )
      )
        return;
      setDeletingId(c.id);
      clearMessages();
      try {
        await deleteCustomCollection(c.id);
        bump();
      } catch (e) {
        showError(e, "Delete failed.");
      } finally {
        setDeletingId(null);
      }
    },
    [clearMessages, bump, showError],
  );

  const textSize = compact ? "text-[11px]" : "text-xs";
  const py = compact ? "py-0.5" : "py-1";
  const selectable = selected !== undefined && onChange !== undefined;

  return (
    <div className="space-y-1" data-testid="custom-collections">
      <div className={`flex items-center justify-between gap-2 ${py}`}>
        <span className={`font-semibold text-slate-600 ${textSize}`}>Custom</span>
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={isUploading}
          className="flex items-center gap-1 text-[10px] font-medium text-blue-600 hover:underline disabled:opacity-50"
          title="Upload a .gmt file, or a two-column gene_set,gene .csv/.tsv"
        >
          <Upload className="h-3 w-3" />
          Upload
        </button>
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPT}
          aria-label="Gene-set file"
          className="hidden"
          onChange={(e) => pickFile(e.target.files?.[0])}
        />
      </div>

      {pendingFile && (
        <div className="space-y-1 rounded border border-blue-100 bg-blue-50/50 p-1.5">
          <div className="truncate text-[10px] text-slate-500" title={pendingFile.name}>
            {pendingFile.name}
          </div>
          <div className="flex items-center gap-1">
            <input
              autoFocus
              value={uploadName}
              maxLength={NAME_MAX}
              aria-label="Collection name"
              placeholder="Collection name"
              onChange={(e) => setUploadName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleUpload();
                else if (e.key === "Escape") cancelUpload();
              }}
              className={`min-w-0 flex-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 ${textSize} focus:border-blue-400 focus:outline-none`}
            />
            <button
              type="button"
              onClick={() => void handleUpload()}
              disabled={isUploading || !uploadName.trim()}
              className="flex items-center gap-1 rounded bg-primary px-2 py-0.5 text-[11px] font-medium text-white hover:bg-primary/90 disabled:opacity-50"
            >
              {isUploading && <Loader2 className="h-3 w-3 animate-spin" />}
              Save
            </button>
            <button
              type="button"
              onClick={cancelUpload}
              aria-label="Cancel upload"
              className="rounded p-0.5 text-slate-400 hover:text-slate-600"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}

      {error && (
        <div role="alert" className="rounded border border-red-200 bg-red-50 p-1.5 text-[11px] text-red-700">
          <div>{error}</div>
          {problems.length > 0 && (
            <ul className="mt-1 list-disc space-y-0.5 pl-4">
              {problems.map((p, i) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      {notice && <div className="text-[10px] text-emerald-700">{notice}</div>}

      {isLoading && collections.length === 0 ? (
        <div className={`flex items-center gap-1 ${py} text-[10px] text-slate-400`}>
          <Loader2 className="h-3 w-3 animate-spin" /> Loading...
        </div>
      ) : collections.length === 0 ? (
        <div className={`${py} text-[10px] text-slate-400`}>
          None yet. Upload a .gmt, or a gene_set,gene .csv/.tsv (up to 10 MB).
        </div>
      ) : (
        collections.map((c) => (
          <div key={c.id} className={`group flex items-start gap-2 ${py} ${textSize} text-slate-700`}>
            {selectable && (
              <input
                type="checkbox"
                checked={selected!.has(c.id)}
                onChange={() => {
                  const next = new Set(selected);
                  if (next.has(c.id)) next.delete(c.id);
                  else next.add(c.id);
                  onChange!(next);
                }}
                aria-label={`Use ${c.name}`}
                className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 rounded border-slate-300 accent-blue-600"
              />
            )}
            <div className="min-w-0 flex-1" title={c.source_filename ? `From ${c.source_filename}` : undefined}>
              {renamingId === c.id ? (
                <input
                  autoFocus
                  value={draftName}
                  maxLength={NAME_MAX}
                  aria-label="Collection name"
                  onChange={(e) => setDraftName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") void commitRename(c);
                    else if (e.key === "Escape") cancelRename();
                  }}
                  onBlur={() => void commitRename(c)}
                  className={`w-full rounded border border-primary px-1 py-0 ${textSize} focus:outline-none`}
                />
              ) : (
                <span className="break-words">{c.name}</span>
              )}
              <div className="text-[10px] text-slate-400">{describeCoverage(c)}</div>
            </div>
            <button
              type="button"
              onClick={() => startRename(c)}
              title="Rename collection"
              aria-label={`Rename ${c.name}`}
              className="mt-0.5 flex-shrink-0 rounded p-0.5 text-slate-300 opacity-0 transition-opacity hover:text-primary focus:opacity-100 group-hover:opacity-100"
            >
              <Pencil className="h-3 w-3" />
            </button>
            <button
              type="button"
              onClick={() => void handleDelete(c)}
              disabled={deletingId === c.id}
              title="Delete collection"
              aria-label={`Delete ${c.name}`}
              className="mt-0.5 flex-shrink-0 rounded p-0.5 text-slate-300 opacity-0 transition-opacity hover:text-red-500 focus:opacity-100 group-hover:opacity-100 disabled:opacity-50"
            >
              {deletingId === c.id ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Trash2 className="h-3 w-3" />
              )}
            </button>
          </div>
        ))
      )}
    </div>
  );
}
