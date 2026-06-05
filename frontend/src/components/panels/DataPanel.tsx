import { useCallback, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  CircleDashed,
  FileUp,
  FolderUp,
  Info,
  Loader2,
  RotateCcw,
  Sparkles,
} from "lucide-react";

import {
  commitIngestSession,
  createIngestSession,
  discardIngestSession,
  getMergePlan,
  setIngestOptions,
  uploadIngestFiles,
  type IngestIssue,
  type IngestOptions,
  type IngestState,
  type IngestUnit,
  type MergePlan,
  type UnitFormat,
} from "@/api/ingest";
import { getDataset, listDatasets, uploadDataset } from "@/api/datasets";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { DatasetLibrary } from "@/components/panels/DatasetLibrary";

const FORMAT_LABELS: Record<UnitFormat, string> = {
  tenx_mex: "10x MEX (matrix + barcodes + features)",
  tenx_h5: "10x HDF5 (.h5)",
  anndata: "AnnData (.h5ad)",
  loom: "Loom (.loom)",
  zarr: "Zarr store",
  dense_table: "Expression table (CSV/TSV)",
  seurat: "Seurat (.rds)",
  unknown: "Unrecognised",
};

const ROLE_LABELS: Record<string, string> = {
  matrix: "matrix (expression)",
  barcodes: "barcodes (cell IDs)",
  features: "features (genes)",
  data: "data",
};

const DEFAULT_OPTIONS: IngestOptions = {
  name: null,
  join: "inner",
  sample_label: "sample",
  apply_reconciliation: true,
  genes_in_rows: true,
};

export function DataPanel() {
  const setCurrentDataset = useDatasetStore((s) => s.setCurrentDataset);
  const setAvailableDatasets = useDatasetStore((s) => s.setAvailableDatasets);
  const setPanel = useViewStore((s) => s.setPanel);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [state, setState] = useState<IngestState | null>(null);
  const [mergePlan, setMergePlan] = useState<MergePlan | null>(null);
  const [options, setOptions] = useState<IngestOptions>(DEFAULT_OPTIONS);

  const [busy, setBusy] = useState(false);
  const [busyNote, setBusyNote] = useState<string | null>(null);
  const [committing, setCommitting] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  // Seurat .rds/.RData go through the R converter (not the ingest wizard).
  const handleSeurat = useCallback(
    async (file: File) => {
      setBusy(true);
      setError(null);
      setBusyNote("Converting Seurat object… this can take a minute.");
      try {
        const result = await uploadDataset(file);
        let dataset = await getDataset(result.id);
        let attempts = 0;
        while (dataset.status === "converting" && attempts < 90) {
          await new Promise((r) => setTimeout(r, 2000));
          dataset = await getDataset(result.id);
          attempts++;
        }
        setAvailableDatasets(await listDatasets());
        if (dataset.status === "error") {
          setError(dataset.error_message || "Seurat conversion failed.");
        } else {
          setCurrentDataset(dataset);
          if (dataset.status === "ready") setPanel("assessment");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Seurat upload failed.");
      } finally {
        setBusy(false);
        setBusyNote(null);
      }
    },
    [setAvailableDatasets, setCurrentDataset, setPanel]
  );

  const addFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      // A single Seurat file → the converter path.
      if (files.length === 1) {
        const ext = files[0].name.split(".").pop()?.toLowerCase();
        if (ext === "rds" || ext === "rdata") {
          await handleSeurat(files[0]);
          return;
        }
      }
      setBusy(true);
      setError(null);
      try {
        let sid = sessionId;
        if (!sid) {
          sid = (await createIngestSession()).session_id;
          setSessionId(sid);
        }
        const next = await uploadIngestFiles(sid, files);
        setState(next);
        if (next.bundle.is_merge) {
          const plan = await getMergePlan(sid);
          setMergePlan(plan.is_merge === false ? null : plan);
          setOptions((o) => ({ ...o, join: plan.recommended_join ?? o.join }));
        } else {
          setMergePlan(null);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to add files.");
      } finally {
        setBusy(false);
      }
    },
    [sessionId, handleSeurat]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      void addFiles(Array.from(e.dataTransfer.files));
    },
    [addFiles]
  );

  const reset = useCallback(async () => {
    if (sessionId) {
      try {
        await discardIngestSession(sessionId);
      } catch {
        /* best effort */
      }
    }
    setSessionId(null);
    setState(null);
    setMergePlan(null);
    setOptions(DEFAULT_OPTIONS);
    setError(null);
  }, [sessionId]);

  const commit = useCallback(async () => {
    if (!sessionId) return;
    setCommitting(true);
    setError(null);
    try {
      await setIngestOptions(sessionId, options);
      const { dataset_id } = await commitIngestSession(sessionId);
      const ds = await getDataset(dataset_id);
      setCurrentDataset(ds);
      setAvailableDatasets(await listDatasets());
      setSessionId(null);
      setState(null);
      setMergePlan(null);
      setOptions(DEFAULT_OPTIONS);
      setPanel("assessment");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create the dataset.");
    } finally {
      setCommitting(false);
    }
  }, [sessionId, options, setCurrentDataset, setAvailableDatasets, setPanel]);

  const bundle = state?.bundle;
  const validation = state?.validation;
  const hasDense = bundle?.units.some((u) => u.format === "dense_table") ?? false;
  const loadable = bundle?.units.filter((u) => u.format !== "unknown") ?? [];
  const canCommit = !!validation?.ok && loadable.length > 0 && !committing && !busy;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Data</h2>
        <p className="mt-1 text-sm text-slate-500">
          Import single-cell data in almost any form — a 10x matrix (the three
          <code className="mx-1 rounded bg-slate-100 px-1">matrix / barcodes / features</code>
          files), a CellRanger <code className="rounded bg-slate-100 px-1">.h5</code>, an
          AnnData <code className="rounded bg-slate-100 px-1">.h5ad</code>, a Seurat{" "}
          <code className="rounded bg-slate-100 px-1">.rds</code>, an expression table, or
          several samples to merge — or reopen one of your datasets below.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          dragActive ? "border-primary bg-primary/5" : "border-slate-300 bg-white"
        }`}
      >
        <FileUp className="h-8 w-8 text-slate-400" />
        <p className="mt-2 text-sm font-medium text-slate-700">Drag files here, or</p>
        <div className="mt-2 flex gap-2">
          <button
            onClick={() => fileInput.current?.click()}
            disabled={busy}
            className="rounded-lg bg-primary px-3 py-1.5 text-sm text-white hover:bg-primary/90 disabled:opacity-50"
          >
            Choose files
          </button>
          <button
            onClick={() => folderInput.current?.click()}
            disabled={busy}
            className="flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-50"
          >
            <FolderUp className="h-4 w-4" /> Choose folder
          </button>
        </div>
        <p className="mt-2 text-xs text-slate-400">
          .h5ad · Seurat .rds · 10x (matrix/barcodes/features or .h5) · loom · CSV/TSV ·
          multiple samples to merge
        </p>
        <input
          ref={fileInput}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => {
            void addFiles(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
        <input
          ref={folderInput}
          type="file"
          multiple
          // @ts-expect-error non-standard but widely supported directory upload
          webkitdirectory=""
          className="hidden"
          onChange={(e) => {
            void addFiles(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
      </div>

      {busy && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />{" "}
          {busyNote ?? "Inspecting files…"}
        </div>
      )}

      {error && <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>}

      {/* Detected samples (units) */}
      {bundle && bundle.units.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-700">
            Detected {bundle.units.length === 1 ? "sample" : `${bundle.units.length} samples`}
          </h3>
          {bundle.units.map((u) => (
            <UnitCard key={u.label} unit={u} />
          ))}
          {bundle.issues.map((msg, i) => (
            <div key={i} className="rounded-lg bg-slate-50 p-3 text-xs text-slate-500">
              {msg}
            </div>
          ))}
        </div>
      )}

      {/* Validation issues */}
      {validation && validation.issues.length > 0 && (
        <div className="space-y-2">
          {validation.issues.map((issue, i) => (
            <IssueCard key={i} issue={issue} />
          ))}
        </div>
      )}

      {/* Merge plan */}
      {mergePlan && <MergeSection plan={mergePlan} options={options} setOptions={setOptions} />}

      {/* Options + commit */}
      {bundle && loadable.length > 0 && (
        <div className="space-y-4 rounded-xl border border-slate-200 bg-white p-4">
          <div>
            <label className="mb-1 block text-sm font-medium text-slate-700">Dataset name</label>
            <input
              type="text"
              value={options.name ?? ""}
              placeholder={loadable.length > 1 ? "merged_dataset" : loadable[0].label}
              onChange={(e) => setOptions((o) => ({ ...o, name: e.target.value || null }))}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-primary focus:outline-none"
            />
          </div>

          {hasDense && (
            <label className="flex items-center gap-2 text-sm text-slate-600">
              <input
                type="checkbox"
                checked={options.genes_in_rows}
                onChange={(e) => setOptions((o) => ({ ...o, genes_in_rows: e.target.checked }))}
              />
              Genes are in rows (transpose the table to cells × genes)
            </label>
          )}

          <div className="flex items-center gap-3 pt-1">
            <button
              onClick={commit}
              disabled={!canCommit}
              className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary/90 disabled:opacity-50"
            >
              {committing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4" />
              )}
              {committing ? "Creating dataset…" : "Create dataset"}
            </button>
            <button
              onClick={reset}
              disabled={busy || committing}
              className="flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              <RotateCcw className="h-4 w-4" /> Start over
            </button>
          </div>
          {!validation?.ok && (
            <p className="text-xs text-amber-600">
              Resolve the issues above before creating the dataset.
            </p>
          )}
        </div>
      )}

      {/* Your datasets library */}
      <DatasetLibrary />
    </div>
  );
}

// --- Sub-components --------------------------------------------------------

function UnitCard({ unit }: { unit: IngestUnit }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="flex items-center gap-2">
        {unit.complete ? (
          <Check className="h-4 w-4 text-emerald-600" />
        ) : (
          <CircleDashed className="h-4 w-4 text-amber-500" />
        )}
        <span className="text-sm font-medium text-slate-800">{unit.label}</span>
        <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-500">
          {FORMAT_LABELS[unit.format]}
        </span>
      </div>
      {unit.format === "tenx_mex" && (
        <ul className="mt-2 space-y-1 pl-6 text-xs">
          {(["matrix", "barcodes", "features"] as const).map((role) => {
            const file = unit.files.find((f) => f.role === role);
            return (
              <li key={role} className="flex items-center gap-1.5">
                {file ? (
                  <Check className="h-3 w-3 text-emerald-600" />
                ) : (
                  <CircleDashed className="h-3 w-3 text-amber-500" />
                )}
                <span className={file ? "text-slate-600" : "text-amber-600"}>
                  {file ? file.name : `${ROLE_LABELS[role]} — still needed`}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

const SEVERITY_STYLE: Record<
  IngestIssue["severity"],
  { box: string; icon: React.ElementType; iconColor: string }
> = {
  error: { box: "bg-red-50 border-red-200 text-red-800", icon: AlertTriangle, iconColor: "text-red-500" },
  warn: { box: "bg-amber-50 border-amber-200 text-amber-800", icon: AlertTriangle, iconColor: "text-amber-500" },
  info: { box: "bg-blue-50 border-blue-200 text-blue-800", icon: Info, iconColor: "text-blue-500" },
};

function IssueCard({ issue }: { issue: IngestIssue }) {
  const s = SEVERITY_STYLE[issue.severity];
  const Icon = s.icon;
  return (
    <div className={`flex gap-2 rounded-lg border p-3 text-sm ${s.box}`}>
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${s.iconColor}`} />
      <div>
        <p className="font-medium">{issue.message}</p>
        {issue.suggestion && <p className="mt-0.5 opacity-80">{issue.suggestion}</p>}
      </div>
    </div>
  );
}

function MergeSection({
  plan,
  options,
  setOptions,
}: {
  plan: MergePlan;
  options: IngestOptions;
  setOptions: React.Dispatch<React.SetStateAction<IngestOptions>>;
}) {
  const recon = plan.reconciliation;
  return (
    <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-slate-700">
        Merging {plan.samples.length} samples
      </h3>
      <p className="text-xs text-slate-500">
        {plan.intersection.toLocaleString()} genes shared across samples
        {plan.union !== plan.intersection && `, ${plan.union.toLocaleString()} in total`}. Cells
        will be labelled by sample.
      </p>

      {recon?.needed && recon.feasible && (
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800">
          <p>{recon.message}</p>
          <label className="mt-2 flex items-center gap-2">
            <input
              type="checkbox"
              checked={options.apply_reconciliation}
              onChange={(e) =>
                setOptions((o) => ({ ...o, apply_reconciliation: e.target.checked }))
              }
            />
            Match genes by {recon.target_basis} IDs (recommended)
          </label>
        </div>
      )}
      {recon?.needed && !recon.feasible && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          {recon.message}
        </div>
      )}

      <div className="flex flex-col gap-1 text-sm text-slate-600">
        <span className="font-medium text-slate-700">Which genes to keep</span>
        <label className="flex items-center gap-2">
          <input
            type="radio"
            checked={options.join === "inner"}
            onChange={() => setOptions((o) => ({ ...o, join: "inner" }))}
          />
          Shared genes only (recommended)
        </label>
        <label className="flex items-center gap-2">
          <input
            type="radio"
            checked={options.join === "outer"}
            onChange={() => setOptions((o) => ({ ...o, join: "outer" }))}
          />
          All genes (missing ones filled with zero)
        </label>
      </div>

      <div>
        <label className="mb-1 block text-sm font-medium text-slate-700">
          Sample label column
        </label>
        <input
          type="text"
          value={options.sample_label}
          onChange={(e) => setOptions((o) => ({ ...o, sample_label: e.target.value }))}
          className="w-48 rounded-lg border border-slate-300 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
        />
      </div>

      {plan.warnings.map((w, i) => (
        <div key={i} className="rounded-lg bg-amber-50 p-2 text-xs text-amber-700">
          {w}
        </div>
      ))}
    </div>
  );
}
