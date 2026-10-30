import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  CircleDashed,
  FileUp,
  FolderUp,
  Globe,
  Info,
  Loader2,
  RefreshCw,
  RotateCcw,
  Sparkles,
} from "lucide-react";

import {
  commitIngestSession,
  convertIngestSession,
  createIngestSession,
  discardIngestSession,
  filePath,
  getIngestState,
  getMergePlan,
  seqQcSideFiles,
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
import { getPublicImportStatus } from "@/api/publicImport";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { DatasetLibrary } from "@/components/panels/DatasetLibrary";
import { PublicImport } from "@/components/panels/PublicImport";
import { PreflightIngestNotes } from "@/components/common/PreflightNote";
import { SequencingQcSideFiles } from "@/components/qc/SequencingQcSideFiles";
import { ClonotypeUnitFiles } from "@/components/unified/ClonotypeUnitFiles";
import { commitConversionNote, conversionNote, isRFile } from "@/lib/conversion";
import {
  hdBinOptions,
  hdDefaultBin,
  hdRefusal,
  itemsFromDrop,
  itemsFromFiles,
  keepHdBin,
  skippedNote,
  splitJunk,
  type HdBinOption,
  type UploadItem,
} from "@/lib/ingestUpload";
import { HdBinChoice } from "@/components/panels/HdBinChoice";

const FORMAT_LABELS: Record<UnitFormat, string> = {
  tenx_mex: "10x MEX (matrix + barcodes + features)",
  tenx_h5: "10x HDF5 (.h5)",
  anndata: "AnnData (.h5ad)",
  loom: "Loom (.loom)",
  zarr: "Zarr store",
  dense_table: "Expression table (CSV/TSV)",
  seurat: "R object (.rds: Seurat, SingleCellExperiment), converted when the dataset is created",
  h5seurat: "Seurat (.h5Seurat)",
  unknown: "Unrecognised",
};

const ROLE_LABELS: Record<string, string> = {
  matrix: "matrix (expression)",
  barcodes: "barcodes (cell IDs)",
  features: "features (genes)",
  data: "data",
};

// How long the panel follows one R conversion; it carries on in the background after.
const CONVERT_WAIT_MS = 2 * 60 * 60 * 1000;

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
  // While a commit converts the session's .rds files: which one, and the converter's stage.
  const [commitNote, setCommitNote] = useState<string | null>(null);
  // "Convert now": the session's .rds files converted before commit, for the merge plan.
  const [converting, setConverting] = useState(false);
  const [convertNote, setConvertNote] = useState<string | null>(null);
  // The session being shown: a conversion stops following one the user started over from.
  const liveSession = useRef<string | null>(null);
  liveSession.current = sessionId;
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Files of a picked/dropped folder that were not sent (BAM, Loupe, report), in one sentence.
  const [notSent, setNotSent] = useState<string | null>(null);
  // A Visium HD folder with several bin sizes, waiting for the user's choice.
  const [hdPick, setHdPick] = useState<{ items: UploadItem[]; options: HdBinOption[] } | null>(null);

  // CELLxGENE Discover import: offered when the server allows it (PUBLIC_IMPORT_ENABLED).
  const [publicEnabled, setPublicEnabled] = useState(false);
  const [publicOpen, setPublicOpen] = useState(false);
  useEffect(() => {
    let live = true;
    getPublicImportStatus()
      .then((s) => live && setPublicEnabled(s.enabled))
      .catch(() => live && setPublicEnabled(false));
    return () => {
      live = false;
    };
  }, []);

  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  // R objects (.rds / .RData: Seurat, SingleCellExperiment, any
  // SummarizedExperiment) go through the R converter, not the ingest wizard.
  // The upload returns at once; the converter's progress names the class it
  // found and what it is writing, and the report of what it kept shows on
  // Data Assessment after.
  const handleRObject = useCallback(
    async (file: File) => {
      setBusy(true);
      setError(null);
      setBusyNote(`Uploading ${file.name}…`);
      try {
        const result = await uploadDataset(file);
        let dataset = await getDataset(result.id);
        const started = Date.now();
        while (dataset.status === "converting" && Date.now() - started < CONVERT_WAIT_MS) {
          setBusyNote(conversionNote(dataset.conversion_progress));
          await new Promise((r) => setTimeout(r, 2000));
          dataset = await getDataset(result.id);
        }
        setAvailableDatasets(await listDatasets());
        if (dataset.status === "error") {
          setError(dataset.error_message || "The R object could not be converted.");
        } else if (dataset.status === "converting") {
          setError("Still converting after 2 hours. It carries on: open it from the library below when it is ready.");
        } else {
          setCurrentDataset(dataset);
          if (dataset.status === "ready") setPanel("assessment");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "The R file could not be uploaded.");
      } finally {
        setBusy(false);
        setBusyNote(null);
      }
    },
    [setAvailableDatasets, setCurrentDataset, setPanel]
  );

  // The merge plan for the session's samples (an .rds among them once converted).
  const refreshMergePlan = useCallback(async (sid: string, next: IngestState) => {
    if (next.bundle.is_merge) {
      const plan = await getMergePlan(sid);
      setMergePlan(plan.is_merge === false ? null : plan);
      setOptions((o) => ({ ...o, join: plan.recommended_join ?? o.join }));
    } else {
      setMergePlan(null);
    }
  }, []);

  const addFiles = useCallback(
    async (picked: UploadItem[], hdBin?: number) => {
      if (picked.length === 0) return;
      // A single R file → the converter path.
      if (picked.length === 1) {
        const first = picked[0]!.file;
        if (isRFile(first.name)) {
          await handleRObject(first);
          return;
        }
      }
      const refusal = hdRefusal(picked);
      if (refusal) {
        setNotSent(null);
        setError(refusal);
        return;
      }
      const options = hdBinOptions(picked);
      if (options.length > 1 && hdBin == null) {
        setHdPick({ items: picked, options });
        return;
      }
      const um = hdBin ?? hdDefaultBin(options);
      const one = um != null ? keepHdBin(picked, um) : { keep: picked, skipped: [] };
      const { keep, skipped } = splitJunk(one.keep);
      setNotSent(skippedNote([...one.skipped, ...skipped]));
      if (keep.length === 0) return;
      setBusy(true);
      setError(null);
      try {
        let sid = sessionId;
        if (!sid) {
          sid = (await createIngestSession()).session_id;
          setSessionId(sid);
        }
        const next = await uploadIngestFiles(sid, keep);
        setState(next);
        await refreshMergePlan(sid, next);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to add files.");
      } finally {
        setBusy(false);
      }
    },
    [sessionId, handleRObject, refreshMergePlan]
  );

  // Convert the session's R objects now (minutes for a large one), following the
  // converter's stages; the merge plan is fetched again once they are in.
  const convertNow = useCallback(async () => {
    const sid = sessionId;
    if (!sid) return;
    setConverting(true);
    setError(null);
    try {
      let s = await convertIngestSession(sid);
      setState(s);
      while (s.conversion_running && liveSession.current === sid) {
        await new Promise((r) => setTimeout(r, 2000));
        s = await getIngestState(sid);
        setConvertNote(commitConversionNote(s.converting));
      }
      if (liveSession.current !== sid) return;
      setState(s);
      await refreshMergePlan(sid, s);
    } catch (e) {
      if (liveSession.current === sid) {
        setError(e instanceof Error ? e.message : "The R object could not be converted.");
      }
    } finally {
      setConverting(false);
      setConvertNote(null);
    }
  }, [sessionId, refreshMergePlan]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      // Folders keep their layout: read the entries now, while the drop event lasts.
      void itemsFromDrop(e.dataTransfer)
        .then(addFiles)
        .catch(() => setError("The dropped folder could not be read; choose it with Choose folder instead."));
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
    setNotSent(null);
    setHdPick(null);
  }, [sessionId]);

  const commit = useCallback(async () => {
    if (!sessionId) return;
    setCommitting(true);
    setError(null);
    // .rds files are converted first: follow the converter's stages meanwhile.
    const hasR = state?.bundle.units.some((u) => u.format === "seurat") ?? false;
    let polling = hasR;
    const follow = async () => {
      while (polling) {
        await new Promise((r) => setTimeout(r, 2000));
        if (!polling) break;
        try {
          const s = await getIngestState(sessionId);
          if (polling) setCommitNote(commitConversionNote(s.converting));
        } catch {
          /* the session is gone once the dataset is saved */
        }
      }
    };
    if (hasR) void follow();
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
      polling = false;
      setCommitNote(null);
      setCommitting(false);
    }
  }, [sessionId, state, options, setCurrentDataset, setAvailableDatasets, setPanel]);

  const bundle = state?.bundle;
  const validation = state?.validation;
  const hasDense = bundle?.units.some((u) => u.format === "dense_table") ?? false;
  const loadable = bundle?.units.filter((u) => u.format !== "unknown") ?? [];
  const canCommit = !!validation?.ok && loadable.length > 0 && !committing && !busy && !converting;
  // An .rds among other samples, not converted yet: "Convert now" puts it in the merge plan.
  const convertedLabels = new Set((state?.converted ?? []).map((c) => c.label));
  const rUnits = loadable.filter((u) => u.format === "seurat");
  const unconverted = rUnits.filter((u) => !convertedLabels.has(u.label));
  const offerConvert = !!sessionId && unconverted.length > 0 && loadable.length > rUnits.length;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Data</h2>
        <p className="mt-1 text-sm text-slate-500">
          Import single-cell data in almost any form — a 10x matrix (the three
          <code className="mx-1 rounded bg-slate-100 px-1">matrix / barcodes / features</code>
          files), a CellRanger <code className="rounded bg-slate-100 px-1">.h5</code>, an
          AnnData <code className="rounded bg-slate-100 px-1">.h5ad</code>, a Seurat or
          SingleCellExperiment <code className="rounded bg-slate-100 px-1">.rds</code>, a Seurat{" "}
          <code className="rounded bg-slate-100 px-1">.h5Seurat</code>, an expression table, or
          several samples to merge (an .rds among them too) — or reopen one of your datasets below.
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
          {publicEnabled && (
            <button
              onClick={() => setPublicOpen((o) => !o)}
              disabled={busy}
              aria-expanded={publicOpen}
              className="flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              <Globe className="h-4 w-4" /> Import public dataset
            </button>
          )}
        </div>
        <p className="mt-2 text-xs text-slate-400">
          .h5ad · Seurat / SingleCellExperiment .rds · Seurat .h5Seurat · 10x (matrix/barcodes/features or
          .h5) · Visium and Visium HD (Space Ranger outs/) · loom · CSV/TSV · multiple samples to merge
        </p>
        <input
          ref={fileInput}
          type="file"
          multiple
          aria-label="Choose files"
          className="hidden"
          onChange={(e) => {
            void addFiles(itemsFromFiles(e.target.files ?? []));
            e.target.value = "";
          }}
        />
        <input
          ref={folderInput}
          type="file"
          multiple
          aria-label="Choose a folder"
          // @ts-expect-error non-standard but widely supported directory upload
          webkitdirectory=""
          data-testid="folder-input"
          className="hidden"
          onChange={(e) => {
            // Each file keeps its path inside the folder (webkitRelativePath).
            void addFiles(itemsFromFiles(e.target.files ?? []));
            e.target.value = "";
          }}
        />
      </div>

      {publicEnabled && publicOpen && <PublicImport onClose={() => setPublicOpen(false)} />}

      {busy && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />{" "}
          {busyNote ?? "Inspecting files…"}
        </div>
      )}

      {error && <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>}

      {hdPick && (
        <HdBinChoice
          options={hdPick.options}
          onCancel={() => setHdPick(null)}
          onChoose={(um) => {
            const { items } = hdPick;
            setHdPick(null);
            void addFiles(items, um);
          }}
        />
      )}

      {notSent && (
        <div className="rounded-lg bg-slate-50 p-3 text-xs text-slate-500" data-testid="not-sent">
          {notSent}
        </div>
      )}

      {/* Detected samples (units) */}
      {bundle && bundle.units.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-700">
            Detected {bundle.units.length === 1 ? "sample" : `${bundle.units.length} samples`}
          </h3>
          {bundle.units.map((u) => (
            <UnitCard key={u.label} unit={u} converted={convertedLabels.has(u.label)} />
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

      {offerConvert && (
        <div className="space-y-1 rounded-lg border border-slate-200 bg-white p-3" data-testid="convert-now">
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={convertNow}
              disabled={converting || committing || busy || !validation?.ok}
              className="flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {converting ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
              {converting ? "Converting…" : "Convert now"}
            </button>
            {converting ? (
              <p className="text-xs text-slate-500" role="status">
                {convertNote ?? "Starting the R converter…"}
              </p>
            ) : (
              <p className="text-xs text-slate-500">
                Converts {unconverted.length === 1 ? "the R object" : `the ${unconverted.length} R objects`} here
                (minutes for a large one); the merge plan then includes{" "}
                {unconverted.length === 1 ? "it" : "them"}.
              </p>
            )}
          </div>
          {!converting && state?.conversion_error && (
            <p className="text-xs text-red-700" role="alert">
              {state.conversion_error}
            </p>
          )}
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
              aria-label="Dataset name"
              type="text"
              value={options.name ?? ""}
              placeholder={loadable.length > 1 ? "merged_dataset" : loadable[0]?.label}
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

          {/* Memory and time each sample needs to load, against what scView has free */}
          {sessionId && (
            <PreflightIngestNotes
              sessionId={sessionId}
              refreshKey={bundle.units.map((u) => `${u.label}:${u.files.length}`).join("|")}
            />
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
          {committing && commitNote && (
            <p className="text-xs text-slate-500" role="status">
              {commitNote}
            </p>
          )}
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

function UnitCard({ unit, converted = false }: { unit: IngestUnit; converted?: boolean }) {
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
          {converted ? "R object (.rds: Seurat, SingleCellExperiment), converted" : FORMAT_LABELS[unit.format]}
        </span>
      </div>
      {unit.format !== "tenx_mex" && unit.files[0]?.rel_path?.includes("/") && (
        <p className="mt-1 break-all pl-6 text-xs text-slate-600" data-testid="unit-path">
          {unit.files[0].rel_path}
        </p>
      )}
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
                <span className={file ? "break-all text-slate-600" : "text-amber-600"}>
                  {file ? filePath(file) : `${ROLE_LABELS[role]} — still needed`}
                </span>
              </li>
            );
          })}
        </ul>
      )}
      <SequencingQcSideFiles files={seqQcSideFiles(unit)} />
      <ClonotypeUnitFiles unit={unit} />
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
          aria-label="Sample label column"
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
