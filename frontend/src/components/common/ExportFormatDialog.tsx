import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Loader2, X } from "lucide-react";

import { downloadUrl } from "@/api/export";
import {
  buildExport,
  exportFileUrl,
  getExportFormatInfo,
  type ExportFormat,
  type ExportFormatInfo,
  type ExportFormatOptions,
  type ReadinessItem,
} from "@/api/exportFormats";
import { InfoTip } from "@/components/common/InfoTip";
import { formatBytes } from "@/lib/formatting";

const TITLES: Record<ExportFormat, string> = {
  cellxgene: "CELLxGENE file (.h5ad)",
  vitessce: "Vitessce bundle (.zip)",
};

const STATUS_STYLE: Record<ReadinessItem["status"], string> = {
  ok: "text-emerald-700",
  warn: "text-amber-700",
  missing: "text-slate-500",
  invalid: "text-rose-700",
};
const STATUS_TEXT: Record<ReadinessItem["status"], string> = {
  ok: "present",
  warn: "check",
  missing: "missing",
  invalid: "fix",
};

interface Props {
  datasetId: string;
  format: ExportFormat;
  onClose: () => void;
}

/**
 * Options, size and contents of an export for another viewer, then the download:
 * the file is written on the server (progress shown) and the browser saves it.
 */
export function ExportFormatDialog({ datasetId, format, onClose }: Props) {
  const [info, setInfo] = useState<ExportFormatInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [embeddings, setEmbeddings] = useState<string[] | undefined>(undefined);
  const [columns, setColumns] = useState<string[] | undefined>(undefined);
  const [includeRaw, setIncludeRaw] = useState(false);
  const [genes, setGenes] = useState<"all" | "variable">("all");
  const [progress, setProgress] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const opts: ExportFormatOptions =
    format === "cellxgene" ? { embeddings, include_raw: includeRaw } : { embeddings, columns, genes };
  const key = JSON.stringify(opts);

  useEffect(() => {
    let live = true;
    setError(null);
    getExportFormatInfo(datasetId, format, JSON.parse(key) as ExportFormatOptions)
      .then((i) => {
        if (!live) return;
        setInfo(i);
        // Fix the defaults the server chose, so the boxes show them.
        setEmbeddings((cur) => cur ?? i.exported_embeddings.map(
          (name) => i.embeddings.find((e) => e.export_name === name)?.key ?? name));
        if (format === "vitessce") setColumns((cur) => cur ?? i.obs_columns);
      })
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [datasetId, format, key]);

  useEffect(() => () => abort.current?.abort(), []);

  const toggle = (list: string[] | undefined, value: string) =>
    (list ?? []).includes(value) ? (list ?? []).filter((v) => v !== value) : [...(list ?? []), value];

  const start = async () => {
    const ctrl = new AbortController();
    abort.current = ctrl;
    setError(null);
    setProgress("Starting…");
    try {
      const built = await buildExport(datasetId, format, opts, setProgress, ctrl.signal);
      downloadUrl(exportFileUrl(datasetId, built.file), built.filename);
      onClose();
    } catch (e) {
      if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : String(e));
    } finally {
      setProgress(null);
      abort.current = null;
    }
  };

  const cancel = () => {
    abort.current?.abort();
    setProgress(null);
  };

  const busy = progress !== null;
  const noEmbedding = (embeddings ?? []).length === 0;

  return createPortal(
    <div className="fixed inset-0 z-[1100] flex items-center justify-center bg-slate-900/40 p-4" role="presentation">
      <div
        role="dialog"
        aria-modal="true"
        aria-label={TITLES[format]}
        className="max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-xl bg-white p-5 shadow-xl"
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-800">{TITLES[format]}</h2>
            <p className="text-xs text-slate-500">
              {format === "cellxgene"
                ? "Opens in the CELLxGENE desktop explorer (cellxgene launch) and CELLxGENE Annotate."
                : "Opens in Vitessce in a browser or Jupyter; README.txt in the zip says how."}
            </p>
          </div>
          <button type="button" aria-label="Close" onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100">
            <X className="h-4 w-4" />
          </button>
        </div>

        {!info && !error && <p className="text-sm text-slate-500">Checking the dataset…</p>}

        {info && (
          <div className="space-y-4 text-sm">
            <fieldset>
              <legend className="mb-1 font-medium text-slate-700">Embeddings</legend>
              {info.embeddings.length === 0 && <p className="text-xs text-slate-500">This dataset has no embeddings yet.</p>}
              {info.embeddings.map((e) => (
                <label key={e.key} className="flex items-center gap-2 text-slate-700">
                  <input
                    type="checkbox"
                    checked={(embeddings ?? []).includes(e.key)}
                    disabled={busy}
                    onChange={() => setEmbeddings(toggle(embeddings, e.key))}
                  />
                  {e.label}
                  <span className="text-xs text-slate-400">
                    {e.n_dims > 3
                      ? `${e.n_dims} columns, first 2 exported`
                      : format === "vitessce" && e.n_dims === 3
                        ? "3D, Vitessce shows its first 2 axes"
                        : `${e.n_dims}D`}
                  </span>
                </label>
              ))}
            </fieldset>

            {format === "cellxgene" && (
              <label className="flex items-center gap-2 text-slate-700">
                <input
                  type="checkbox"
                  checked={includeRaw}
                  disabled={busy || !info.has_counts}
                  onChange={(e) => setIncludeRaw(e.target.checked)}
                />
                Include raw counts (as .raw)
                <InfoTip label="About raw counts">
                  The desktop explorer shows X (log-normalised values) only. CELLxGENE Discover requires the raw
                  counts as raw.X, so include them when the file is meant for a submission.
                  {!info.has_counts && " This dataset has no raw counts."}
                </InfoTip>
              </label>
            )}

            {format === "vitessce" && (
              <>
                <fieldset>
                  <legend className="mb-1 font-medium text-slate-700">Cell sets (label columns)</legend>
                  {info.label_columns.map((c) => (
                    <label key={c.name} className="flex items-center gap-2 text-slate-700">
                      <input
                        type="checkbox"
                        checked={(columns ?? []).includes(c.name)}
                        disabled={busy}
                        onChange={() => setColumns(toggle(columns, c.name))}
                      />
                      {c.name}
                      <span className="text-xs text-slate-400">{c.n_values} values</span>
                    </label>
                  ))}
                </fieldset>
                <fieldset>
                  <legend className="mb-1 font-medium text-slate-700">Genes</legend>
                  <label className="mr-4 inline-flex items-center gap-2">
                    <input type="radio" name="genes" checked={genes === "all"} disabled={busy} onChange={() => setGenes("all")} />
                    All genes
                  </label>
                  <label className="inline-flex items-center gap-2">
                    <input
                      type="radio"
                      name="genes"
                      checked={genes === "variable"}
                      disabled={busy || !info.has_variable_genes}
                      onChange={() => setGenes("variable")}
                    />
                    Variable genes only
                  </label>
                </fieldset>
              </>
            )}

            <p className="text-xs text-slate-600" data-testid="export-size">
              About {formatBytes(info.size_bytes)} · {info.n_cells.toLocaleString()} cells ×{" "}
              {info.n_genes.toLocaleString()} genes · expression from {info.x_from}
              {info.raw_from ? `, raw counts from ${info.raw_from}` : ""}
            </p>

            {(info.notes.length > 0 || Object.keys(info.dropped).length > 0) && (
              <details className="text-xs text-slate-600">
                <summary className="cursor-pointer">What changes and what is left out</summary>
                <ul className="ml-4 mt-1 list-disc space-y-0.5">
                  {info.notes.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                  {Object.entries(info.dropped).map(([where, keys]) => (
                    <li key={where}>
                      Left out from {where}: {keys.join(", ")}
                    </li>
                  ))}
                </ul>
              </details>
            )}

            {info.readiness && (
              <section aria-label="CELLxGENE Discover readiness">
                <h3 className="mb-1 flex items-center gap-1 font-medium text-slate-700">
                  Ready for CELLxGENE Discover? {info.readiness.n_ok} of {info.readiness.n_items} checks pass
                  <InfoTip label="About the readiness report" align="right">
                    Checked against the published Discover schema {info.readiness.schema_version}. A report only:
                    scView never fills in ontology terms; the curator chooses them. The organism term goes in
                    uns, and several terms in one cell are joined with " || ".
                  </InfoTip>
                </h3>
                <table className="w-full text-xs">
                  <tbody>
                    {info.readiness.items.map((it) => (
                      <tr key={it.key} className="border-t border-slate-100 align-top">
                        <td className="py-0.5 pr-2 text-slate-700">{it.label}</td>
                        <td className={`py-0.5 pr-2 font-medium ${STATUS_STYLE[it.status]}`}>{STATUS_TEXT[it.status]}</td>
                        <td className="py-0.5 text-slate-500">{it.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {info.readiness.hints.map((h) => (
                  <p key={h} className="mt-1 text-xs text-slate-500">{h}</p>
                ))}
                <a
                  href={info.readiness.schema_url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1 inline-block text-xs text-blue-600 hover:underline"
                >
                  Discover schema {info.readiness.schema_version}
                </a>
              </section>
            )}
          </div>
        )}

        {error && <p role="alert" className="mt-3 text-sm text-rose-700">{error}</p>}

        <div className="mt-4 flex items-center justify-end gap-2">
          {busy && (
            <span className="mr-auto flex items-center gap-1 text-xs text-slate-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> {progress}
            </span>
          )}
          {busy ? (
            <button type="button" onClick={cancel} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm">
              Cancel
            </button>
          ) : (
            <button type="button" onClick={onClose} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm">
              Close
            </button>
          )}
          <button
            type="button"
            onClick={start}
            disabled={!info || busy || noEmbedding}
            className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            Download
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
