import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { DeconvolutionCheck } from "@/components/unified/DeconvolutionCheck";
import {
  deconvolutionCsvUrl,
  getDeconvolution,
  getDeconvolutionDefaults,
  referenceZipUrl,
  runDeconvolution,
  type DeconvDefaults,
  type DeconvMethod,
  type DeconvResult,
} from "@/api/deconvolution";
import { categoryColor, rgbCss } from "@/lib/categoryColors";
import { pct, stackedBarTraces, type FractionKind } from "@/lib/deconvolution";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { isGroupingColumn } from "@/lib/obsColumns";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const SELECT =
  "rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200";
const TABLE_LIMIT = 200; // bulk samples in the table; the CSV has all
const AUTO_DONOR = "__auto__";

/**
 * Bulk deconvolution (C085): cell-type fractions of uploaded bulk RNA-seq samples, with
 * this dataset's cells as the reference (backend core/deconvolution.py: MuSiC-style
 * weighted NNLS, or NNLS on marker genes). A self-check on the dataset's own donors and a
 * download of the reference for MuSiC, BayesPrism or CIBERSORTx.
 */
export function DeconvolutionSubtab() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const palette = useSettingsStore((s) => s.categoricalPalette);
  const methodId = useId();
  const [defaults, setDefaults] = useState<DeconvDefaults | null>(null);
  const [result, setResult] = useState<DeconvResult | null>(null);
  const [celltype, setCelltype] = useState("");
  const [donor, setDonor] = useState(AUTO_DONOR);
  const [method, setMethod] = useState<DeconvMethod>("auto");
  const [file, setFile] = useState<File | null>(null);
  const [kind, setKind] = useState<FractionKind>("cell");
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The defaults and the last run, restored on opening the tab.
  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setResult(null);
    setDefaults(null);
    Promise.all([getDeconvolutionDefaults(datasetId), getDeconvolution(datasetId)])
      .then(([d, r]) => {
        if (cancelled) return;
        setDefaults(d);
        setResult(r);
        setCelltype(r?.celltype_column ?? d.celltype_column ?? "");
        setDonor(r ? (r.donor_column ?? "") : AUTO_DONOR);
      })
      .catch((e) => !cancelled && setError(errorMessage(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const columns = useMemo(() => {
    if (!dataset) return [];
    const names = dataset.obs_columns.map((c) => c.name);
    return dataset.obs_columns.filter((c) => isGroupingColumn(c, names, dataset.n_cells ?? 0));
  }, [dataset]);

  const typeColumn = result?.celltype_column ?? celltype;
  const colInfo = dataset?.obs_columns.find((c) => c.name === typeColumn);
  const colorOf = useCallback(
    (t: string) =>
      rgbCss(categoryColor(datasetId ?? "", typeColumn, colInfo?.values ?? result?.types ?? [], t, palette, colInfo?.color_slots)),
    [datasetId, typeColumn, colInfo, result, palette],
  );
  const traces = useMemo(() => (result ? stackedBarTraces(result, kind, colorOf) : []), [result, kind, colorOf]);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  const donorOption = donor === AUTO_DONOR ? null : donor;
  const options = { celltype_column: celltype || null, donor_column: donorOption, method };
  const donorColumn = donor === AUTO_DONOR ? (defaults?.donor_column ?? null) : donor || null;
  const nDonors = donorColumn
    ? (dataset.obs_columns.find((c) => c.name === donorColumn)?.n_unique ?? defaults?.n_donors ?? 0)
    : 0;
  const canRun = !running && (file != null || result != null);

  const run = () => {
    setRunning(true);
    setError(null);
    runDeconvolution(datasetId, file, options)
      .then((r) => {
        setResult(r);
        setFile(null);
      })
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setRunning(false));
  };

  const samples = result?.samples.slice(0, TABLE_LIMIT) ?? [];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-slate-200">
        Bulk deconvolution
        <InfoTip label="About bulk deconvolution" width={300}>
          A bulk RNA-seq sample is a mixture of cells. With this dataset&apos;s cell types as the reference, each
          bulk sample&apos;s counts are split into the share of cells of each type that best explains them. These
          are estimates: a cell type missing from this dataset cannot be found, and the reference should come
          from the same tissue and condition as the bulk samples.
        </InfoTip>
      </div>

      <div className="flex flex-wrap items-end gap-2 text-xs">
        <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
          Cell types
          <select value={celltype} onChange={(e) => setCelltype(e.target.value)} className={SELECT}>
            {!celltype && <option value="">(choose)</option>}
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                {prettyObsLabel(c.name)} ({c.n_unique})
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
          Donors
          <select value={donor} onChange={(e) => setDonor(e.target.value)} className={SELECT}>
            <option value={AUTO_DONOR}>
              Auto{defaults?.donor_column ? ` (${prettyObsLabel(defaults.donor_column)})` : " (none found)"}
            </option>
            <option value="">None</option>
            {columns
              .filter((c) => c.name !== celltype)
              .map((c) => (
                <option key={c.name} value={c.name}>
                  {prettyObsLabel(c.name)} ({c.n_unique})
                </option>
              ))}
          </select>
        </label>
        <div className="flex flex-col gap-0.5 text-[11px] text-slate-500">
          <span className="flex items-center gap-1">
            <label htmlFor={methodId}>Method</label>
            <InfoTip label="About the methods" width={300}>
              Weighted (MuSiC-style, Wang 2019): every gene, weighted down where the donors of this dataset
              disagree, so it needs {defaults?.min_donors_weighted ?? 2}+ donors. NNLS: the marker genes of each
              type, equally weighted; works without donors. Auto picks weighted when the donors allow it.
            </InfoTip>
          </span>
          <select
            id={methodId}
            value={method}
            onChange={(e) => setMethod(e.target.value as DeconvMethod)}
            className={SELECT}
          >
            <option value="auto">Auto</option>
            <option value="weighted">Weighted (MuSiC-style)</option>
            <option value="nnls">NNLS on markers</option>
          </select>
        </div>
      </div>

      <div className="space-y-1 text-xs">
        <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
          Bulk expression file
          <input
            type="file"
            accept=".tsv,.csv,.txt,.gct,.gz,.tab"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="text-xs text-slate-700 file:mr-2 file:rounded file:border file:border-slate-300 file:bg-white file:px-2 file:py-0.5 file:text-xs dark:text-slate-200 dark:file:border-slate-600 dark:file:bg-slate-800 dark:file:text-slate-200"
          />
        </label>
        <p className="text-[11px] text-slate-500">
          Genes in rows and samples in columns (or the other way round): TSV, CSV or GCT, optionally .gz, up to{" "}
          {defaults?.max_bulk_mb ?? 100} MB. Raw counts work best; TPM gives relative values. Gene symbols or Ensembl
          IDs (versions are fine). Not log values.
        </p>
        <button
          type="button"
          onClick={run}
          disabled={!canRun}
          className="flex items-center gap-1 rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {running && <Loader2 className="h-3 w-3 animate-spin" />}
          {running ? "Deconvolving…" : file || !result ? "Deconvolve" : `Run again on ${result.filename}`}
        </button>
      </div>

      {loading && !result && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Loading the last result…
        </div>
      )}
      <RequestError message={error} />

      {result && (
        <div className="space-y-2" data-testid="deconv-result">
          <Caption n={[{ value: result.n_samples, unit: "bulk samples" }, { value: result.types.length, unit: "cell types" }]}>
            {result.filename}: {result.method_name}, reference {prettyObsLabel(result.celltype_column)} (
            {result.n_reference_cells.toLocaleString()} cells
            {result.n_reference_donors ? `, ${result.n_reference_donors} donors in ${prettyObsLabel(result.donor_column)}` : ""}
            ), {result.n_genes_used.toLocaleString()} genes; {pct(result.gene_share, 0)} of this dataset&apos;s genes
            were in the file.
          </Caption>
          {result.method_reason !== "requested" && <p className="text-[11px] text-slate-500">{result.method_reason}</p>}
          {result.warnings.map((w) => (
            <p key={w} className="text-[11px] text-amber-700 dark:text-amber-400">
              {w}
            </p>
          ))}
          {result.notes.map((n) => (
            <p key={n} className="text-[11px] text-slate-500">
              {n}
            </p>
          ))}

          <div className="flex items-center gap-1 text-[11px]" role="group" aria-label="Fractions shown">
            {(["cell", "rna"] as const).map((k) => (
              <button
                key={k}
                type="button"
                aria-pressed={kind === k}
                onClick={() => setKind(k)}
                className={`rounded px-2 py-0.5 ${
                  kind === k
                    ? "bg-primary/10 font-medium text-primary"
                    : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                }`}
              >
                {k === "cell" ? "Cell fractions" : "RNA fractions"}
              </button>
            ))}
            <InfoTip label="Cell or RNA fractions" width={260}>
              Cell fractions: the share of cells. RNA fractions: the share of the sample&apos;s RNA, which is larger
              for types with more RNA per cell (cell size from this dataset).
            </InfoTip>
            <a
              href={deconvolutionCsvUrl(datasetId, result.result_id)}
              download
              className="ml-auto flex items-center gap-1 text-primary hover:underline"
              title="Cell and RNA fractions of every sample"
            >
              <Download className="h-3 w-3" /> CSV
            </a>
          </div>
          <div style={{ height: Math.min(360, 180 + 6 * result.types.length) }}>
            <Plot
              data={traces}
              layout={{
                autosize: true,
                barmode: "stack",
                margin: { l: 40, r: 6, t: 6, b: 60 },
                showlegend: true,
                legend: { font: { size: 9 } },
                xaxis: { tickfont: { size: 9 }, automargin: true },
                yaxis: {
                  title: { text: kind === "cell" ? "cells (%)" : "RNA (%)", font: { size: 10 } },
                  range: [0, 100],
                  tickfont: { size: 9 },
                },
              }}
              config={plotConfig(`deconvolution_${kind}_fractions`)}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
            />
          </div>
          <div className="max-h-80 overflow-auto rounded-lg border border-slate-200 dark:border-slate-700">
            <table className="w-full text-[11px]" aria-label="Estimated fractions">
              <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800">
                <tr className="border-b border-slate-200 text-left text-slate-600 dark:border-slate-700 dark:text-slate-300">
                  <th className="px-2 py-1 font-medium">Sample</th>
                  {result.types.map((t) => (
                    <th key={t} className="px-1 py-1 text-right font-medium" title={`${kind === "cell" ? "Cell" : "RNA"} fraction of ${t}`}>
                      {t}
                    </th>
                  ))}
                  <th className="px-1 py-1 text-right font-medium" title="Share of the sample's variation the fit explains (R²)">
                    Fit R²
                  </th>
                </tr>
              </thead>
              <tbody>
                {samples.map((s) => {
                  const fr = kind === "cell" ? s.cell_fractions : s.rna_fractions;
                  return (
                    <tr key={s.sample} className="border-b border-slate-100 dark:border-slate-800">
                      <td className="px-2 py-0.5 font-medium text-slate-700 dark:text-slate-200">{s.sample}</td>
                      {result.types.map((t) => (
                        <td key={t} className="px-1 py-0.5 text-right tabular-nums text-slate-700 dark:text-slate-200">
                          {pct(fr[t])}
                        </td>
                      ))}
                      <td
                        className="px-1 py-0.5 text-right tabular-nums text-slate-500"
                        title={s.converged ? `${s.iterations} rounds` : "did not settle; the last estimate is shown"}
                      >
                        {s.r2 == null ? "–" : s.r2.toFixed(2)}
                        {s.converged ? "" : " *"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {result.n_samples > samples.length && (
            <p className="text-[11px] text-slate-500">
              The first {samples.length} of {result.n_samples.toLocaleString()} samples; the CSV has all.
            </p>
          )}
        </div>
      )}

      <p className="text-[11px] text-slate-500" data-testid="deconv-caveats">
        Cell types absent from this dataset cannot be found, and their cells are shared out among the others. A
        reference from another tissue, condition or protocol shifts the estimates. TPM input gives relative values.
      </p>

      <DeconvolutionCheck
        key={`${datasetId}|${celltype}|${donor}|${method}`}
        datasetId={datasetId}
        options={options}
        nDonors={nDonors}
        minDonors={defaults?.min_donors_check ?? 3}
        estimateSeconds={nDonors && defaults ? defaults.check_seconds_per_donor * nDonors : null}
        colorOf={colorOf}
      />

      <div className="space-y-1">
        <a
          href={referenceZipUrl(datasetId, celltype || null, donorOption)}
          download
          className="inline-flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-800"
        >
          <Download className="h-3 w-3" /> Download reference
        </a>
        <p className="text-[11px] text-slate-500">
          A zip of this dataset as a reference for other tools: raw counts in the 10x layout with cell types and
          donors (MuSiC, BayesPrism), a CIBERSORTx single-cell reference (up to 5,000 cells), a signature, pseudobulk
          and cell sizes.
        </p>
      </div>

      <p className="text-[10px] text-slate-400">
        Weighted method from MuSiC (Wang et al. 2019, Nat Commun 10:380), written from the paper, not its code.
      </p>
    </div>
  );
}
