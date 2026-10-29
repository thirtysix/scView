import { useEffect, useMemo, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { Caption, CapNote } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { SortableTh, ariaSort } from "@/components/common/SortableTh";
import {
  estimateSeconds,
  getSpatialAutocorr,
  runSpatialAutocorr,
  spatialAutocorrCsvUrl,
  type MoranResult,
  type MoranRow,
} from "@/api/spatialAutocorr";
import { errorMessage } from "@/lib/errorMessage";
import { formatPValue } from "@/lib/formatting";
import { useDatasetStore } from "@/stores/datasetStore";

const TABLE_LIMIT = 200;
const PADJ = 0.05;
const LONG_SECONDS = 4; // backend spatial_autocorr.LONG_SECONDS

type SortKey = "gene" | "I" | "pval_adj";
const COLUMNS: { key: SortKey; label: string; title: string; align: "left" | "right" }[] = [
  { key: "gene", label: "Gene", title: "Gene; click to colour the plot by it", align: "left" },
  { key: "I", label: "Moran's I", title: "Spatial autocorrelation: 1 = neighbours alike, 0 = random, < 0 = neighbours unlike", align: "right" },
  { key: "pval_adj", label: "Adjusted p", title: "Benjamini-Hochberg adjusted p (normal approximation, one-tailed in the direction of I)", align: "right" },
];

/** Rows in the chosen order; missing values last. */
export function sortMoranRows(rows: MoranRow[], key: SortKey, dir: "asc" | "desc"): MoranRow[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    if (key === "gene") return sign * a.gene.localeCompare(b.gene);
    const x = a[key];
    const y = b[key];
    if (x == null || y == null) return x == null ? (y == null ? 0 : 1) : -1;
    return sign * (x - y);
  });
}

function seconds(s: number): string {
  return s < 60 ? `${Math.max(1, Math.round(s))} s` : `${Math.round(s / 60)} min`;
}

/**
 * Spatially variable genes (C099): Moran's I per gene on the spots' neighbour graph
 * (backend core/spatial_autocorr.py, = squidpy's defaults). Clicking a gene colours
 * the plot (the tissue view too) by it, as the Expression tab does.
 */
export function SpatialGenesSubtab({
  onGeneClick,
  activeGene,
}: {
  onGeneClick: (gene: string) => void;
  activeGene?: string | null;
}) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const spatial = dataset?.spatial ?? null;
  const hasSpatial = spatial != null;
  const [result, setResult] = useState<MoranResult | null>(null);
  const [genes, setGenes] = useState<"hvg" | "all">("hvg");
  const [library, setLibrary] = useState("");
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "I", dir: "desc" });

  // The last run, restored on opening the tab.
  useEffect(() => {
    if (!datasetId || !hasSpatial) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setResult(null);
    getSpatialAutocorr(datasetId)
      .then((r) => !cancelled && setResult(r))
      .catch((e) => !cancelled && setError(errorMessage(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, hasSpatial]);

  const rows = useMemo(
    () => (result ? sortMoranRows(result.rows, sort.key, sort.dir).slice(0, TABLE_LIMIT) : []),
    [result, sort],
  );

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  if (!spatial) {
    return <div className="text-xs text-slate-400">This dataset has no spot or cell positions.</div>;
  }

  const units = spatial.unit === "spot" ? "spots" : "cells";
  const libs = spatial.libraries;
  const nObs = library ? (libs.find((l) => l.id === library)?.n_obs ?? 0) : (dataset.n_cells ?? 0);
  const allSeconds = estimateSeconds(nObs, dataset.n_genes ?? 0);

  const run = () => {
    setRunning(true);
    setError(null);
    runSpatialAutocorr(datasetId, { genes, library_id: library || null })
      .then(setResult)
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setRunning(false));
  };

  const onSort = (key: SortKey) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: key === "gene" ? "asc" : key === "I" ? "desc" : "asc" }));

  const resultUnits = result?.unit === "cell" ? "cells" : "spots";
  const sortLabel = COLUMNS.find((c) => c.key === sort.key)?.label ?? sort.key;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
        Spatially variable genes
        <InfoTip label="About Moran's I" width={300}>
          Moran&apos;s I asks whether neighbouring {units} have similar values of a gene: 1 means neighbours are
          alike, about 0 means the pattern is random, and below 0 means neighbours differ. Each {spatial.unit}{" "}
          is compared with its 6 nearest {units} in the same section.
        </InfoTip>
      </div>

      <div className="flex flex-wrap items-end gap-2 text-xs">
        <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
          Genes
          <select
            value={genes}
            onChange={(e) => setGenes(e.target.value as "hvg" | "all")}
            className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700"
          >
            <option value="hvg">Variable genes</option>
            <option value="all">All genes</option>
          </select>
        </label>
        {libs.length > 1 && (
          <label className="flex flex-col gap-0.5 text-[11px] text-slate-500">
            Section
            <select
              value={library}
              onChange={(e) => setLibrary(e.target.value)}
              className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700"
            >
              <option value="">All sections</option>
              {libs.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.id} ({l.n_obs.toLocaleString()} {units})
                </option>
              ))}
            </select>
          </label>
        )}
        <button
          type="button"
          onClick={run}
          disabled={running}
          className="flex items-center gap-1 rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {running && <Loader2 className="h-3 w-3 animate-spin" />}
          {running ? "Computing…" : result ? "Run again" : "Find spatial genes"}
        </button>
      </div>
      {genes === "all" && allSeconds >= LONG_SECONDS && (
        <p className="text-[11px] text-slate-500" data-testid="time-note">
          All {(dataset.n_genes ?? 0).toLocaleString()} genes on {nObs.toLocaleString()} {units}: about{" "}
          {seconds(allSeconds)}.
        </p>
      )}
      {libs.length > 1 && !library && (
        <p className="text-[11px] text-slate-500">Neighbours are found within each section, never across sections.</p>
      )}

      {loading && !result && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Loading the last result…
        </div>
      )}
      <RequestError message={error} />

      {result && (
        <div className="space-y-2">
          <Caption n={[{ value: result.n_obs, unit: resultUnits }, { value: result.n_genes, unit: "genes" }]}>
            Moran&apos;s I of {result.genes === "hvg" ? "the variable genes" : result.genes === "all" ? "every gene" : "the chosen genes"}
            {result.libraries.length === 1 && libs.length > 1 ? ` in section ${result.libraries[0]}` : ""};{" "}
            {result.n_significant.toLocaleString()} spatially variable (I &gt; 0, adjusted p &lt; {PADJ}).
          </Caption>
          {result.notes.map((n) => (
            <p key={n} className="text-[11px] text-amber-700">
              {n}
            </p>
          ))}
          <div className="flex items-center justify-between text-[11px] text-slate-500">
            <CapNote
              shown={rows.length}
              total={result.rows_shown}
              by={sortLabel}
              rest={
                result.rows_shown < result.n_genes
                  ? `of the ${result.rows_shown.toLocaleString()} with the highest I; the CSV has all ${result.n_genes.toLocaleString()}`
                  : "the CSV has all"
              }
            />
            <a
              href={spatialAutocorrCsvUrl(datasetId, result.result_id)}
              download
              className="ml-auto flex items-center gap-1 text-primary hover:underline"
              title="Every tested gene with I, its expected value and variance, z, p and adjusted p"
            >
              <Download className="h-3 w-3" /> CSV
            </a>
          </div>
          <div className="max-h-96 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-slate-50">
                <tr className="border-b border-slate-200 text-slate-600">
                  {COLUMNS.map((c) => (
                    <SortableTh
                      key={c.key}
                      onSort={() => onSort(c.key)}
                      sort={ariaSort(c.key, sort.key, sort.dir)}
                      title={c.title}
                      align={c.align}
                      className="px-2 py-1"
                    >
                      {c.label}
                    </SortableTh>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r.gene}
                    className={`border-b border-slate-100 ${
                      r.gene === activeGene ? "bg-primary/10" : ""
                    }`}
                  >
                    <td className="px-2 py-0.5">
                      <button
                        type="button"
                        onClick={() => onGeneClick(r.gene)}
                        className="font-medium text-primary hover:underline"
                        aria-pressed={r.gene === activeGene}
                      >
                        {r.gene}
                      </button>
                    </td>
                    <td className="px-2 py-0.5 text-right tabular-nums text-slate-700">
                      {r.I == null ? "–" : r.I.toFixed(3)}
                    </td>
                    <td className="px-2 py-0.5 text-right tabular-nums text-slate-700">
                      {r.pval_adj == null ? "–" : formatPValue(r.pval_adj)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
