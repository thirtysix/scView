import { prettyObsLabel } from "@/lib/formatting";
import { useEffect, useMemo, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { InfoTip } from "@/components/common/InfoTip";
import { Caption, CapNote } from "@/components/common/Caption";
import { RequestError } from "@/components/common/RequestError";
import { useDatasetStore } from "@/stores/datasetStore";
import { getActivityCells, type ActivityRef } from "@/api/activity";
import {
  getLastTfMotifs,
  getTfMotifStatus,
  prepareTfMotifs,
  runTfMotifs,
  type TfMotifProgress,
  type TfMotifResult,
  type TfMotifRow,
  type TfMotifStatus,
} from "@/api/tfMotifs";
import { activityLabel } from "@/lib/activity";
import { downloadCsv } from "@/lib/csv";
import { errorMessage } from "@/lib/errorMessage";
import { formatP, sortTfRows, TF_CSV_HEADERS, tfCsvRows, tfGene, type TfSortKey } from "@/lib/tfMotifs";

interface Props {
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
  /** Colour the embedding by a TF's activity score. */
  onScore: (scores: Float32Array, name: string, genes?: string[], activity?: ActivityRef) => void;
  /** Colour the embedding by a gene's expression. */
  onGeneClick?: (gene: string) => void;
}

const ROWS_SHOWN = 50;
const SORTS: { key: TfSortKey; label: string; title: string }[] = [
  { key: "padj", label: "padj", title: "Benjamini-Hochberg adjusted p across the motifs tested for this group; click to sort" },
  { key: "odds_ratio", label: "OR", title: "Odds ratio: how much more often the markers' promoters carry the motif than the other genes'; click to sort" },
  { key: "k", label: "k / n", title: "Marker genes whose promoter has the motif (k) of the group's markers with a promoter (n); click to sort" },
];

/**
 * Upstream TF candidates (promoter motifs): which JASPAR motifs are found in more of a
 * group's marker promoters than in the other measured genes' (hypergeometric, BH per
 * group). The gene x motif table is built once per genome (a download and a scan of a
 * few minutes in the background); each run then takes about a second.
 */
export function TfMotifCandidates({ groupByColumn, setGroupByColumn, onScore, onGeneClick }: Props) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [status, setStatus] = useState<TfMotifStatus | null>(null);
  const [result, setResult] = useState<TfMotifResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [progress, setProgress] = useState<TfMotifProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [group, setGroup] = useState("");
  const [sort, setSort] = useState<TfSortKey>("padj");
  const [colouring, setColouring] = useState<string | null>(null);

  const groupColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter(
      (c) =>
        (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
        c.n_unique >= 2 &&
        c.n_unique <= 100 &&
        !(nCells > 0 && c.n_unique / nCells >= 0.9),
    );
  }, [dataset]);
  const column = groupColumns.some((c) => c.name === groupByColumn) ? groupByColumn : "";

  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    setResult(null);
    setError(null);
    Promise.all([getTfMotifStatus(datasetId), getLastTfMotifs(datasetId)])
      .then(([s, last]) => {
        if (cancelled) return;
        setStatus(s);
        setResult(last);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  useEffect(() => {
    if (result && !result.groups.some((g) => g.group === group)) setGroup(result.groups[0]?.group ?? "");
  }, [result, group]);

  const asm = status?.dataset?.assembly ?? null;
  const asmStatus = status?.assemblies.find((a) => a.assembly === asm) ?? null;
  const prepared = !!status?.dataset?.prepared;

  const prepare = async () => {
    if (!asmStatus || !datasetId) return;
    setPreparing(true);
    setError(null);
    setProgress(null);
    try {
      await prepareTfMotifs({ species: asmStatus.species, assembly: asmStatus.assembly }, setProgress);
      setStatus(await getTfMotifStatus(datasetId));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setPreparing(false);
    }
  };

  const run = async () => {
    if (!datasetId || !column) return;
    setRunning(true);
    setError(null);
    try {
      setResult(await runTfMotifs(datasetId, { column }));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setRunning(false);
    }
  };

  /** Colour the cells by the TF's CollecTRI activity when it has one, else its expression. */
  const colourBy = async (row: TfMotifRow) => {
    if (!datasetId || colouring) return;
    const gene = tfGene(row);
    setColouring(row.motif_id);
    try {
      if (result?.activity.available && row.activity_agrees !== null) {
        for (const source of [gene, ...row.tfs.filter((t) => t !== gene)]) {
          try {
            const res = await getActivityCells(datasetId, "collectri", source.toUpperCase());
            onScore(new Float32Array(res.scores), activityLabel("collectri", res.source), undefined, {
              network: "collectri",
              source: res.source,
            });
            return;
          } catch {
            // not a CollecTRI source here: try the next member, then expression
          }
        }
      }
      onGeneClick?.(gene);
    } finally {
      setColouring(null);
    }
  };

  const current = result?.groups.find((g) => g.group === group) ?? null;
  const rows = useMemo(() => (current ? sortTfRows(current.rows, sort) : []), [current, sort]);
  const shown = rows.slice(0, ROWS_SHOWN);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  const selectCls =
    "w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200";
  const toDownload = (asmStatus?.files ?? []).filter((f) => !f.downloaded).reduce((s, f) => s + f.bytes, 0);
  const minutes = Math.max(1, Math.round((asmStatus?.estimate_seconds ?? 120) / 60));

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5">
        <h3 className="text-xs font-medium text-slate-700 dark:text-slate-200">Upstream TF candidates (promoter motifs)</h3>
        <InfoTip label="About upstream TF candidates" width={320}>
          For each group, the transcription factor motifs (JASPAR 2024) found in more of its
          marker genes&apos; promoters (1 kb upstream of the start) than in the other measured
          genes&apos; promoters. A motif counts in a promoter when one site scores at p &lt; 1e-4
          on either strand. Each row is a candidate to check, not a proven regulator.
        </InfoTip>
      </div>

      {status && !asm && (
        <p className="text-[11px] text-amber-700 dark:text-amber-400">
          The species of this dataset&apos;s genes is unknown: promoter motifs need human or mouse gene symbols.
        </p>
      )}

      {status && asmStatus && !prepared && (
        <div className="space-y-2 rounded border border-slate-200 p-2 dark:border-slate-700">
          <p className="text-[11px] text-slate-600 dark:text-slate-300">
            First, a table of motif hits in every gene&apos;s promoter is built for {asmStatus.label}: a
            one-off {toDownload > 0 ? `download of ${(toDownload / 1e6).toFixed(0)} MB and ` : ""}scan of
            about {minutes} min, kept for later runs.
            {!status.auto_download && toDownload > 0 && " Downloads are switched off on this server (TF_MOTIFS_AUTO_DOWNLOAD)."}
          </p>
          <button
            type="button"
            onClick={prepare}
            disabled={preparing}
            className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
          >
            {preparing ? (
              <span className="flex items-center gap-1.5">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Preparing…
              </span>
            ) : asmStatus.building ? (
              "Follow the running preparation"
            ) : (
              "Prepare motif table"
            )}
          </button>
          {preparing && progress && (
            <div className="space-y-1" data-testid="tf-motif-progress">
              <p className="text-[11px] text-slate-500 dark:text-slate-400">{progress.message}</p>
              {progress.total ? (
                <progress
                  className="h-1.5 w-full"
                  max={progress.total}
                  value={progress.done ?? 0}
                  aria-label="Motif scan progress"
                />
              ) : null}
            </div>
          )}
        </div>
      )}

      {prepared && (
        <div className="flex items-end gap-2">
          <label className="flex-1 text-[11px] font-medium text-slate-500 dark:text-slate-400">
            Groups
            <select value={column} onChange={(e) => setGroupByColumn(e.target.value)} className={`mt-1 ${selectCls}`}>
              <option value="">Choose a column</option>
              {groupColumns.map((c) => (
                <option key={c.name} value={c.name}>
                  {prettyObsLabel(c.name)} ({c.n_unique})
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={run}
            disabled={!column || running}
            className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
          >
            {running ? (
              <span className="flex items-center gap-1.5">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Testing…
              </span>
            ) : (
              "Find candidate TFs"
            )}
          </button>
        </div>
      )}
      <RequestError message={error} />

      {result && (
        <div className="space-y-2">
          <div className="flex items-end gap-2">
            <label className="flex-1 text-[11px] font-medium text-slate-500 dark:text-slate-400">
              Group ({prettyObsLabel(result.column)}{result.column !== column ? ", the last run" : ""})
              <select value={group} onChange={(e) => setGroup(e.target.value)} className={`mt-1 ${selectCls}`}>
                {result.groups.map((g) => (
                  <option key={g.group} value={g.group}>
                    {g.group} ({g.n_significant} with padj &lt; 0.05)
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              onClick={() => downloadCsv(`tf_motif_candidates_${result.column}`, TF_CSV_HEADERS, tfCsvRows(result))}
              className="flex items-center gap-1 rounded border border-slate-200 px-2 py-1.5 text-xs text-slate-600 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
            >
              <Download className="h-3.5 w-3.5" /> CSV
            </button>
          </div>

          {current && (
            <>
              <Caption
                n={[
                  { value: current.n, of: current.n_markers, unit: "markers with a promoter" },
                  { value: result.N, unit: "measured genes with a promoter" },
                  { value: current.n_cells, unit: "cells" },
                ]}
                testId="tf-motif-caption"
              >
                Motifs in {current.group}&apos;s marker promoters vs all measured genes&apos;, lowest p first; {current.n_significant} with
                padj &lt; 0.05.
              </Caption>
              {current.note && <p className="text-[11px] text-amber-700 dark:text-amber-400">{current.note}</p>}
              <CapNote
                shown={shown.length}
                total={current.n_reported}
                by={sort === "padj" ? "adjusted p" : sort === "odds_ratio" ? "odds ratio" : "overlap"}
                rest={current.n_reported > current.rows.length ? `the server keeps the ${current.rows.length} lowest p; the CSV has those` : "the CSV has all"}
              />
              <div className="max-h-80 overflow-auto rounded-lg border border-slate-200 dark:border-slate-700">
                <table className="w-full text-xs" data-testid="tf-motif-table">
                  <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800">
                    <tr className="border-b border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-300">
                      <th className="px-2 py-1.5 text-left font-medium" title="The motif's TF name in JASPAR (a dimer names two); click a row to colour the cells by the TF">
                        TF (motif)
                      </th>
                      {SORTS.map((s) => (
                        <th key={s.key} className="px-1 py-1.5 text-right font-medium" title={s.title} aria-sort={sort === s.key ? (s.key === "padj" ? "ascending" : "descending") : "none"}>
                          <button type="button" onClick={() => setSort(s.key)} className={sort === s.key ? "underline" : ""}>
                            {s.label}
                          </button>
                        </th>
                      ))}
                      <th className="px-1 py-1.5 text-left font-medium" title="Share of the group's cells expressing the TF (for a dimer, the more expressed member) and its log fold change in the group's marker result">
                        TF expressed
                      </th>
                      <th className="px-1 py-1.5 text-center font-medium" title="Whether the TF's CollecTRI activity is higher in this group than in the rest and in its top 10% of TFs; blank without a TF activity result or when the TF is not in CollecTRI">
                        Activity
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {shown.map((r) => (
                      <tr
                        key={r.motif_id}
                        onClick={() => colourBy(r)}
                        className="cursor-pointer border-b border-slate-50 hover:bg-blue-50 dark:border-slate-800 dark:hover:bg-slate-700"
                      >
                        <td className="px-2 py-1.5 text-slate-800 dark:text-slate-100" title={`${r.motif_id}; overlap: ${r.overlap.join(", ")}${r.n_overlap > r.overlap.length ? ", …" : ""}`}>
                          <span className="font-mono font-medium">{r.motif}</span>{" "}
                          <span className="text-[10px] text-slate-400">{r.motif_id}</span>
                          {r.gc_rich && (
                            <span className="ml-1 rounded bg-amber-100 px-1 text-[9px] text-amber-800 dark:bg-amber-900 dark:text-amber-200" title="GC-rich motif: CpG-island promoters carry it often, so it is enriched more easily">
                              GC-rich
                            </span>
                          )}
                          {colouring === r.motif_id && <Loader2 className="ml-1 inline h-3 w-3 animate-spin" />}
                        </td>
                        <td className="px-1 py-1.5 text-right tabular-nums text-slate-600 dark:text-slate-300" title={`p = ${formatP(r.p)}`}>
                          {formatP(r.padj)}
                        </td>
                        <td className="px-1 py-1.5 text-right tabular-nums text-slate-600 dark:text-slate-300">{r.odds_ratio.toFixed(1)}</td>
                        <td className="px-1 py-1.5 text-right tabular-nums text-slate-600 dark:text-slate-300" title={`K = ${r.K.toLocaleString()} of N = ${r.N.toLocaleString()} genes have it`}>
                          {r.k}/{r.n}
                        </td>
                        <td className="px-1 py-1.5 text-slate-600 dark:text-slate-300">
                          {r.tf_expressed ? (
                            <span className={r.tf_expressed.expressed ? "" : "text-slate-400"}>
                              {r.tf_expressed.gene} {Math.round(r.tf_expressed.pct * 100)}%
                              {r.tf_expressed.logfc !== null && ` (${r.tf_expressed.logfc > 0 ? "+" : ""}${r.tf_expressed.logfc.toFixed(1)})`}
                            </span>
                          ) : (
                            <span className="text-slate-400" title="The TF's gene is not measured in this dataset">not measured</span>
                          )}
                        </td>
                        <td className="px-1 py-1.5 text-center text-slate-600 dark:text-slate-300">
                          {r.activity_agrees === null ? "" : r.activity_agrees ? "yes" : "no"}
                        </td>
                      </tr>
                    ))}
                    {shown.length === 0 && (
                      <tr>
                        <td colSpan={6} className="px-2 py-4 text-center text-slate-400">
                          No motif has 3 or more of this group&apos;s markers.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-slate-400">Click a row to colour the cells by the TF (its activity when scored, else its expression).</p>
            </>
          )}

          {!result.activity.available && result.activity.hint && (
            <p className="text-[11px] text-slate-500 dark:text-slate-400" data-testid="tf-activity-hint">{result.activity.hint}</p>
          )}
          <p className="text-[11px] leading-snug text-amber-700 dark:text-amber-400" data-testid="tf-motif-caveats">
            {result.caveats}
          </p>
          <p className="text-[10px] leading-snug text-slate-400">
            {result.n_motifs_tested.toLocaleString()} of {result.n_motifs.toLocaleString()} motifs tested (
            {result.n_unreachable} too short to reach p &lt; 1e-4); markers: up to {result.n_markers_requested} per group
            (in ≥ 20% of its cells, adjusted p &lt; 0.05, by fold change); hypergeometric, BH per group; {result.assembly};{" "}
            {result.elapsed_seconds}s. {result.citation}.
          </p>
        </div>
      )}
      {!result && status && (
        <p className="text-[10px] leading-snug text-slate-400">{status.citation}.</p>
      )}
    </div>
  );
}
