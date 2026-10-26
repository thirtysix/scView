import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import {
  getPerturbation,
  previewAssignment,
  runMixscape,
  saveAssignment,
  type AssignmentPreview,
  type GuideSource,
  type MixscapeProgress,
  type MixscapeTarget,
  type PerturbationSummary,
} from "@/api/perturbation";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import {
  deHandoff,
  ldaTraces,
  mappingChanges,
  MIXSCAPE_CLASS,
  MIXSCAPE_GLOBAL,
  MIXSCAPE_P_KO,
  pctText,
  posteriorTraces,
  sortTargets,
  type TargetSortKey,
} from "@/lib/perturbation";
import { errorText } from "@/lib/programs";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";
import { useDESettingsStore } from "@/stores/deSettingsStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const MAX_MAPPING_ROWS = 300;
const btn = "rounded border px-1.5 py-0.5 text-[11px]";
const chip = (on: boolean) =>
  `${btn} ${on ? "border-slate-600 bg-slate-600 text-white" : "border-slate-300 text-slate-600 hover:bg-slate-50"}`;

/**
 * A pooled CRISPR screen (backend core/crispr.py, core/mixscape.py): each cell's guide
 * and target, then Mixscape's split of each target's cells into perturbed (KO) and not
 * perturbed (NP), with a per-target table, the posterior per target and an LDA view.
 */
export function PerturbationSubtab() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const refresh = useDatasetRefresh();
  const [summary, setSummary] = useState<PerturbationSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<GuideSource | "">("");
  const [column, setColumn] = useState("");
  const [minUmi, setMinUmi] = useState(5);
  const [minRatio, setMinRatio] = useState(2);
  const [control, setControl] = useState("");
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [filter, setFilter] = useState("");
  const [preview, setPreview] = useState<AssignmentPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [splitBy, setSplitBy] = useState("");
  const [progress, setProgress] = useState<MixscapeProgress | null>(null);
  const [running, setRunning] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const [sortKey, setSortKey] = useState<TargetSortKey>("ko_fraction");
  const [desc, setDesc] = useState(true);
  const [shown, setShown] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!datasetId) return;
    try {
      const s = await getPerturbation(datasetId);
      setSummary(s);
      setError(null);
      const src = s.call?.source ?? s.suggested.source ?? "";
      setSource(src);
      setColumn(s.call?.column ?? s.suggested.column ?? s.columns[0]?.column ?? "");
      setControl(s.call?.control ?? "");
      setMinUmi(Number(s.call?.params?.min_umi ?? s.defaults.min_umi));
      setMinRatio(Number(s.call?.params?.min_ratio ?? s.defaults.min_ratio));
      // The control matches the result on show: a saved run's "Within" choice.
      setSplitBy((s.mixscape ?? s.mixscape_saved)?.split_by ?? "");
      setMapping({});
    } catch (e) {
      setError(errorText(e));
    }
  }, [datasetId]);

  useEffect(() => {
    setSummary(null);
    setPreview(null);
    load();
  }, [load]);

  // The preview follows every choice (nothing is saved until "Save").
  useEffect(() => {
    if (!datasetId || !source || (source === "column" && !column)) return;
    const t = setTimeout(() => {
      previewAssignment(datasetId, {
        source,
        column: source === "column" ? column : null,
        min_umi: minUmi,
        min_ratio: minRatio,
        control: control || null,
        mapping,
      })
        .then((p) => {
          setPreview(p);
          setPreviewError(null);
        })
        .catch((e) => {
          setPreview(null);
          setPreviewError(errorText(e));
        });
    }, 300);
    return () => clearTimeout(t);
  }, [datasetId, source, column, minUmi, minRatio, control, mapping]);

  const save = async () => {
    if (!datasetId || !preview) return;
    setSaving(true);
    setError(null);
    try {
      const s = await saveAssignment(datasetId, {
        source: source || null,
        column: source === "column" ? column : null,
        min_umi: minUmi,
        min_ratio: minRatio,
        control: control || preview.control,
        mapping: mappingChanges(preview.mapping, preview.auto_mapping),
      });
      const col = s.call?.columns.perturbation;
      await refresh(datasetId, col);
      if (col) useSettingsStore.getState().setColorBy(col);
      await load();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setSaving(false);
    }
  };

  const run = async () => {
    if (!datasetId) return;
    const ctl = new AbortController();
    abort.current = ctl;
    setRunning(true);
    setError(null);
    setProgress(null);
    try {
      await runMixscape(datasetId, { split_by: splitBy || null }, { onProgress: setProgress, signal: ctl.signal });
      await refresh(datasetId, MIXSCAPE_CLASS);
      useSettingsStore.getState().setColorBy(MIXSCAPE_CLASS);
      await load();
    } catch (e) {
      if (!ctl.signal.aborted) setError(errorText(e));
    } finally {
      setRunning(false);
      setProgress(null);
      abort.current = null;
    }
  };

  const result = summary?.mixscape ?? null;
  const saved = summary?.mixscape_saved ?? null;
  const rows = useMemo(
    () => sortTargets((result ?? saved)?.targets ?? [], sortKey, desc),
    [result, saved, sortKey, desc],
  );
  const shownRow = rows.find((r) => r.target === shown) ?? null;
  const splitColumns = useMemo(
    () =>
      (dataset?.obs_columns ?? []).filter(
        (c) => (c.dtype === "category" || c.dtype === "object") && c.n_unique >= 2 && c.n_unique <= 50,
      ),
    [dataset],
  );
  const mappingRows = useMemo(() => {
    const entries = Object.entries(preview?.mapping ?? {});
    const f = filter.trim().toLowerCase();
    return entries.filter(([g, t]) => !f || g.toLowerCase().includes(f) || t.toLowerCase().includes(f));
  }, [preview, filter]);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  const nothing = summary && !summary.has_counts && !summary.has_cellranger && !summary.columns.length && !summary.call;

  const sortHeader = (key: TargetSortKey, label: string, title: string, right = true) => (
    <th className={`px-1.5 py-1 font-medium ${right ? "text-right" : "text-left"}`} title={title}>
      <button
        onClick={() => (sortKey === key ? setDesc(!desc) : (setSortKey(key), setDesc(key !== "target")))}
        className="hover:underline"
      >
        {label}
        {sortKey === key ? (desc ? " ▾" : " ▴") : ""}
      </button>
    </th>
  );

  return (
    <div className="space-y-3 text-xs">
      <div className="flex items-center gap-1.5 font-medium text-slate-700">
        Perturbation (CRISPR screen)
        <InfoTip label="About CRISPR screens" width={320}>
          In a pooled screen each cell carries a guide RNA that targets one gene. First each cell gets its guide
          and target. Then Mixscape (Papalexi et al. 2021) asks, per target, which cells really lost the target&apos;s
          effect: each cell is compared with its 20 most similar non-targeting (control) cells, and the cells whose
          change looks like the target&apos;s typical change are called KO; the rest are NP (not perturbed, e.g. an
          in-frame edit).
        </InfoTip>
      </div>
      <RequestError message={error} />
      {!summary && !error && (
        <div className="flex items-center gap-2 text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…
        </div>
      )}
      {nothing && (
        <p role="note" className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-slate-600">
          No guides found. Import a screen with its guide counts (10x &quot;CRISPR Guide Capture&quot; features) or
          Cell Ranger&apos;s crispr_analysis/protospacer_calls_per_cell.csv beside the matrix, or a dataset with a
          column naming each cell&apos;s guide or target (e.g. perturbation, gene, guide_ID).
        </p>
      )}

      {summary && !nothing && (
        <section aria-label="Guide assignment" className="space-y-2">
          <div className="font-medium text-slate-600">1. Each cell&apos;s guide</div>
          <div className="flex flex-wrap items-end gap-2">
            <label className="flex flex-col gap-0.5">
              <span className="text-[11px] text-slate-500">Source</span>
              <select
                aria-label="Guide source"
                value={source}
                onChange={(e) => setSource(e.target.value as GuideSource)}
                className="rounded border border-slate-300 bg-white px-1.5 py-1"
              >
                {summary.has_cellranger && <option value="cellranger">Cell Ranger guide calls</option>}
                {summary.has_counts && <option value="counts">Guide counts ({summary.guides.length} guides)</option>}
                <option value="column">An obs column</option>
              </select>
            </label>
            {source === "column" && (
              <label className="flex flex-col gap-0.5">
                <span className="text-[11px] text-slate-500">Column</span>
                <select
                  aria-label="Guide column"
                  value={column}
                  onChange={(e) => setColumn(e.target.value)}
                  className="rounded border border-slate-300 bg-white px-1.5 py-1"
                >
                  {!column && <option value="">Pick a column</option>}
                  {summary.columns.map((c) => (
                    <option key={c.column} value={c.column}>
                      {c.column} ({c.n_values} values{c.control ? `, control ${c.control}` : ""})
                    </option>
                  ))}
                  {summary.obs_columns
                    .filter((c) => !summary.columns.some((x) => x.column === c))
                    .map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                </select>
              </label>
            )}
            {source === "counts" && (
              <>
                <label className="flex flex-col gap-0.5" title="A guide counts in a cell from this many UMIs">
                  <span className="text-[11px] text-slate-500">Min UMIs</span>
                  <input
                    type="number"
                    min={1}
                    value={minUmi}
                    onChange={(e) => setMinUmi(Number(e.target.value) || 1)}
                    className="w-16 rounded border border-slate-300 px-1.5 py-1"
                  />
                </label>
                <label
                  className="flex flex-col gap-0.5"
                  title="With two or more guides, keep the top one alone when it has at least this many times the UMIs of the second (0: off, every guide counts)"
                >
                  <span className="text-[11px] text-slate-500">Top / second ≥</span>
                  <input
                    type="number"
                    min={0}
                    step={0.5}
                    value={minRatio}
                    onChange={(e) => setMinRatio(Math.max(0, Number(e.target.value) || 0))}
                    className="w-16 rounded border border-slate-300 px-1.5 py-1"
                  />
                </label>
              </>
            )}
            {preview && (
              <label className="flex flex-col gap-0.5" title="The non-targeting guides' target: Mixscape compares every target with these cells">
                <span className="text-[11px] text-slate-500">Control (non-targeting)</span>
                <select
                  aria-label="Control label"
                  value={control || preview.control || ""}
                  onChange={(e) => setControl(e.target.value)}
                  className="rounded border border-slate-300 bg-white px-1.5 py-1"
                >
                  {!preview.control && <option value="">Pick the control</option>}
                  {preview.per_target.map((t) => (
                    <option key={t.target} value={t.target}>
                      {t.target}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          <RequestError message={previewError} />
          {preview && (
            <>
              <Caption n={{ value: preview.n_cells, unit: "cells" }}>
                From {preview.source_label}: {preview.counts.assigned.toLocaleString()} cells with one target (
                {preview.counts.control.toLocaleString()} {preview.control ?? "control"}),{" "}
                {preview.counts.multiple.toLocaleString()} with guides for different genes (&quot;multiple&quot;),{" "}
                {preview.counts.none.toLocaleString()} with none; {preview.targets.length} targets.
              </Caption>
              <details>
                <summary className="cursor-pointer text-[11px] text-slate-600">
                  Guide → target ({Object.keys(preview.mapping).length} guides; edit a target to fix a name)
                </summary>
                <input
                  aria-label="Filter guides"
                  placeholder="Filter"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  className="mt-1 w-40 rounded border border-slate-300 px-1.5 py-0.5"
                />
                <div className="mt-1 max-h-48 overflow-auto rounded border border-slate-200">
                  <table className="w-full">
                    <tbody>
                      {mappingRows.slice(0, MAX_MAPPING_ROWS).map(([g, t]) => (
                        <tr key={g} className="border-b border-slate-50">
                          <td className="px-1.5 py-0.5 font-mono text-[10px] text-slate-600">{g}</td>
                          <td className="px-1.5 py-0.5">
                            <input
                              aria-label={`Target of ${g}`}
                              defaultValue={t}
                              onBlur={(e) => {
                                const v = e.target.value.trim();
                                if (v && v !== t) setMapping((m) => ({ ...m, [g]: v }));
                              }}
                              className="w-full rounded border border-slate-200 px-1 py-0.5 font-mono text-[10px]"
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {mappingRows.length > MAX_MAPPING_ROWS && (
                  <p className="text-[10px] text-slate-400">
                    First {MAX_MAPPING_ROWS} of {mappingRows.length.toLocaleString()} guides; filter to find others.
                  </p>
                )}
              </details>
              <button
                onClick={save}
                disabled={saving || !preview.control}
                className="rounded bg-primary px-2 py-1 font-medium text-white disabled:opacity-50"
              >
                {saving ? "Saving…" : summary.call ? "Save again" : "Save assignment"}
              </button>
            </>
          )}
          {summary.call && (
            <div className="flex flex-wrap items-center gap-1">
              <span className="text-[11px] text-slate-500">
                Saved: {summary.call.n_targets} targets in &apos;{summary.call.columns.perturbation}&apos;. Colour by
              </span>
              {[summary.call.columns.perturbation, summary.call.columns.guide_call].map((c) => (
                <button key={c} aria-pressed={colorBy === c} onClick={() => useSettingsStore.getState().setColorBy(c)} className={chip(colorBy === c)}>
                  {c}
                </button>
              ))}
            </div>
          )}
        </section>
      )}

      {summary?.call && (
        <section aria-label="Mixscape" className="space-y-2 border-t border-slate-200 pt-2">
          <div className="flex items-center gap-1.5 font-medium text-slate-600">
            2. Mixscape: perturbed or not
            <InfoTip label="About Mixscape" width={320}>
              Per target: a Wilcoxon test of its cells against the controls on the variable genes (|log2FC| ≥ 0.25,
              adjusted p &lt; 0.05). Each cell&apos;s perturbation signature (its expression minus the mean of its 20
              nearest control cells in PCA) is projected on the target&apos;s mean change, and a two-part Gaussian
              mixture, the control part fixed, gives each cell a probability of being perturbed (P(KO)); above 0.5 is
              KO. A target with fewer than 5 such genes is all NP. Checked against pertpy 1.0.3 (same class for 99.6% of
              Papalexi 2021&apos;s 20,729 cells).
            </InfoTip>
          </div>
          <div className="flex flex-wrap items-end gap-2">
            <label className="flex flex-col gap-0.5" title="Find each cell's control neighbours only within its own replicate / batch">
              <span className="text-[11px] text-slate-500">Within (replicate / batch)</span>
              <select
                aria-label="Split by"
                value={splitBy}
                onChange={(e) => setSplitBy(e.target.value)}
                className="rounded border border-slate-300 bg-white px-1.5 py-1"
              >
                <option value="">None</option>
                {splitColumns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name} ({c.n_unique})
                  </option>
                ))}
              </select>
            </label>
            {running ? (
              <button onClick={() => abort.current?.abort()} className="rounded border border-slate-300 px-2 py-1 text-slate-700">
                Cancel
              </button>
            ) : (
              <button
                onClick={run}
                disabled={summary.running}
                className="rounded bg-primary px-2 py-1 font-medium text-white disabled:opacity-50"
              >
                {result || saved ? "Run Mixscape again" : "Run Mixscape"}
              </button>
            )}
          </div>
          {(running || summary.running) && (
            <p role="status" className="flex items-center gap-1.5 text-slate-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              {progress ? `${progress.message} (step ${progress.done + 1} of ${progress.total})` : "Starting…"}
            </p>
          )}
        </section>
      )}

      {(result || saved) && (
        <section aria-label="Mixscape result" className="space-y-2">
          {(() => {
            const r = (result ?? saved)!;
            return (
              <Caption n={[{ value: r.n_cells, unit: "cells" }, { value: r.n_targets, unit: "targets" }]}>
                {r.summary.n_ko.toLocaleString()} of {r.summary.n_target_cells.toLocaleString()} target cells KO (
                {pctText(r.summary.ko_fraction)}); {r.summary.n_targets_with_ko} targets with KO cells; against{" "}
                {r.n_control.toLocaleString()} &apos;{r.control}&apos; cells, on {r.gene_source}
                {r.split_by ? `, neighbours within ${r.split_by}` : ""}.{r.warnings.length ? ` ${r.warnings.join(" ")}` : ""}
              </Caption>
            );
          })()}
          <div className="flex flex-wrap items-center gap-1">
            <span className="text-[11px] text-slate-500">Colour by</span>
            {(
              [
                [MIXSCAPE_CLASS, "class", "Target KO / NP, or NT (obs mixscape_class)"],
                [MIXSCAPE_GLOBAL, "KO / NP / NT", "obs mixscape_class_global"],
                [MIXSCAPE_P_KO, "P(KO)", "Posterior probability of being perturbed (obs mixscape_p_ko)"],
              ] as const
            ).map(([c, label, title]) => (
              <button key={c} title={title} aria-pressed={colorBy === c} onClick={() => useSettingsStore.getState().setColorBy(c)} className={chip(colorBy === c)}>
                {label}
              </button>
            ))}
          </div>
          <div className="max-h-80 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full">
              <thead className="sticky top-0 bg-slate-50 text-slate-600">
                <tr className="border-b border-slate-200">
                  {sortHeader("target", "Target", "The targeted gene", false)}
                  {sortHeader("n_cells", "Cells", "Cells with this target alone")}
                  {sortHeader("ko_fraction", "KO", "Share of the target's cells called perturbed (KO)")}
                  {sortHeader("n_de_genes", "DE genes", "Genes passing |log2FC| ≥ 0.25 and adjusted p < 0.05 against the controls")}
                  <th className="px-1.5 py-1" />
                </tr>
              </thead>
              <tbody>
                {rows.map((t: MixscapeTarget) => (
                  <tr
                    key={t.target}
                    onClick={() => setShown(t.target === shown ? null : t.target)}
                    className={`cursor-pointer border-b border-slate-50 hover:bg-blue-50 ${shown === t.target ? "bg-blue-50" : ""}`}
                    data-testid={`mixscape-row-${t.target}`}
                  >
                    <td className="px-1.5 py-1 font-medium text-slate-800" title={t.note ?? undefined}>
                      {t.target}
                      {t.note && <div className="text-[10px] font-normal text-slate-400">{t.note}</div>}
                    </td>
                    <td className="px-1.5 py-1 text-right tabular-nums text-slate-600">{t.n_cells.toLocaleString()}</td>
                    <td className="px-1.5 py-1 text-right" title={`${t.n_ko} of ${t.n_cells} cells`}>
                      <div className="flex items-center justify-end gap-1">
                        <div className="h-1.5 w-12 rounded bg-slate-100" aria-hidden>
                          <div className="h-1.5 rounded bg-red-500" style={{ width: `${Math.round(t.ko_fraction * 100)}%` }} />
                        </div>
                        <span className="w-8 tabular-nums text-slate-600">{pctText(t.ko_fraction)}</span>
                      </div>
                    </td>
                    <td className="px-1.5 py-1 text-right tabular-nums text-slate-600">{t.n_de_genes}</td>
                    <td className="px-1.5 py-1 text-right">
                      {t.n_ko > 0 && (
                        <button
                          title={`Open DE: '${t.target} KO' cells against the controls (Between groups on ${MIXSCAPE_CLASS})`}
                          onClick={(e) => {
                            e.stopPropagation();
                            useDESettingsStore.getState().update(deHandoff(t.target));
                            useUnifiedViewStore.getState().setActiveSubtab("de");
                          }}
                          className="text-[10px] font-medium text-primary hover:underline"
                        >
                          DE
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {shownRow && (
            <div data-testid="mixscape-detail">
              {shownRow.top_de_genes.length > 0 && (
                <p className="text-[11px] text-slate-600">
                  Top DE genes of {shownRow.target}:{" "}
                  {shownRow.top_de_genes.map((g) => `${g.gene} (${g.logfc > 0 ? "+" : ""}${g.logfc.toFixed(2)})`).join(", ")}
                </p>
              )}
              {shownRow.posterior_hist && (
                <>
                  <div style={{ height: 160 }}>
                    <Plot
                      data={posteriorTraces(shownRow.posterior_hist)}
                      layout={{
                        autosize: true,
                        barmode: "stack",
                        margin: { l: 36, r: 6, t: 6, b: 30 },
                        showlegend: true,
                        legend: { orientation: "h", y: 1.18, font: { size: 9 } },
                        xaxis: { title: { text: "P(KO)", font: { size: 10 } }, range: [0, 1], tickfont: { size: 9 } },
                        yaxis: { title: { text: "cells", font: { size: 9 } }, tickfont: { size: 9 } },
                      }}
                      config={plotConfig(`mixscape_posterior_${shownRow.target}`)}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </div>
                  <Caption n={{ value: shownRow.n_cells, unit: "cells" }}>
                    Probability of being perturbed for each {shownRow.target} cell; above 0.5 is KO.
                  </Caption>
                </>
              )}
            </div>
          )}
          {result?.lda && (
            <div data-testid="mixscape-lda">
              <div style={{ height: 240 }}>
                <Plot
                  data={ldaTraces(result.lda)}
                  layout={{
                    autosize: true,
                    margin: { l: 30, r: 6, t: 6, b: 30 },
                    showlegend: true,
                    legend: { font: { size: 9 } },
                    xaxis: { title: { text: "LD1", font: { size: 10 } }, tickfont: { size: 9 }, zeroline: false },
                    yaxis: { title: { text: result.lda.dims > 1 ? "LD2" : "(spread)", font: { size: 10 } }, tickfont: { size: 9 }, zeroline: false },
                  }}
                  config={plotConfig("mixscape_lda")}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                />
              </div>
              <Caption n={{ value: result.lda.n_shown, unit: "cells shown", of: result.lda.n_cells }}>
                Linear discriminant view of the KO and control cells on their perturbation signature (the DE genes):
                targets with a shared effect sit together.
              </Caption>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
