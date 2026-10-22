import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Loader2, Play, Tag } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Panel } from "@/components/common/Panel";
import { InfoTip } from "@/components/common/InfoTip";
import { Caption } from "@/components/common/Caption";
import {
  getHashing,
  getHashingPlotData,
  runHashingDemux,
  saveHashingSamples,
  type HashingMethod,
  type HashingPlotData,
  type HashingSummary,
} from "@/api/hashing";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { rgbCss } from "@/lib/categoryColors";
import { plotlyColorscale } from "@/lib/colors";
import {
  classSentence,
  doubletSourceText,
  HASH_COLUMNS,
  heatmapTrace,
  initialMapping,
  initialSelection,
  lfcFigure,
  MAX_TAG_PLOTS,
  negativeMeaning,
  orderedSelection,
  scrubletText,
  tagHistFigure,
  validColumnName,
} from "@/lib/hashing";
import { errorText } from "@/lib/qcStates";
import { plotConfig } from "@/lib/plotConfig";
import { useObsRevisionStore } from "@/stores/obsRevisionStore";
import { useSettingsStore } from "@/stores/settingsStore";

const FILL_STYLE = { width: "100%", height: "100%" } as const;
const TH = "whitespace-nowrap px-1.5 py-1 text-left font-medium";
const NUM = "whitespace-nowrap px-1.5 py-1 text-right tabular-nums";

const SOURCE_LABEL: Record<string, string> = {
  "Multiplexing Capture": "CellPlex",
  "Antibody Capture": "hashtag antibody",
};

function Th({ children, tip, right = false }: { children: ReactNode; tip: string; right?: boolean }) {
  return (
    <th className={right ? `${TH} text-right` : TH} title={tip}>
      {children}
    </th>
  );
}

/**
 * Cell hashing (C118): which sample each cell came from, from its hashtag (HTO) or
 * CellPlex (CMO) counts. Tags are picked (hashtag antibodies are suggested by name),
 * called with HTODemux or hashedDrops (core/hashing.py), shown per tag with the
 * cutoff and as a tag x call heatmap, and the singlets named per sample into a
 * sample column. Hidden when the dataset has neither tag counts nor antibodies.
 */
export function Hashing({
  datasetId,
  refreshKey = 0,
  onSaved,
}: {
  datasetId: string;
  refreshKey?: number;
  /** After a call is saved: the doublet views read hash_class. */
  onSaved?: () => void;
}) {
  const [summary, setSummary] = useState<HashingSummary | null>(null);
  const [open, setOpen] = useState(false);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [method, setMethod] = useState<HashingMethod>("htodemux");
  const [quantile, setQuantile] = useState("0.99");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plot, setPlot] = useState<HashingPlotData | null>(null);
  const [plotError, setPlotError] = useState<string | null>(null);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [column, setColumn] = useState("hash_sample");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const refresh = useDatasetRefresh();
  const colorOf = useCategoryColors("hash_call");
  const colormap = useSettingsStore((s) => s.continuousColormap);

  const adopt = (s: HashingSummary) => {
    setSummary(s);
    setPicked(initialSelection(s));
    if (s.call) {
      setMethod(s.call.method);
      setMapping(initialMapping(s.call, s.samples?.mapping));
    }
    setColumn(s.samples?.column ?? s.default_sample_column);
  };

  useEffect(() => {
    let cancelled = false;
    setSummary(null);
    setPlot(null);
    setError(null);
    setSaved(null);
    getHashing(datasetId)
      .then((s) => {
        if (!cancelled) adopt(s);
      })
      .catch(() => {
        if (!cancelled) setSummary(null); // no card rather than a broken one
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, refreshKey]);

  const callWhen = summary?.call?.when ?? null;
  useEffect(() => {
    if (!open || !callWhen) return;
    let cancelled = false;
    setPlotError(null);
    getHashingPlotData(datasetId)
      .then((p) => {
        if (!cancelled) setPlot(p);
      })
      .catch((e) => {
        if (!cancelled) setPlotError(errorText(e));
      });
    return () => {
      cancelled = true;
    };
  }, [open, datasetId, callWhen, refreshKey]);

  const hist = useMemo(
    () => (plot?.per_tag ? tagHistFigure(plot.per_tag, (t) => rgbCss(colorOf(t))) : null),
    [plot, colorOf],
  );
  const heat = useMemo(
    () => (plot?.heatmap ? heatmapTrace(plot.heatmap, plotlyColorscale(colormap)) : null),
    [plot, colormap],
  );
  const lfc = useMemo(() => (plot?.lfc ? lfcFigure(plot.lfc) : null), [plot]);

  if (!summary || !summary.available) return null;
  const call = summary.call;
  const selection = orderedSelection(summary, picked);
  const q = Number(quantile);
  const qOk = method !== "htodemux" || (Number.isFinite(q) && q > 0.5 && q < 1);
  const nOk = selection.length >= summary.limits.min_tags && selection.length <= summary.limits.max_tags;
  const movedNow = selection.filter((t) => summary.proteins.includes(t));
  const nothingLooksLikeATag = summary.tags.length === 0 && summary.suggested.length === 0;

  const toggle = (name: string) =>
    setPicked((p) => {
      const next = new Set(p);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  const run = async () => {
    setRunning(true);
    setError(null);
    setSaved(null);
    try {
      const s = await runHashingDemux(datasetId, {
        tags: selection,
        method,
        ...(method === "htodemux" ? { quantile: q } : {}),
      });
      await refresh(datasetId, "hash_call"); // the columns show in colour-by straight away
      for (const c of HASH_COLUMNS.slice(1)) useObsRevisionStore.getState().bump(datasetId, c);
      if (s.samples) useObsRevisionStore.getState().bump(datasetId, s.samples.column);
      adopt(s);
      onSaved?.();
    } catch (e) {
      setError(`Could not demultiplex: ${errorText(e)}`);
    } finally {
      setRunning(false);
    }
  };

  const saveSamples = async () => {
    if (!call || !validColumnName(column)) return;
    setSaving(true);
    setError(null);
    setSaved(null);
    try {
      const s = await saveHashingSamples(datasetId, mapping, column.trim());
      await refresh(datasetId, column.trim());
      adopt(s);
      if (s.samples) {
        const per = Object.entries(s.samples.counts)
          .map(([k, v]) => `${v.toLocaleString()} ${k}`)
          .join(", ");
        setSaved(
          `Saved ${s.samples.column}: ${per}; ${s.samples.n_unassigned.toLocaleString()} doublets and negatives left empty.`,
        );
      }
    } catch (e) {
      setError(`Could not save the sample column: ${errorText(e)}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3" data-testid="hashing">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold text-slate-700">Cell hashing</h3>
          <InfoTip label="About cell hashing" width={320}>
            Hashtag oligos (TotalSeq hashtags, or 10x CellPlex CMOs) label each sample&apos;s cells before
            they share a lane. HTODemux (Seurat, Stoeckius et al. 2018) normalises each tag by CLR across
            cells, groups the cells by k-medoids into tags + 1 groups, fits a negative binomial to each
            tag&apos;s lowest group and calls a cell positive above the fit&apos;s 0.99 quantile: one positive
            tag is a singlet, two or more a doublet, none a negative. hashedDrops (DropletUtils) removes an
            ambient estimate and uses the log fold change of the top tag over the second (confident singlet)
            and of the second over the ambient level (doublet). Both are ports checked against R on the same
            counts.
          </InfoTip>
        </div>
        <span className="flex items-center gap-2">
          {call && (
            <span className="text-[11px] text-slate-500" data-testid="hashing-badge">
              {call.label}: {call.counts.Singlet.toLocaleString()} singlets
            </span>
          )}
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="rounded border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
          >
            {open ? "Hide" : "Show"}
          </button>
        </span>
      </div>
      {!open && !call && (
        <p className="text-[11px] text-slate-500">
          {summary.tags.length > 0
            ? `${summary.tags.length} tag counts found (${summary.tags.slice(0, 4).join(", ")}${summary.tags.length > 4 ? ", …" : ""}): not demultiplexed yet.`
            : summary.suggested.length > 0
              ? `${summary.suggested.length} antibody features look like hashtags (${summary.suggested.slice(0, 4).join(", ")}${summary.suggested.length > 4 ? ", …" : ""}): not demultiplexed yet.`
              : "No hashtags found. If this run was hashed, mark the hashtags among the antibody features."}
        </p>
      )}
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {error}
        </p>
      )}
      {open && (
        <div className="space-y-2 text-[11px] text-slate-600">
          {nothingLooksLikeATag && (
            <p className="rounded-md bg-slate-50 px-2 py-1.5" data-testid="hashing-none">
              No hashtags found: no CellPlex tags, and no antibody name looks like a hashtag (Hashtag, HTO,
              TotalSeq A/B/C0251-0265, CMO3xx). If this run was hashed, tick its hashtags below.
            </p>
          )}
          <fieldset className="space-y-1">
            <legend className="font-medium text-slate-700" title="The features whose counts say which sample a cell came from.">
              Tags ({selection.length} selected)
            </legend>
            <div className="max-h-40 overflow-y-auto rounded border border-slate-100 p-1.5" data-testid="hashing-tags">
              {[...summary.tags, ...summary.proteins].map((name) => {
                const isTag = summary.tags.includes(name);
                const src = isTag ? summary.tag_sources[name] : null;
                return (
                  <label key={name} className="mr-3 inline-flex items-center gap-1 whitespace-nowrap">
                    <input
                      type="checkbox"
                      checked={picked.has(name)}
                      onChange={() => toggle(name)}
                      disabled={running}
                      aria-label={name}
                    />
                    <span className="text-slate-700">{name}</span>
                    <span className="text-[10px] text-slate-400">
                      {isTag
                        ? (SOURCE_LABEL[src ?? ""] ?? src ?? "tag")
                        : summary.suggested.includes(name)
                          ? "antibody, looks like a hashtag"
                          : "antibody"}
                    </span>
                  </label>
                );
              })}
            </div>
            {movedNow.length > 0 && (
              <p className="text-[10px] text-slate-400">
                {movedNow.length} antibody feature{movedNow.length > 1 ? "s" : ""} ({movedNow.slice(0, 4).join(", ")}
                {movedNow.length > 4 ? ", …" : ""}) will leave the protein modality when the call is saved; the protein
                CLR is made again without them. Untick a tag later to put it back.
              </p>
            )}
          </fieldset>
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="hashing-method">Method</label>
            <select
              id="hashing-method"
              value={method}
              onChange={(e) => setMethod(e.target.value as HashingMethod)}
              disabled={running}
              title="HTODemux: a negative binomial background per tag, positive above its 0.99 quantile. hashedDrops: log fold changes of the top two tags after removing the ambient level."
              className="rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600"
            >
              {summary.methods.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
            {method === "htodemux" && (
              <>
                <label htmlFor="hashing-quantile">Positive quantile</label>
                <input
                  id="hashing-quantile"
                  value={quantile}
                  onChange={(e) => setQuantile(e.target.value)}
                  disabled={running}
                  inputMode="decimal"
                  title="A cell is positive for a tag above this quantile of the tag's background fit (Seurat's positive.quantile, default 0.99). Higher: fewer positives, fewer doublets, more negatives."
                  className="w-14 rounded border border-slate-200 px-1.5 py-0.5 text-xs"
                />
              </>
            )}
            <button
              type="button"
              onClick={run}
              disabled={running || !nOk || !qOk}
              title={
                nOk
                  ? "Call every cell and save hash_call, hash_class, hash_max, hash_second and hash_margin."
                  : `Pick ${summary.limits.min_tags} to ${summary.limits.max_tags} tags.`
              }
              className="inline-flex items-center gap-1 rounded bg-slate-700 px-2 py-0.5 font-medium text-white hover:bg-slate-800 disabled:opacity-40"
            >
              {running ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
              {call ? "Demultiplex again" : "Demultiplex"}
            </button>
          </div>
          {call && (
            <div className="space-y-2 border-t border-slate-100 pt-2" data-testid="hashing-result">
              <p className="text-xs text-slate-700" data-testid="hashing-summary">
                {call.label} on {summary.n_cells.toLocaleString()} cells, {call.tags.length} tags:{" "}
                {classSentence(call.counts)}.
                {call.n_cells_at_call !== summary.n_cells &&
                  ` Called on ${call.n_cells_at_call.toLocaleString()} cells; the counts are of the cells kept since.`}
              </p>
              <Caption>
                {negativeMeaning(call)} {doubletSourceText(summary.doublet_source)}
              </Caption>
              {scrubletText(summary.scrublet) && (
                <Caption testId="hashing-scrublet">{scrubletText(summary.scrublet)}</Caption>
              )}
              <div className="overflow-x-auto">
                <table className="w-full text-[11px] text-slate-600" data-testid="hashing-table">
                  <thead className="border-b border-slate-200 text-slate-500">
                    <tr>
                      <Th tip="The tag (hashtag or CMO).">Tag</Th>
                      {call.method === "htodemux" ? (
                        <Th
                          right
                          tip="A cell is positive for the tag above this many counts: the 0.99 quantile of a negative binomial fitted to the tag's background group (the k-medoids group with the lowest mean)."
                        >
                          Cutoff
                        </Th>
                      ) : (
                        <Th
                          right
                          tip="The tag's ambient level: the mean count of its lower mode (2-centre k-means on log1p counts), removed from every cell before the fold changes."
                        >
                          Ambient
                        </Th>
                      )}
                      <Th right tip="Cells called singlets of this tag (on the current cells).">
                        Singlets
                      </Th>
                      <Th tip="The sample this tag labelled: singlets get this name in the sample column.">Sample name</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {call.tags.map((t) => {
                      const fit = call.per_tag.find((p) => p.tag === t);
                      return (
                        <tr key={t} className="border-b border-slate-100">
                          <td className="px-1.5 py-1 font-medium text-slate-700">
                            <span
                              className="mr-1 inline-block h-2 w-2 rounded-full align-middle"
                              style={{ background: rgbCss(colorOf(t)) }}
                            />
                            {t}
                          </td>
                          <td className={NUM}>
                            {call.method === "htodemux" ? `> ${fit?.cutoff ?? "—"}` : (fit?.ambient?.toFixed(1) ?? "—")}
                          </td>
                          <td className={NUM}>{(call.per_call[t] ?? 0).toLocaleString()}</td>
                          <td className="px-1.5 py-0.5">
                            <input
                              value={mapping[t] ?? t}
                              onChange={(e) => setMapping((m) => ({ ...m, [t]: e.target.value }))}
                              aria-label={`Sample name for ${t}`}
                              className="w-32 rounded border border-slate-200 px-1.5 py-0.5 text-xs"
                            />
                          </td>
                        </tr>
                      );
                    })}
                    <tr className="text-slate-500">
                      <td className="px-1.5 py-1">Doublet / Negative</td>
                      <td className={NUM}>—</td>
                      <td className={NUM}>
                        {(call.per_call.Doublet ?? 0).toLocaleString()} / {(call.per_call.Negative ?? 0).toLocaleString()}
                      </td>
                      <td className="px-1.5 py-1 text-[10px]">no sample (left empty)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <label htmlFor="hashing-column">Sample column</label>
                <input
                  id="hashing-column"
                  value={column}
                  onChange={(e) => setColumn(e.target.value)}
                  title="Name of the obs column to write (a letter first; letters, digits, space, _ . -)."
                  className="w-28 rounded border border-slate-200 px-1.5 py-0.5 text-xs"
                />
                <button
                  type="button"
                  onClick={saveSamples}
                  disabled={saving || !validColumnName(column)}
                  title="Write each singlet's sample name as a categorical column (doublets and negatives empty): the donor / sample column for QC by sample, pseudobulk and composition."
                  className="inline-flex items-center gap-1 rounded bg-slate-700 px-2 py-0.5 font-medium text-white hover:bg-slate-800 disabled:opacity-40"
                >
                  {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Tag className="h-3 w-3" />}
                  Use as sample column
                </button>
                {summary.samples && !saved && (
                  <span className="text-slate-500">Current: {summary.samples.column}.</span>
                )}
              </div>
              {saved && (
                <p className="text-slate-500" data-testid="hashing-saved">
                  {saved}
                </p>
              )}
              {plotError && <p className="text-amber-700">{plotError}</p>}
              {hist && plot?.per_tag && (
                <>
                  <Panel title="Tag counts (CLR) by call">
                    <div style={{ height: hist.height }}>
                      <Plot
                        data={hist.data}
                        layout={{ ...hist.layout, autosize: true }}
                        config={plotConfig("hashing_tags")}
                        style={FILL_STYLE}
                        useResizeHandler
                      />
                    </div>
                  </Panel>
                  <Caption
                    n={{ value: plot.n_cells, unit: "cells" }}
                    testId="hashing-hist-caption"
                  >
                    Each tag&apos;s counts across cells, CLR-normalised per tag, stacked by the call: the tag&apos;s
                    own singlets, all doublets, the other cells.
                    {call.method === "htodemux"
                      ? " Dashed red: the tag's count cutoff in CLR units (the CLR is monotone in a tag's counts)."
                      : " hashedDrops has no per-tag cutoff: its thresholds are on the fold changes below."}
                    {plot.per_tag.length > MAX_TAG_PLOTS &&
                      ` First ${MAX_TAG_PLOTS} of ${plot.per_tag.length} tags shown.`}
                  </Caption>
                </>
              )}
              {heat && plot?.heatmap && (
                <>
                  <Panel title="Tag x call heatmap" className="h-[280px]">
                    <Plot
                      data={[heat]}
                      layout={{
                        autosize: true,
                        margin: { l: 70, r: 10, t: 10, b: 80 },
                        xaxis: { tickfont: { size: 9 }, tickangle: -35 },
                        yaxis: { tickfont: { size: 9 }, autorange: "reversed" },
                      }}
                      config={plotConfig("hashing_heatmap")}
                      style={FILL_STYLE}
                      useResizeHandler
                    />
                  </Panel>
                  <Caption n={{ value: plot.n_cells, unit: "cells" }}>
                    Mean CLR of each tag (rows) in the cells of each call (columns, with their n). A clean call has
                    one bright cell per tag column; doublets are bright for two tags, negatives for none.
                  </Caption>
                </>
              )}
              {lfc && (
                <>
                  <Panel title="hashedDrops fold changes" className="h-[220px]">
                    <Plot
                      data={lfc.data}
                      layout={{ ...lfc.layout, autosize: true }}
                      config={plotConfig("hashing_lfc")}
                      style={FILL_STYLE}
                      useResizeHandler
                    />
                  </Panel>
                  <Caption n={{ value: plot?.n_cells ?? 0, unit: "cells" }}>
                    Per cell, after removing the ambient level: the top tag over the second (a confident singlet is
                    above the left line) and the second over the ambient level (a doublet is above the right line).
                    Recomputed on the current cells; the lines are the saved call&apos;s thresholds.
                  </Caption>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
