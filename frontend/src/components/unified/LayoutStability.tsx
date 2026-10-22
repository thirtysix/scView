import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { LayoutGrid, Loader2, X } from "lucide-react";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import {
  getLayoutsInfo,
  getLayoutThumb,
  keepLayout,
  runLayouts,
  type LayoutEvent,
  type LayoutRelation,
  type LayoutsInfo,
  type LayoutStart,
  type LayoutSummary,
  type LayoutThumb,
} from "@/api/layouts";
import { Caption, CapNote } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { useEmbeddingReady } from "@/hooks/useEmbeddingReady";
import { decodeArrowBuffer } from "@/lib/arrow";
import { errorText } from "@/lib/embeddingReliability";
import { mapCategoryToColor } from "@/lib/colors";
import { prettyEmbeddingLabel, prettyObsLabel } from "@/lib/formatting";
import { formatDuration, secondsLeft } from "@/lib/hvgSensitivity";
import {
  countsSentence,
  drawThumbnail,
  plotSentence,
  relationSentence,
  relationTone,
  relationVerdict,
  contactTitle,
  shortLayoutLabel,
  visibleRelations,
} from "@/lib/layoutStability";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

type RGB = readonly [number, number, number];

const THUMB_PX = 150;
const MAX_GROUP_VALUES = 100;

/**
 * "Layouts" in the Unified View toolbar: a large floating panel that lays the
 * cells out again under other UMAP seeds and neighbour counts (the same PCA, the
 * graph rebuilt per count), shows the layouts side by side in the plot's colours,
 * and lists which pairs of groups touch in every, some or no layout, with the
 * contacts of the user's plot that are rare first.
 */
export function LayoutStabilityControl() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [open, setOpen] = useState(false);
  if (!datasetId) return null;
  return (
    <>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        title="Which structure in this plot survives another UMAP seed or neighbour count?"
        className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
          open ? "bg-blue-50 text-blue-700 ring-1 ring-blue-300" : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        <LayoutGrid className="h-3.5 w-3.5" />
        Layouts
      </button>
      {open &&
        createPortal(
          <LayoutStabilityPanel key={datasetId} datasetId={datasetId} onClose={() => setOpen(false)} />,
          document.body,
        )}
    </>
  );
}

export function LayoutStabilityPanel({ datasetId, onClose }: { datasetId: string; onClose: () => void }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const embedding = useSettingsStore((s) => s.embedding);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const palette = useSettingsStore((s) => s.categoricalPalette);
  const refresh = useDatasetRefresh();

  // Grouping columns: categories the app lists (at most 100 values).
  const groupColumns = useMemo(
    () =>
      (dataset?.id === datasetId ? dataset.obs_columns : []).filter(
        (c) =>
          (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
          c.n_unique <= MAX_GROUP_VALUES &&
          (c.values?.length ?? 0) > 0,
      ),
    [dataset, datasetId],
  );
  const [groupBy, setGroupBy] = useState<string | null>(() =>
    groupColumns.some((c) => c.name === colorBy) ? colorBy : null,
  );
  const [info, setInfo] = useState<LayoutsInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [summary, setSummary] = useState<LayoutSummary | null>(null);
  const [start, setStart] = useState<LayoutStart | null>(null);
  const [events, setEvents] = useState<LayoutEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [thumbs, setThumbs] = useState<Record<string, LayoutThumb>>({});
  const [hover, setHover] = useState<LayoutRelation | null>(null);
  const [showNever, setShowNever] = useState(false);
  const [keeping, setKeeping] = useState<number | null>(null);
  const [kept, setKept] = useState<Record<number, string>>({});
  const [keepNote, setKeepNote] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const runningRef = useRef(false);

  // Thumbnails by layout index; the user's plot as "current|<embedding>".
  const loadThumb = useCallback(
    async (index: number | "current", plot?: string | null) => {
      try {
        const t = await getLayoutThumb(datasetId, index, index === "current" ? { embedding: plot } : {});
        setThumbs((prev) => ({ ...prev, [index === "current" ? `current|${plot}` : String(index)]: t }));
      } catch (e) {
        if (index !== "current") setError(`Could not load layout ${index}: ${errorText(e)}`);
      }
    },
    [datasetId],
  );

  // The defaults, and a held run summarised for this grouping and plot.
  useEffect(() => {
    let cancelled = false;
    getLayoutsInfo(datasetId, { groupBy, embedding })
      .then((i) => {
        if (cancelled) return;
        setInfo(i);
        setInfoError(null);
        if (groupBy == null && i.plan?.group_by) setGroupBy(i.plan.group_by);
        if (i.result && !runningRef.current) setSummary(i.result);
      })
      .catch((e) => {
        if (!cancelled) setInfoError(errorText(e));
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupBy, embedding]);

  // Closing the panel stops a running check.
  useEffect(() => () => abortRef.current?.abort(), []);

  // Thumbnails of a held run (the same for any grouping): fetch the missing ones.
  const doneKey = summary ? summary.done.map((d) => d.index).join(",") + `|${summary.current ?? ""}` : "";
  useEffect(() => {
    if (!summary || runningRef.current) return;
    for (const d of summary.done) if (!thumbs[String(d.index)]) void loadThumb(d.index);
    if (summary.current && !thumbs[`current|${summary.current}`]) void loadThumb("current", summary.current);
  }, [doneKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = useCallback(async () => {
    if (running || !info?.available) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    runningRef.current = true;
    setRunning(true);
    setError(null);
    setSummary(null);
    setStart(null);
    setEvents([]);
    setThumbs({});
    setKept({});
    setKeepNote(null);
    setHover(null);
    let laidOut = 0;
    let plot: string | null = null;
    try {
      await runLayouts(
        datasetId,
        { group_by: groupBy, embedding },
        {
          onStart: (s) => {
            plot = s.current;
            setStart(s);
          },
          onLayout: (e) => {
            setEvents((prev) => [...prev, e]);
            void loadThumb(e.index);
            if (laidOut++ === 0 && plot) void loadThumb("current", plot);
          },
          onSummary: setSummary,
          signal: ctrl.signal,
        },
      );
    } catch (e) {
      if (!ctrl.signal.aborted) setError(errorText(e));
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      runningRef.current = false;
      setRunning(false);
    }
    if (ctrl.signal.aborted && laidOut > 0) {
      // Stopped: what was laid out is held; show it.
      getLayoutsInfo(datasetId, { groupBy, embedding })
        .then((i) => i.result && setSummary(i.result))
        .catch(() => undefined);
    }
  }, [datasetId, embedding, groupBy, info, running, loadThumb]);

  // --- Colours: the plot's codes for the grouping, the one palette and saved order ---
  const column = summary?.group_by ?? groupBy;
  const embeddingOk = useEmbeddingReady(datasetId, embedding);
  const codesQuery = useQuery({
    queryKey: ["embedding", datasetId, embedding, column],
    queryFn: () => fetchEmbeddingBinary(datasetId, embedding, column ?? undefined),
    enabled: embeddingOk && !!column,
    staleTime: 5 * 60 * 1000,
    gcTime: 60 * 1000,
  });
  const codes = useMemo(() => {
    if (!codesQuery.data || !column) return null;
    const v = decodeArrowBuffer(codesQuery.data)[column];
    return v ? Int32Array.from(v as ArrayLike<number>) : null;
  }, [codesQuery.data, column]);
  const colInfo = dataset?.obs_columns.find((c) => c.name === column);
  const categories = useMemo(() => colInfo?.values ?? [], [colInfo]);
  const slots = colInfo?.color_slots;
  const colorOf = useCallback(
    (code: number): RGB => mapCategoryToColor(code, palette, slots).slice(0, 3) as unknown as RGB,
    [palette, slots],
  );
  const outline = useMemo(() => {
    if (!hover) return null;
    return new Set([categories.indexOf(hover.a), categories.indexOf(hover.b)].filter((i) => i >= 0));
  }, [hover, categories]);

  const plan = summary ?? start ?? info?.plan ?? null;
  const plotThumb = plan?.current ? thumbs[`current|${plan.current}`] : undefined;
  const total = plan?.layouts.length ?? 0;
  const doneCount = summary?.n_done ?? events.length;
  const left = running
    ? secondsLeft(
        events.map((e) => e.seconds),
        total,
        start && total ? start.estimate_seconds / total : null,
      )
    : null;
  const doneByIndex = useMemo(() => {
    const m = new Map<number, { exists: boolean; same: boolean; key: string }>();
    for (const e of events) m.set(e.index, { exists: e.exists, same: false, key: e.key });
    for (const d of summary?.done ?? []) m.set(d.index, { exists: d.exists, same: d.same_as_current, key: d.key });
    return m;
  }, [events, summary]);

  const keep = async (index: number) => {
    setKeeping(index);
    setError(null);
    setKeepNote(null);
    try {
      const res = await keepLayout(datasetId, index);
      if (res.created) await refresh(datasetId);
      setKept((prev) => ({ ...prev, [index]: res.key }));
      if (res.note) setKeepNote(res.note);
    } catch (e) {
      setError(`Could not keep that layout: ${errorText(e)}`);
    } finally {
      setKeeping(null);
    }
  };
  const show = (key: string) => useSettingsStore.getState().setEmbedding(key);

  const rows = useMemo(() => (summary ? visibleRelations(summary.relations, showNever) : []), [summary, showNever]);
  const flagged = summary?.relations.filter((r) => r.flag) ?? [];
  const neverCount = summary ? summary.relations.filter((r) => r.status === "never" && r.current !== true).length : 0;
  const rec = plan?.recorded;
  const curveText = rec
    ? rec.min_dist != null
      ? `min_dist ${rec.min_dist}`
      : rec.a != null
        ? `the stored UMAP curve (a ${rec.a.toPrecision(3)}, b ${rec.b?.toPrecision(3)})`
        : ""
    : "";

  return (
    <div
      role="dialog"
      aria-label="Layout stability check"
      className="fixed right-4 top-20 z-50 flex max-h-[calc(100vh-6rem)] w-[62rem] max-w-[calc(100vw-2rem)] flex-col gap-2 overflow-y-auto rounded-xl border border-slate-200 bg-white p-4 text-xs shadow-xl"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-800">
            Which structure in this plot survives another seed or neighbour count?
          </div>
          <p className="mt-0.5 text-[11px] text-slate-500">
            A UMAP is one draw of a random layout. This lays the same cells out again
            {plan ? ` with seeds ${plan.seeds.join(", ")} and ${plan.n_neighbors.join(", ")} neighbours` : ""}
            {rec ? ` (${curveText}; for each neighbour count the graph is rebuilt from ${rec.use_rep})` : ""} and
            checks which groups touch in each layout. Islands that touch in only some layouts are not a property of the
            cells. Nothing is saved until you keep a layout.
          </p>
        </div>
        <button onClick={onClose} aria-label="Close" className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600">
          <X className="h-4 w-4" />
        </button>
      </div>

      {infoError && <p role="alert" className="rounded bg-red-50 px-2 py-1.5 text-red-700">{infoError}</p>}
      {info && !info.available && (
        <p role="alert" className="rounded bg-amber-50 px-2 py-1.5 text-amber-800">{info.reason}</p>
      )}
      {!info && !infoError && (
        <div className="flex items-center gap-1.5 text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking the dataset…
        </div>
      )}

      {info?.available && (
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-1.5 text-[11px] text-slate-500">
            Groups
            <select
              aria-label="Grouping column"
              value={groupBy ?? ""}
              disabled={running}
              onChange={(e) => setGroupBy(e.target.value || null)}
              className="max-w-[12rem] rounded border border-slate-300 px-1 py-0.5 text-xs text-slate-700"
            >
              {groupBy == null && <option value="">(choose)</option>}
              {groupColumns.map((c) => (
                <option key={c.name} value={c.name}>
                  {prettyObsLabel(c.name)} ({c.n_unique})
                </option>
              ))}
            </select>
          </label>
          {running ? (
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="rounded border border-slate-300 px-3 py-1 text-xs text-slate-600 hover:bg-slate-50"
            >
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void run()}
              disabled={!groupBy || info.running}
              className="rounded bg-blue-500 px-3 py-1 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-50"
            >
              {summary ? "Run again" : "Run"}
            </button>
          )}
          {running ? (
            <span className="flex items-center gap-1.5 text-slate-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Layout {Math.min(events.length + 1, total)} of {total}
              {left != null && left > 0 ? `, about ${formatDuration(left)} left` : ""}…
            </span>
          ) : (
            <span className="text-[11px] text-slate-500">
              {info.running
                ? "A check is already running on this dataset."
                : info.estimate_seconds != null && plan
                  ? `About ${formatDuration(info.estimate_seconds)} for ${total} layouts of ${plan.n_laid_out.toLocaleString()} cells (the first run after the server starts adds ~${formatDuration(info.first_run_seconds)}).`
                  : null}
            </span>
          )}
        </div>
      )}

      {plan && plan.notes.length > 0 && (
        <ul className="list-disc pl-4 text-[11px] text-slate-500">
          {plan.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}

      {summary && summary.relations.length > 0 && (
        <div className="rounded bg-slate-50 px-2 py-1.5 text-[11px] text-slate-700" data-testid="layouts-headline">
          <p>
            {countsSentence(summary)} {plotSentence(summary)}
          </p>
          {flagged.length > 0 && (
            <ul className="mt-1 list-disc pl-4">
              {flagged.slice(0, 5).map((r) => (
                <li key={`${r.a}|${r.b}`} className={relationTone(r)}>
                  {relationSentence(r)}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {plan && (doneCount > 0 || running) && (
        <div className="flex flex-wrap items-start gap-3" data-testid="layout-grid">
          {plan.current && (
            <Thumbnail
              testId="thumb-current"
              title={`Your plot (${prettyEmbeddingLabel(plan.current)})`}
              thumb={plotThumb}
              codes={codes}
              colorOf={colorOf}
              outline={outline}
              outlineNames={hover ? [hover.a, hover.b] : null}
              touch={hover ? hover.current : null}
              highlight
            />
          )}
          <div className="grid grid-cols-3 gap-2">
            {plan.layouts.map((l) => {
              const d = doneByIndex.get(l.index);
              const key = kept[l.index] ?? (d?.exists ? l.key : null);
              return (
                <Thumbnail
                  key={l.index}
                  testId={`thumb-${l.index}`}
                  title={shortLayoutLabel(l.seed, l.n_neighbors)}
                  thumb={thumbs[String(l.index)]}
                  codes={codes}
                  colorOf={colorOf}
                  outline={outline}
                  outlineNames={hover ? [hover.a, hover.b] : null}
                  touch={hover && d ? (hover.layouts[l.index] ?? null) : null}
                  pending={!d}
                  same={d?.same ?? false}
                  footer={
                    d ? (
                      key ? (
                        <button
                          onClick={() => show(key)}
                          title={`Saved as ${key}: show it in the main plot`}
                          className="rounded px-1 text-[10px] text-blue-600 hover:bg-blue-50"
                        >
                          Show
                        </button>
                      ) : (
                        <button
                          onClick={() => void keep(l.index)}
                          disabled={keeping !== null || running}
                          title={`Save this layout as the embedding ${l.key}`}
                          className="flex items-center gap-1 rounded border border-slate-300 px-1 text-[10px] text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                        >
                          {keeping === l.index && <Loader2 className="h-2.5 w-2.5 animate-spin" />}
                          Keep this layout
                        </button>
                      )
                    ) : null
                  }
                />
              );
            })}
          </div>
        </div>
      )}
      {plan && doneCount > 0 && (
        <Caption
          n={[
            { value: plan.n_laid_out, unit: "cells laid out", of: plan.sampled ? plan.n_cells : undefined },
            { value: doneCount, unit: doneCount === 1 ? "layout" : "layouts" },
          ]}
        >
          Each thumbnail draws the same{" "}
          {(plotThumb ?? thumbs["0"])?.n_shown.toLocaleString() ?? ""} cells, coloured by{" "}
          {column ? prettyObsLabel(column) : "the grouping"} as in the main plot.
          {plan.sampled ? " The layouts are of a seeded sample of the cells." : ""}
          {hover ? ` Outlined: ${hover.a} and ${hover.b}.` : " Hover a row below to outline its two groups in every layout."}
        </Caption>
      )}
      {keepNote && <p className="text-[11px] text-slate-500">{keepNote}</p>}

      {summary && summary.relations.length > 0 && (
        <div className="space-y-1">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1 text-[11px] font-medium text-slate-600">
              Pairs of {prettyObsLabel(summary.group_by ?? "")} that touch
              <InfoTip width={320}>{summary.rule}</InfoTip>
            </div>
            {neverCount > 0 && (
              <label className="flex items-center gap-1 text-[11px] text-slate-500">
                <input type="checkbox" checked={showNever} onChange={(e) => setShowNever(e.target.checked)} />
                Show the {neverCount} pairs that never touch
              </label>
            )}
          </div>
          <table className="w-full border-collapse text-[11px]" data-testid="relations-table">
            <thead>
              <tr className="border-b border-slate-200 text-left text-slate-500">
                <th className="py-1 pr-2 font-medium" title="The two groups of the pair">Groups</th>
                <th className="py-1 pr-2 font-medium" title="Whether the two groups touch in your plot (same rule as the layouts)">
                  Your plot
                </th>
                <th
                  className="py-1 pr-2 font-medium"
                  title="Layouts in which the two groups touch: one mark per layout in the grid's order (rows: neighbour counts, columns: seeds)"
                >
                  Layouts
                </th>
                <th
                  className="py-1 pr-2 text-right font-medium"
                  title="Contact cells in your plot: cells of either group in its main body with a main-body cell of the other among their nearest cells on the plot"
                >
                  Contact
                </th>
                <th className="py-1 font-medium" title="Every / some / no layout; flagged when your plot differs from most layouts">
                  Verdict
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={`${r.a}|${r.b}`}
                  tabIndex={0}
                  data-testid="relation-row"
                  onMouseEnter={() => setHover(r)}
                  onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover(r)}
                  onBlur={() => setHover(null)}
                  title={relationSentence(r)}
                  className={`cursor-default border-b border-slate-100 hover:bg-slate-50 focus:bg-slate-50 focus:outline-none ${
                    r.flag ? "bg-amber-50/40" : ""
                  }`}
                >
                  <td className="max-w-[16rem] truncate py-0.5 pr-2 font-medium text-slate-700">
                    {r.a} — {r.b}
                  </td>
                  <td className="py-0.5 pr-2 text-slate-600">{r.current == null ? "—" : r.current ? "touch" : "apart"}</td>
                  <td className="py-0.5 pr-2">
                    <span className="inline-flex items-center gap-0.5 align-middle" aria-label={`${r.n_touch} of ${r.n_layouts} layouts`}>
                      {r.layouts.map((t, i) => (
                        <span
                          key={i}
                          className={`inline-block h-2 w-2 rounded-sm ${t ? "bg-slate-600" : "bg-slate-200"}`}
                        />
                      ))}
                    </span>
                    <span className="ml-1.5 tabular-nums text-slate-600">
                      {r.n_touch}/{r.n_layouts}
                    </span>
                  </td>
                  <td
                    className="py-0.5 pr-2 text-right tabular-nums text-slate-500"
                    title={contactTitle(r) ?? undefined}
                  >
                    {r.current_contact ? (r.current_contact[0] + r.current_contact[1]).toLocaleString() : "—"}
                  </td>
                  <td className={`py-0.5 ${relationTone(r)}`}>{relationVerdict(r)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <CapNote shown={summary.n_listed} total={summary.n_pairs} by="flag, then share of layouts" />
          {rows.length === 0 && <p className="text-[11px] text-slate-500">No two groups touch in any layout.</p>}
        </div>
      )}

      {summary && summary.groups.length > 0 && (
        <details className="text-[11px]">
          <summary className="cursor-pointer text-slate-600">
            Placement by group across layouts{" "}
            <InfoTip width={300}>
              Placement (as in Plot reliability): the share of a cell&rsquo;s graph neighbours (the dataset&rsquo;s own
              neighbour count, the graph rebuilt on the laid-out cells) that lie among its nearest 5% of cells on the
              plot; the median over the group&rsquo;s cells. A wide range means the group sits near its real neighbours
              in some layouts and away from them in others.
            </InfoTip>
          </summary>
          <table className="mt-1 w-full border-collapse" data-testid="groups-table">
            <thead>
              <tr className="border-b border-slate-200 text-left text-slate-500">
                <th className="py-1 pr-2 font-medium">Group</th>
                <th className="py-1 pr-2 text-right font-medium" title="Laid-out cells of the group">Cells</th>
                <th className="py-1 pr-2 text-right font-medium" title="Median placement in your plot">Your plot</th>
                <th className="py-1 pr-2 text-right font-medium" title="Lowest and highest median placement over the layouts">
                  Range across layouts
                </th>
              </tr>
            </thead>
            <tbody>
              {summary.groups.map((g) => (
                <tr key={g.group} className="border-b border-slate-100">
                  <td className="py-0.5 pr-2 text-slate-700">{g.group}</td>
                  <td className="py-0.5 pr-2 text-right tabular-nums text-slate-500">{g.n_cells.toLocaleString()}</td>
                  <td className="py-0.5 pr-2 text-right tabular-nums text-slate-600">
                    {g.current == null ? "—" : g.current.toFixed(2)}
                  </td>
                  <td className={`py-0.5 pr-2 text-right tabular-nums ${(g.range ?? 0) >= 0.2 ? "font-semibold text-amber-700" : "text-slate-600"}`}>
                    {g.min == null ? "—" : `${g.min.toFixed(2)} – ${g.max!.toFixed(2)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}

      {error && <p role="alert" className="rounded bg-red-50 px-2 py-1.5 text-red-700">{error}</p>}

      {summary && (
        <p className="text-[10px] leading-snug text-slate-400">
          {summary.rule} &ldquo;Keep this layout&rdquo; saves it as an embedding (X_umap_s&lt;seed&gt;_k&lt;neighbours&gt;) you
          can pick in the toolbar; the other layouts are held in memory only until the dataset changes or the server
          restarts.
        </p>
      )}
    </div>
  );
}

function Thumbnail({
  testId,
  title,
  thumb,
  codes,
  colorOf,
  outline,
  outlineNames,
  touch,
  pending = false,
  same = false,
  highlight = false,
  footer,
}: {
  testId: string;
  title: string;
  thumb: LayoutThumb | undefined;
  codes: Int32Array | null;
  colorOf: (code: number) => RGB;
  outline: ReadonlySet<number> | null;
  outlineNames: [string, string] | null;
  /** Whether the hovered pair touches here (null: nothing hovered) */
  touch: boolean | null;
  pending?: boolean;
  same?: boolean;
  /** The user's plot: framed */
  highlight?: boolean;
  footer?: ReactNode;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !thumb) return;
    const ratio = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    canvas.width = THUMB_PX * ratio;
    canvas.height = THUMB_PX * ratio;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawThumbnail(ctx, { thumb, codes, colorOf, width: THUMB_PX, height: THUMB_PX, outline });
  }, [thumb, codes, colorOf, outline]);

  return (
    <figure
      data-testid={testId}
      data-outline={outlineNames ? outlineNames.join("|") : ""}
      className={`flex flex-col gap-0.5 rounded border p-1 ${
        highlight || same ? "border-blue-400 ring-1 ring-blue-200" : "border-slate-200"
      }`}
    >
      <figcaption className="flex items-center justify-between gap-1 text-[10px] text-slate-600">
        <span className="font-medium">{title}</span>
        {same && <span className="text-blue-600" title="These settings reproduce your plot exactly">= your plot</span>}
        {touch != null && (
          <span className={touch ? "font-medium text-slate-800" : "text-slate-400"}>{touch ? "touch" : "apart"}</span>
        )}
      </figcaption>
      <div className="relative" style={{ width: THUMB_PX, height: THUMB_PX }}>
        <canvas ref={ref} style={{ width: THUMB_PX, height: THUMB_PX }} aria-label={`${title} layout`} />
        {!thumb && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-50 text-[10px] text-slate-400">
            {pending ? "waiting…" : <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          </div>
        )}
      </div>
      <div className="flex h-4 items-center justify-end">{footer}</div>
    </figure>
  );
}
