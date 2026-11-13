import { useEffect, useMemo, useState } from "react";
import { getCoOccurrence, runCoOccurrence, type CoOccurrenceInfo, type CoOccurrenceResult } from "@/api/spatialDomains";
import { Caption } from "@/components/common/Caption";
import { Help } from "@/components/common/Help";
import { HoverTip } from "@/components/common/InfoTip";
import { PreflightNote } from "@/components/common/PreflightNote";
import { RequestError } from "@/components/common/RequestError";
import Plot from "@/components/plots/Plot";
import { SpatialRunStatus } from "@/components/unified/SpatialRunStatus";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { rgbCss } from "@/lib/categoryColors";
import { curvesOf, csvHeaders, csvRows, distanceSource, firstUsable, matrixAt, reachOf, scoreScale, unitText } from "@/lib/cooccurrence";
import { downloadCsv } from "@/lib/csv";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSpatialDomainsRunStore } from "@/stores/spatialDomainsRunStore";

/** Cell values are written in the heatmap up to this many labels. */
const MAX_LABELLED = 15;
const INPUT = "rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700";
const FIELD = "flex flex-col gap-0.5 text-[11px] text-slate-500";
const OUTLINE =
  "rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary disabled:opacity-50";
const SOLID = "rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50";
const AXIS = { size: 11, color: "#64748b" };
const TICK = { size: 10, color: "#64748b" };

/**
 * Spatial co-occurrence (backend core/spatial_cooccurrence.py, = squidpy's co_occurrence):
 * score against distance for one label against the others, and labels x labels at one
 * distance. Reads positions and labels only.
 */
export function CoOccurrenceCard({
  columns,
  labelDefault,
}: {
  columns: { name: string; n_unique: number }[];
  labelDefault: string;
}) {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const version = useSpatialDomainsRunStore((s) => s.version);
  const busy = useSpatialDomainsRunStore((s) => s.running);
  const start = useSpatialDomainsRunStore((s) => s.start);

  const [picked, setPicked] = useState("");
  const [nDistances, setNDistances] = useState(50);
  const [upTo, setUpTo] = useState("");
  const [splits, setSplits] = useState(false);
  const [section, setSection] = useState("");
  const [info, setInfo] = useState<CoOccurrenceInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const column = picked || (columns.some((c) => c.name === colorBy) ? colorBy : labelDefault);
  const combine = splits ? "squidpy_splits" : "sum";
  const perUnit = info?.plan?.um_per_unit ?? info?.result.um_per_unit ?? 1;
  const reach = Number.parseFloat(upTo);
  const maxDistance = Number.isFinite(reach) && reach > 0 ? reach / perUnit : undefined;

  useEffect(() => {
    if (!datasetId || !column) return;
    let cancelled = false;
    setError(null);
    getCoOccurrence(datasetId, { column, interval: nDistances, combine, max_distance: maxDistance, section: section || undefined })
      .then((i) => !cancelled && setInfo(i))
      .catch((e) => {
        if (cancelled) return;
        if (section) setSection(""); // that section is not in the run shown: back to all sections
        else setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, column, nDistances, combine, maxDistance, version, section]);

  if (!datasetId) return null;
  const plan = info?.plan ?? null;
  const result = info?.result;
  const canSplit = !!plan && plan.libraries.length === 1 && plan.n_labelled > (info?.defaults.split_above ?? Infinity);
  const unit = unitText(plan?.distance_unit ?? result?.distance_unit);

  const go = () => {
    setPicked(column);
    setSection("");
    void start(
      datasetId,
      "cooccurrence",
      "Co-occurrence",
      (h) => runCoOccurrence(datasetId, { cluster_key: column, interval: nDistances, combine, max_distance: maxDistance }, h),
      () => null,
    );
  };

  return (
    <section className="space-y-2 border-t border-slate-200 pt-3" aria-label="Co-occurrence" data-testid="cooccurrence-card">
      <div className="text-xs font-medium text-slate-700">
        <HoverTip
          width={320}
          tip={<>
            The same statistic as squidpy&apos;s co_occurrence: for labels a and b within a distance, P(b | a) / P(b), both from
            the pairs of observations no further apart than that distance, so each distance includes the shorter ones. Above 1,
            b is near a more often than among all pairs at that range. Pairs are counted within each section, never between
            sections, and each section is scored against its own label shares before the sections are combined. On spots or
            bins that mix cells it speaks about their labels, not about touching cells. It is a description, not a test.
          </>}
        >
          Co-occurrence
        </HoverTip>
      </div>
      <div className="flex flex-wrap items-end gap-2 text-xs">
        <label className={FIELD}>
          <Help id="domains.cooccur-labels">Labels</Help>
          <select
            value={column}
            onChange={(e) => {
              setPicked(e.target.value);
              setSection("");
            }}
            className={INPUT}
            aria-label="Co-occurrence labels"
          >
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                {prettyObsLabel(c.name)} ({c.n_unique})
              </option>
            ))}
          </select>
        </label>
        <label className={FIELD}>
          <Help id="domains.cooccur-distances">Distances</Help>
          <input
            type="number"
            min={2}
            max={200}
            value={nDistances}
            onChange={(e) => {
              const n = Number.parseInt(e.target.value, 10);
              if (Number.isFinite(n) && n >= 2 && n <= 200) setNDistances(n);
            }}
            className={`${INPUT} w-16`}
          />
        </label>
        <label className={FIELD}>
          <Help id="domains.cooccur-reach">Up to ({unit})</Help>
          <input
            type="number"
            min={0}
            value={upTo}
            placeholder={plan && !upTo ? `${Math.round(reachOf(plan)).toLocaleString()}` : ""}
            onChange={(e) => setUpTo(e.target.value)}
            className={`${INPUT} w-20`}
          />
        </label>
        <button type="button" disabled={busy || !column || !plan} onClick={go} className={SOLID}>
          Compute co-occurrence
        </button>
      </div>
      {canSplit && (
        <label
          className="flex items-center gap-1.5 text-[11px] text-slate-500"
          title="squidpy's automatic n_splits above its size limit: the observations cut by row order, one score per pair of splits, then the mean"
        >
          <input type="checkbox" checked={splits} onChange={(e) => setSplits(e.target.checked)} />
          Reproduce squidpy&apos;s default for this size (mean over row-order splits; depends on the row order, and scores run low)
        </label>
      )}
      {info?.reason && <p className="text-[11px] leading-snug text-slate-500">{info.reason}</p>}
      {info?.verdict && <PreflightNote verdict={info.verdict} />}
      {plan?.notes
        .filter((n) => !result?.notes?.includes(n))
        .map((n) => (
          <p key={n} className="text-[11px] leading-snug text-slate-500">
            {n}
          </p>
        ))}
      <SpatialRunStatus kind="cooccurrence" />
      <RequestError message={error} />
      {result?.available && (
        <CoOccurrenceResults
          key={result.column}
          result={result}
          section={section}
          onSection={setSection}
        />
      )}
    </section>
  );
}

function CoOccurrenceResults({
  result,
  section,
  onSection,
}: {
  result: CoOccurrenceResult;
  section: string;
  onSection: (section: string) => void;
}) {
  const labels = useMemo(() => result.labels ?? [], [result]);
  const distances = useMemo(() => result.distances ?? [], [result]);
  const [condition, setCondition] = useState(labels[0] ?? "");
  const [at, setAt] = useState(() => firstUsable(result));
  const colorOf = useCategoryColors(result.column);
  const unit = unitText(result.distance_unit);
  const own = labels.includes(condition) ? condition : (labels[0] ?? "");
  const t = Math.min(at, Math.max(0, distances.length - 1));

  const curves = useMemo(() => curvesOf(result, own), [result, own]);
  const matrix = useMemo(() => matrixAt(result, t), [result, t]);
  const scale = useMemo(() => scoreScale(matrix), [matrix]);
  const height = Math.min(460, Math.max(220, labels.length * 22 + 110));
  const units = result.unit === "cell" ? "cells" : `${result.unit ?? "cell"}s`;
  const first = distances[0];
  const last = distances[distances.length - 1];
  const sampled = (result.n_used ?? 0) < (result.n_labelled ?? 0);
  const perSection = result.sections ?? [];
  const shownSection = result.section ?? null;
  const combined = result.sections_combined === "within_sections" && !shownSection;

  return (
    <div className="space-y-2" data-testid="cooccurrence-results">
      {result.stale && <p className="text-[11px] leading-snug text-amber-700">{result.stale_reason}</p>}
      <div className="flex flex-wrap items-end gap-2 text-xs">
        <label className={FIELD}>
          <Help id="domains.cooccur-near">Near which label</Help>
          <select value={own} onChange={(e) => setCondition(e.target.value)} className={INPUT} aria-label="Conditioning label">
            {labels.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </label>
        {perSection.length > 1 && (
          <label className={FIELD}>
            <Help id="domains.cooccur-section">Section</Help>
            <select
              value={section}
              onChange={(e) => onSection(e.target.value)}
              className={INPUT}
              aria-label="Co-occurrence section"
            >
              <option value="">All sections, combined</option>
              {perSection.map((s) => (
                <option key={s.library_id} value={s.library_id}>
                  {s.library_id} ({s.pairs.toLocaleString()} pairs)
                </option>
              ))}
            </select>
          </label>
        )}
        <button
          type="button"
          className={OUTLINE}
          onClick={() =>
            downloadCsv(
              `co_occurrence_${result.column ?? ""}${shownSection ? `_${shownSection}` : ""}`,
              csvHeaders(result),
              csvRows(result),
            )
          }
        >
          Download CSV
        </button>
      </div>
      <figure className="space-y-1" data-testid="cooccurrence-curves">
        <Plot
          data={curves.map((c) => ({
            type: "scatter" as const,
            mode: "lines" as const,
            name: c.label,
            x: c.distances,
            y: c.scores,
            customdata: c.pairs,
            line: { color: rgbCss(colorOf(c.label)), width: c.label === own ? 2.5 : 1.5 },
            hovertemplate: `${c.label} near ${own}<br>within %{x} ${unit}: score %{y:.2f}<br>%{customdata:,} pairs<extra></extra>`,
          }))}
          layout={{
            height: 280,
            margin: { l: 44, r: 10, t: 6, b: 36 },
            xaxis: { title: { text: `Distance (${unit})`, font: AXIS }, tickfont: TICK, gridcolor: "#f1f5f9", zeroline: false },
            yaxis: { title: { text: "Co-occurrence score", font: AXIS }, tickfont: TICK, gridcolor: "#f1f5f9", zeroline: false },
            shapes: [
              {
                type: "line" as const,
                xref: "paper" as const,
                x0: 0,
                x1: 1,
                y0: 1,
                y1: 1,
                line: { color: "#94a3b8", width: 1, dash: "dash" as const },
              },
            ],
            legend: { font: { size: 10, color: "#64748b" } },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
          }}
          config={plotConfig(`co_occurrence_${result.column ?? ""}_${own}`)}
          useResizeHandler
          style={{ width: "100%" }}
        />
        <Caption
          n={[
            sampled
              ? { value: result.n_used ?? 0, of: result.n_labelled ?? 0, unit: units }
              : { value: result.n_used ?? 0, unit: units },
            { value: labels.length, unit: "labels" },
          ]}
        >
          Co-occurrence of {prettyObsLabel(result.column ?? "")} near {own}, within each distance from {first?.toLocaleString()} to{" "}
          {last?.toLocaleString()} {unit} (each includes the shorter ones). A score above 1: that label is found within the distance
          of label {own} more often than among all pairs at that range; below 1, less often; the dashed line is 1. Nearby{" "}
          {units} are not necessarily interacting. {distanceSource(result)}
          {shownSection ? ` Section ${shownSection} alone, against its own label shares.` : ""}
          {combined
            ? ` ${perSection.length} sections: each scored against its own label shares, then combined, each weighted by the pairs of the two labels it would have at random.`
            : ""}
        </Caption>
      </figure>
      <figure className="space-y-1" data-testid="cooccurrence-heatmap">
        <label className={FIELD}>
          <Help id="domains.cooccur-heatmap">Heatmap within ({unit})</Help>
          <select value={t} onChange={(e) => setAt(Number(e.target.value))} className={`${INPUT} w-28`} aria-label="Heatmap distance">
            {distances.map((d, i) => (
              <option key={i} value={i}>
                {d.toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <Plot
          data={[
            {
              type: "heatmap" as const,
              x: labels,
              y: labels,
              z: matrix,
              ...scale,
              xgap: 1,
              ygap: 1,
              colorbar: { title: { text: "score", font: { size: 10 } }, thickness: 10 },
              hoverongaps: false,
              hovertemplate: "%{x} near %{y}<br>score %{z:.2f}<extra></extra>",
              ...(labels.length <= MAX_LABELLED ? { texttemplate: "%{z:.1f}", textfont: { size: 9 } } : {}),
            },
          ]}
          layout={{
            height,
            margin: { l: 10, r: 10, t: 6, b: 10 },
            xaxis: { type: "category" as const, tickangle: -45, automargin: true, tickfont: { size: 10 } },
            yaxis: { type: "category" as const, automargin: true, tickfont: { size: 10 }, autorange: "reversed" as const },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
          }}
          config={plotConfig(`co_occurrence_heatmap_${result.column ?? ""}`)}
          useResizeHandler
          style={{ width: "100%", height }}
        />
        <Caption n={{ value: labels.length, unit: "labels" }}>
          Score within {distances[t]?.toLocaleString()} {unit}; blue below 1, red above. A blank cell has no pair of {units} at
          this range. The score is symmetric: a near b equals b near a.
        </Caption>
      </figure>
      {result.notes?.map((n) => (
        <p key={n} className="text-[11px] text-amber-700">
          {n}
        </p>
      ))}
    </div>
  );
}
