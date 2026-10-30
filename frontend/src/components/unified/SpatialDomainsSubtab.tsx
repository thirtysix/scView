import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  getSpatialDomains,
  runDomains,
  runEnrichment,
  runNiches,
  type DomainsInfo,
  type DomainsResult,
  type EnrichmentResult,
  type NichesResult,
  type StabilityCurve,
} from "@/api/spatialDomains";
import { SPATIAL_EMBEDDING } from "@/api/spatial";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import Plot from "@/components/plots/Plot";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { errorMessage } from "@/lib/errorMessage";
import { formatDuration } from "@/lib/hvgSensitivity";
import { prettyObsLabel } from "@/lib/formatting";
import { plotConfig } from "@/lib/plotConfig";
import { unitWord } from "@/lib/spatial";
import { labelColumns, planText, repName, runShare, runText, scanText, strongestPairs, zRange } from "@/lib/spatialDomains";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSpatialDomainsRunStore } from "@/stores/spatialDomainsRunStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

/** Cell values are written in a heatmap up to this many labels. */
const MAX_LABELLED = 15;
const INPUT = "rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700";
const FIELD = "flex flex-col gap-0.5 text-[11px] text-slate-500";
const OUTLINE =
  "rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary disabled:opacity-50";
const SOLID = "rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50";

const num = (v: string, fallback: number) => {
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? n : fallback;
};

/**
 * Spatial domains and niches (backend core/spatial_domains.py): CellCharter-style domains
 * with a stability curve for k, squidpy's neighbourhood enrichment, composition niches.
 * A finished run colours the tissue plot by its column.
 */
export function SpatialDomainsSubtab({
  groupByColumn,
  setGroupByColumn,
}: {
  groupByColumn: string;
  setGroupByColumn: (column: string) => void;
}) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const spatial = dataset?.spatial ?? null;
  const run = useSpatialDomainsRunStore();
  const live = run.datasetId === datasetId;
  const refreshDataset = useDatasetRefresh();
  const setColorBy = useSettingsStore((s) => s.setColorBy);
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);
  const setActiveSubtab = useUnifiedViewStore((s) => s.setActiveSubtab);

  const [info, setInfo] = useState<DomainsInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [k, setK] = useState(7);
  const [hops, setHops] = useState(3);
  const [pcs, setPcs] = useState(10);
  const [name, setName] = useState("");
  const [nameEdited, setNameEdited] = useState(false);
  const [replace, setReplace] = useState(false);
  const [enrichColumn, setEnrichColumn] = useState("");
  const [nicheLabel, setNicheLabel] = useState("");
  const [nicheWindow, setNicheWindow] = useState(20);
  const [nicheCount, setNicheCount] = useState(8);

  useEffect(() => {
    if (!datasetId || !spatial) return;
    let cancelled = false;
    setError(null);
    getSpatialDomains(datasetId, { k, n_hops: hops, n_pcs: pcs, enrichment_column: enrichColumn || undefined })
      .then((i) => !cancelled && setInfo(i))
      .catch((e) => !cancelled && setError(errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [datasetId, !!spatial, k, hops, pcs, enrichColumn, run.version]); // eslint-disable-line react-hooks/exhaustive-deps

  // The name offered follows k until the user types their own.
  useEffect(() => {
    if (info && !nameEdited) setName(info.free_column);
  }, [info, nameEdited]);

  const units = unitWord(spatial);
  const unit = unitWord(spatial, false);
  const columns = useMemo(() => labelColumns(dataset?.obs_columns ?? []), [dataset]);
  const labelDefault = columns.some((c) => c.name === groupByColumn) ? groupByColumn : (columns[0]?.name ?? "");
  const enrichPick = enrichColumn || labelDefault;
  const nichePick = nicheLabel || labelDefault;

  // A finished run, once: list its column, then colour the tissue by it (the panel draws the
  // tissue only once its embedding has loaded).
  const finished = run.finished;
  useEffect(() => {
    if (!finished || !datasetId || !live) return;
    useSpatialDomainsRunStore.setState({ finished: null });
    void refreshDataset(datasetId, finished)
      .then(() => {
        setColorBy(finished);
        setEmbedding(SPATIAL_EMBEDDING);
      })
      .catch(() => undefined);
  }, [finished, datasetId, live, refreshDataset, setColorBy, setEmbedding]);

  const showColumn = useCallback(
    (column: string) => {
      setColorBy(column);
      setEmbedding(SPATIAL_EMBEDDING);
    },
    [setColorBy, setEmbedding],
  );
  const findMarkers = useCallback(
    (column: string) => {
      setGroupByColumn(column);
      setActiveSubtab("markers");
    },
    [setGroupByColumn, setActiveSubtab],
  );

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  if (!spatial) {
    return <div className="text-xs text-slate-400">This dataset has no spot or cell positions.</div>;
  }

  const plan = info?.runnable.plan ?? null;
  const domains = info?.domains;
  const nameHeld = !!domains?.columns.includes(name);
  const nameBlocked = !name.trim() || (nameHeld && !replace);
  const busy = run.running;
  const curve: StabilityCurve | null = (live && run.curve) || domains?.scan || null;

  const startDomains = (asked: number | null) => {
    const body = { k: asked, n_hops: hops, n_pcs: pcs, column: name, overwrite: replace };
    void run
      .start(datasetId, "domains", "Finding domains", (h) => runDomains(datasetId, body, h), (r: DomainsResult) => r.column ?? null)
      .then((r) => {
        if (!r) return;
        setNameEdited(false);
        setReplace(false);
      });
  };
  const startEnrichment = () => {
    setEnrichColumn(enrichPick);
    void run.start(
      datasetId,
      "enrichment",
      "Neighbourhood enrichment",
      (h) => runEnrichment(datasetId, { column: enrichPick }, h),
      () => null,
    );
  };
  const startNiches = () => {
    const column = info?.free_niche_column ?? "spatial_niche"; // a new column each run: nothing is replaced
    void run.start(
      datasetId,
      "niches",
      "Composition niches",
      (h) => runNiches(datasetId, { label_column: nichePick, k_window: nicheWindow, n_niches: nicheCount, column }, h),
      (r: NichesResult) => r.column ?? null,
    );
  };

  return (
    <div className="space-y-4">
      {/* Domains */}
      <section className="space-y-2" aria-label="Spatial domains">
        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
          Spatial domains
          <InfoTip label="About spatial domains" width={300}>
            CellCharter&apos;s method (Varrone et al. 2024, Nat Genet): each {unit}&apos;s profile is joined with the mean
            profile of its neighbours at 1, 2, ... hops, and a Gaussian mixture clusters those. With 0 hops it is the same
            clustering with no spatial information.
          </InfoTip>
        </div>
        <p className="text-[11px] leading-snug text-slate-500">
          A spatial domain is a stretch of tissue whose {units} look alike and have alike surroundings, such as a cortical
          layer or a tumour edge; it is a place in the tissue, not a cell type.
        </p>
        {info && !info.runnable.ok && <p className="text-[11px] leading-snug text-slate-500">{info.runnable.reason}</p>}

        <div className="flex flex-wrap items-end gap-2 text-xs">
          <label className={FIELD}>
            Domains (k)
            <input
              type="number"
              min={2}
              max={40}
              value={k}
              onChange={(e) => setK(num(e.target.value, k))}
              className={`${INPUT} w-14`}
            />
          </label>
          <label className={FIELD} title="How many rings of neighbours each profile is joined with; 0 = no spatial information">
            Hops
            <input
              type="number"
              min={0}
              max={6}
              value={hops}
              onChange={(e) => setHops(num(e.target.value, hops))}
              className={`${INPUT} w-14`}
            />
          </label>
          <label className={FIELD} title="Leading components of the embedding each profile uses">
            Components
            <input
              type="number"
              min={2}
              max={plan?.n_pcs_available ?? 100}
              value={pcs}
              onChange={(e) => setPcs(num(e.target.value, pcs))}
              className={`${INPUT} w-14`}
            />
          </label>
          <label className={FIELD}>
            Save as
            <input
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                setNameEdited(true);
              }}
              className={`${INPUT} w-40`}
            />
          </label>
          <button type="button" disabled={busy || !info?.runnable.ok || nameBlocked} onClick={() => startDomains(k)} className={SOLID}>
            Find domains
          </button>
          <button
            type="button"
            disabled={busy || !info?.runnable.ok || nameBlocked}
            onClick={() => startDomains(null)}
            className={OUTLINE}
            title="Fit each k several times and use the k whose result agrees best with its neighbours'"
          >
            Suggest k
          </button>
        </div>
        {nameHeld && (
          <label className="flex items-center gap-1.5 text-[11px] text-amber-700">
            <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
            Replace the earlier result in &quot;{name}&quot;
          </label>
        )}
        {plan && (
          <div className="space-y-0.5 text-[11px] leading-snug text-slate-500" data-testid="domains-plan">
            <p>{planText({ ...plan, k })}</p>
            <p>{scanText(plan)}</p>
            {plan.notes
              .filter((n) => !domains?.notes?.includes(n))
              .map((n) => (
                <p key={n}>{n}</p>
              ))}
          </div>
        )}
        <RunStatus kind="domains" />
        <RequestError message={error} />

        {curve && (
          <StabilityPlot
            curve={curve}
            current={domains?.k ?? null}
            onPick={(picked) => {
              setK(picked);
              setNameEdited(false);
            }}
          />
        )}
        {domains?.available && (
          <DomainsTable
            result={domains}
            units={units}
            onShow={showColumn}
            onMarkers={findMarkers}
          />
        )}
      </section>

      {/* Neighbourhood enrichment */}
      <section className="space-y-2 border-t border-slate-200 pt-3" aria-label="Neighbourhood enrichment">
        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
          Neighbourhood enrichment
          <InfoTip label="About neighbourhood enrichment" width={300}>
            Counts how often each pair of labels sits side by side on the tissue graph, against the same labels shuffled
            within each section ({(info?.defaults.n_perms ?? 1000).toLocaleString()} times): z above 0 means more often than
            chance, below 0 less often. The same statistic as squidpy&apos;s nhood_enrichment. On {units} that mix cells it
            speaks about the {units}&apos; labels, not about touching cells.
          </InfoTip>
        </div>
        <div className="flex flex-wrap items-end gap-2 text-xs">
          <label className={FIELD}>
            Labels
            <select value={enrichPick} onChange={(e) => setEnrichColumn(e.target.value)} className={INPUT}>
              {columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {prettyObsLabel(c.name)} ({c.n_unique})
                </option>
              ))}
            </select>
          </label>
          <button type="button" disabled={busy || !enrichPick} onClick={startEnrichment} className={SOLID}>
            Compute enrichment
          </button>
          {info && <span className="text-[11px] text-slate-500">about {formatDuration(info.estimates.enrichment_seconds)}</span>}
        </div>
        <RunStatus kind="enrichment" />
        {info?.enrichment.available && <EnrichmentHeatmap result={info.enrichment} />}
      </section>

      {/* Composition niches */}
      <section className="space-y-2 border-t border-slate-200 pt-3" aria-label="Composition niches">
        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
          Composition niches
          <InfoTip label="About composition niches" width={300}>
            Each {unit}&apos;s window is its nearest {units} in the section (itself included); {units} whose windows hold
            the same mix of labels form one niche (k-means, Schurch et al. 2020, Cell).
          </InfoTip>
        </div>
        {info && !info.niches_allowed.ok && info.niches_allowed.reason ? (
          <p className="text-[11px] leading-snug text-slate-500" data-testid="niches-refused">
            {info.niches_allowed.reason}
          </p>
        ) : (
          <>
            {info?.niches_allowed.note && <p className="text-[11px] leading-snug text-amber-700">{info.niches_allowed.note}</p>}
            <div className="flex flex-wrap items-end gap-2 text-xs">
              <label className={FIELD}>
                Labels
                <select value={nichePick} onChange={(e) => setNicheLabel(e.target.value)} className={INPUT}>
                  {columns.map((c) => (
                    <option key={c.name} value={c.name}>
                      {prettyObsLabel(c.name)} ({c.n_unique})
                    </option>
                  ))}
                </select>
              </label>
              <label className={FIELD}>
                Window
                <input
                  type="number"
                  min={2}
                  max={200}
                  value={nicheWindow}
                  onChange={(e) => setNicheWindow(num(e.target.value, nicheWindow))}
                  className={`${INPUT} w-14`}
                />
              </label>
              <label className={FIELD}>
                Niches
                <input
                  type="number"
                  min={2}
                  max={40}
                  value={nicheCount}
                  onChange={(e) => setNicheCount(num(e.target.value, nicheCount))}
                  className={`${INPUT} w-14`}
                />
              </label>
              <button type="button" disabled={busy || !nichePick} onClick={startNiches} className={SOLID}>
                Find niches
              </button>
            </div>
            <RunStatus kind="niches" />
            {info?.niches.available && <NicheHeatmap result={info.niches} onShow={showColumn} onMarkers={findMarkers} />}
          </>
        )}
      </section>
    </div>
  );
}

function RunStatus({ kind }: { kind: "domains" | "enrichment" | "niches" }) {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const run = useSpatialDomainsRunStore();
  if (run.datasetId !== datasetId || run.kind !== kind) return null;
  return (
    <>
      {run.running && (
        <div className="space-y-1" data-testid={`${kind}-progress`}>
          <p className="flex items-center gap-1.5 text-[11px] text-slate-600">
            <Loader2 className="h-3 w-3 animate-spin" />
            {runText(run.stage, run.progress)}
          </p>
          <div className="h-1 w-full rounded bg-slate-100">
            <div className="h-1 rounded bg-primary" style={{ width: `${Math.round(runShare(run.stage, run.progress) * 100)}%` }} />
          </div>
          <button type="button" onClick={() => run.cancel()} className={OUTLINE}>
            Cancel
          </button>
        </div>
      )}
      <RequestError message={run.error} />
      {run.warnings.map((w) => (
        <p key={w} className="text-[11px] text-amber-700">
          {w}
        </p>
      ))}
    </>
  );
}

function StabilityPlot({
  curve,
  current,
  onPick,
}: {
  curve: StabilityCurve;
  current: number | null;
  onPick: (k: number) => void;
}) {
  const inRange = current != null && curve.ks.includes(current);
  return (
    <figure className="space-y-1" data-testid="stability-curve">
      <Plot
        data={[
          {
            type: "scatter" as const,
            mode: "lines+markers" as const,
            x: curve.ks,
            y: curve.mean,
            error_y: { type: "data" as const, array: curve.sd, visible: true, thickness: 1, width: 2, color: "#94a3b8" },
            line: { color: "#2563eb", width: 1.5 },
            marker: { size: 6, color: "#2563eb" },
            hovertemplate: "k = %{x}<br>stability %{y:.2f}<extra></extra>",
          },
          ...(inRange
            ? [
                {
                  type: "scatter" as const,
                  mode: "markers" as const,
                  x: [current],
                  y: [curve.mean[curve.ks.indexOf(current)] ?? 0],
                  marker: { size: 11, color: "rgba(0,0,0,0)", line: { color: "#dc2626", width: 2 } },
                  hoverinfo: "skip" as const,
                },
              ]
            : []),
        ]}
        layout={{
          height: 170,
          margin: { l: 40, r: 10, t: 6, b: 32 },
          xaxis: { title: { text: "k", font: { size: 11, color: "#64748b" } }, dtick: 1, tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9", zeroline: false },
          yaxis: { title: { text: "stability", font: { size: 11, color: "#64748b" } }, tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9", zeroline: false },
          showlegend: false,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }}
        config={plotConfig("spatial_domain_stability")}
        useResizeHandler
        style={{ width: "100%" }}
        onClick={(ev: { points?: { x?: unknown }[] }) => {
          const x = ev.points?.[0]?.x;
          if (typeof x === "number") onPick(x);
        }}
      />
      <Caption n={{ value: curve.n_fit, unit: "observations per fit" }}>
        Stability of each k: mean ARI of its {curve.repeats} fits against the other fits at k - 1 and k + 1 (error bars: SD).{" "}
        {curve.has_peak ? `Suggested k = ${curve.suggested}, the highest local peak.` : `No local peak, so k = ${curve.suggested}, the highest value.`}{" "}
        It advises; the biology decides. Click a point to use that k.
      </Caption>
    </figure>
  );
}

function DomainsTable({
  result,
  units,
  onShow,
  onMarkers,
}: {
  result: DomainsResult;
  units: string;
  onShow: (column: string) => void;
  onMarkers: (column: string) => void;
}) {
  const column = result.column ?? "";
  const libs = Object.keys(result.per_library ?? {});
  const sizes = result.sizes ?? [];
  return (
    <div className="space-y-1.5">
      {result.stale && (
        <p className="text-[11px] leading-snug text-amber-700" data-testid="domains-stale">
          {result.stale_reason}
        </p>
      )}
      <Caption n={{ value: result.n_obs ?? 0, unit: units }}>
        {result.k} domains in &quot;{column}&quot; (k {result.suggested ? "suggested" : "chosen"}), from the first {result.n_pcs}{" "}
        components of {repName(result.rep ?? "")} and {result.n_hops} hops.{" "}
        {(result.n_fit ?? 0) < (result.n_obs ?? 0) ? `Fitted on a seeded sample of ${(result.n_fit ?? 0).toLocaleString()}. ` : ""}
        {result.coherence
          ? `${Math.round(result.coherence.within * 100)}% of neighbouring ${units} share a domain (same labels shuffled: ${Math.round(result.coherence.shuffled * 100)}%).`
          : ""}
      </Caption>
      {result.notes?.map((n) => (
        <p key={n} className="text-[11px] leading-snug text-amber-700">
          {n}
        </p>
      ))}
      <div className="flex flex-wrap gap-1.5">
        <button type="button" onClick={() => onShow(column)} className={OUTLINE}>
          Colour the tissue
        </button>
        <button type="button" onClick={() => onMarkers(column)} className={OUTLINE} title="Open Markers grouped by these domains">
          Find markers
        </button>
      </div>
      <div className="max-h-56 overflow-auto rounded-lg border border-slate-200">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-slate-50">
            <tr className="border-b border-slate-200 text-slate-600">
              <th className="px-2 py-1 text-left" title="Numbered by size">
                Domain
              </th>
              <th className="px-2 py-1 text-right">{units}</th>
              {libs.length > 1 && libs.map((l) => <th key={l} className="px-2 py-1 text-right">{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {sizes.map((n, i) => (
              <tr key={i} className="border-b border-slate-100">
                <td className="px-2 py-0.5 font-medium text-slate-700">{i}</td>
                <td className="px-2 py-0.5 text-right tabular-nums text-slate-700">{n.toLocaleString()}</td>
                {libs.length > 1 &&
                  libs.map((l) => (
                    <td key={l} className="px-2 py-0.5 text-right tabular-nums text-slate-700">
                      {(result.per_library?.[l]?.[i] ?? 0).toLocaleString()}
                    </td>
                  ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] leading-snug text-slate-400">{result.method}</p>
    </div>
  );
}

function EnrichmentHeatmap({ result }: { result: EnrichmentResult }) {
  const labels = result.labels ?? [];
  const z = result.zscore ?? [];
  const top = zRange(z);
  const pairs = strongestPairs(labels, z);
  const units = result.unit === "cell" ? "cells" : `${result.unit ?? "cell"}s`;
  const height = Math.min(460, Math.max(220, labels.length * 22 + 110));
  return (
    <figure className="space-y-1" data-testid="enrichment-heatmap">
      {result.stale && <p className="text-[11px] leading-snug text-amber-700">{result.stale_reason}</p>}
      <Plot
        data={[
          {
            type: "heatmap" as const,
            x: labels,
            y: labels,
            z,
            colorscale: "RdBu", // Plotly's RdBu: blue low, red high; the range is symmetric about 0
            zmin: -top,
            zmax: top,
            xgap: 1,
            ygap: 1,
            colorbar: { title: { text: "z", font: { size: 10 } }, thickness: 10 },
            hovertemplate: "%{y} next to %{x}<br>z %{z:.1f}<extra></extra>",
            ...(labels.length <= MAX_LABELLED ? { texttemplate: "%{z:.0f}", textfont: { size: 9 } } : {}),
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
        config={plotConfig(`nhood_enrichment_${result.column ?? ""}`)}
        useResizeHandler
        style={{ width: "100%", height }}
      />
      <Caption n={[{ value: result.n_labelled ?? 0, unit: units }, { value: labels.length, unit: "labels" }]}>
        Neighbourhood enrichment of {prettyObsLabel(result.column ?? "")} ({(result.n_perms ?? 0).toLocaleString()} permutations).{" "}
        {pairs.high ? `Most often side by side: ${pairs.high[0]} and ${pairs.high[1]} (z ${pairs.high[2].toFixed(1)}). ` : ""}
        {pairs.low ? `Most apart: ${pairs.low[0]} and ${pairs.low[1]} (z ${pairs.low[2].toFixed(1)}).` : ""}
      </Caption>
      {result.notes?.map((n) => (
        <p key={n} className="text-[11px] text-amber-700">
          {n}
        </p>
      ))}
    </figure>
  );
}

function NicheHeatmap({
  result,
  onShow,
  onMarkers,
}: {
  result: NichesResult;
  onShow: (column: string) => void;
  onMarkers: (column: string) => void;
}) {
  const labels = result.labels ?? [];
  const niches = result.niches ?? [];
  const sizes = result.sizes ?? [];
  const column = result.column ?? "";
  const units = result.unit === "cell" ? "cells" : `${result.unit ?? "cell"}s`;
  const height = Math.min(420, Math.max(200, niches.length * 24 + 110));
  const rows = niches.map((n, i) => `${n} (${(sizes[i] ?? 0).toLocaleString()})`);
  return (
    <figure className="space-y-1" data-testid="niche-heatmap">
      {result.stale && <p className="text-[11px] leading-snug text-amber-700">{result.stale_reason}</p>}
      <div className="flex flex-wrap gap-1.5">
        <button type="button" onClick={() => onShow(column)} className={OUTLINE}>
          Colour the tissue
        </button>
        <button type="button" onClick={() => onMarkers(column)} className={OUTLINE} title="Open Markers grouped by these niches">
          Find markers
        </button>
      </div>
      <Plot
        data={[
          {
            type: "heatmap" as const,
            x: labels,
            y: rows,
            z: result.fractions ?? [],
            colorscale: "Blues",
            reversescale: true,
            zmin: 0,
            zmax: 1,
            xgap: 1,
            ygap: 1,
            colorbar: { title: { text: "share", font: { size: 10 } }, thickness: 10 },
            hovertemplate: "niche %{y}<br>%{x}: %{z:.0%}<extra></extra>",
            ...(labels.length <= MAX_LABELLED ? { texttemplate: "%{z:.0%}", textfont: { size: 9 } } : {}),
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
        config={plotConfig(`niches_${column}`)}
        useResizeHandler
        style={{ width: "100%", height }}
      />
      <Caption n={{ value: result.n_labelled ?? 0, unit: units }}>
        {result.n_niches} niches in &quot;{column}&quot;: the mean share of each {prettyObsLabel(result.label_column ?? "")} label in the
        windows of each niche (the {result.k_window} nearest).
      </Caption>
      {result.notes?.map((n) => (
        <p key={n} className="text-[11px] text-amber-700">
          {n}
        </p>
      ))}
    </figure>
  );
}
