import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { getCoverage, getFragments, type Coverage } from "@/api/atacFragments";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import Plot from "@/components/plots/Plot";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { coverageFigure } from "@/lib/atacPlots";
import { rgbCss } from "@/lib/categoryColors";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const MAX_GROUPS_SHOWN = 24;

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** A region typed as chr1:1000-5000 (commas fine), else a gene. */
export function asRequest(text: string): { gene?: string; region?: string } {
  const t = text.trim();
  return /^[^:\s]+:[\d,]+-[\d,]+$/.test(t) ? { region: t } : { gene: t };
}

/**
 * Coverage tracks from the ATAC fragments file (backend core/atac_fragments.py): one filled
 * track per group for a gene (its body +- 5 kb) or a region of up to 2 Mb, peak ticks and gene
 * arrows beneath. Renders nothing until a fragments file and its index are attached.
 */
export function AtacCoverage({ datasetId }: { datasetId: string }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const [attached, setAttached] = useState(false);
  const [maxWindow, setMaxWindow] = useState(2_000_000);
  const [genePad, setGenePad] = useState(5000);
  const [text, setText] = useState("");
  const [groupBy, setGroupBy] = useState("");
  const [cov, setCov] = useState<Coverage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  const columns = useMemo(
    () => (dataset?.obs_columns ?? []).filter(
      (c) => (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") && c.n_unique >= 1 && c.n_unique <= 100,
    ),
    [dataset],
  );
  // The group column starts as the current colour-by when that is categorical.
  useEffect(() => {
    setGroupBy((cur) => {
      if (cur && columns.some((c) => c.name === cur)) return cur;
      return columns.some((c) => c.name === colorBy) ? colorBy : "";
    });
  }, [columns, colorBy]);

  useEffect(() => {
    let cancelled = false;
    getFragments(datasetId)
      .then((s) => {
        if (cancelled) return;
        setAttached(s.attached);
        if (s.max_window) setMaxWindow(s.max_window);
        if (s.gene_pad) setGenePad(s.gene_pad);
      })
      .catch(() => !cancelled && setAttached(false));
    return () => {
      cancelled = true;
      abort.current?.abort();
    };
  }, [datasetId]);

  const colors = useCategoryColors(groupBy || null);
  const figure = useMemo(() => {
    if (!cov) return null;
    return coverageFigure(cov, (g) => (cov.group_by ? rgbCss(colors(g)) : "#2563eb"));
  }, [cov, colors]);

  const draw = async () => {
    if (!text.trim()) {
      setError("Type a gene or a region like chr1:1000-5000.");
      return;
    }
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setBusy(true);
    setError(null);
    try {
      setCov(await getCoverage(datasetId, { ...asRequest(text), groupBy: groupBy || null }, ctl.signal));
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setCov(null);
        setError(errorMessage(e));
      }
    } finally {
      if (abort.current === ctl) setBusy(false);
    }
  };

  if (!attached) return null;
  const nGroups = columns.find((c) => c.name === groupBy)?.n_unique ?? 1;

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Coverage tracks">
      <h4 className="flex items-center gap-1 text-xs font-semibold text-slate-600">
        <span>Coverage tracks</span>
        <InfoTip label="About coverage tracks" width={320}>
          How many fragments cover each base, per group of cells, drawn from the fragments file for the window you ask
          for. A peak in one group&rsquo;s track only is accessible chromatin in those cells. A gene shows its body{" "}
          {(genePad / 1000).toFixed(0)} kb either side, so the promoter is in view.
        </InfoTip>
      </h4>
      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void draw();
        }}
      >
        <input
          aria-label="Gene or region"
          className="w-52 rounded border border-slate-200 px-2 py-1 text-xs text-slate-700"
          placeholder="MS4A1 or chr11:60450000-60480000"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <label className="flex items-center gap-1 text-[11px] text-slate-500">
          Group by
          <select
            aria-label="Group by"
            className="rounded border border-slate-200 bg-white px-1 py-0.5 text-xs text-slate-700"
            value={groupBy}
            onChange={(e) => setGroupBy(e.target.value)}
          >
            <option value="">All cells</option>
            {columns.map((c) => (
              <option key={c.name} value={c.name}>{c.name}</option>
            ))}
          </select>
        </label>
        <button
          type="submit"
          disabled={busy}
          className="flex items-center gap-1 rounded bg-blue-600 px-2 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {busy && <Loader2 className="h-3 w-3 animate-spin" />} Draw tracks
        </button>
      </form>
      <p className="text-[11px] leading-snug text-slate-400">
        Up to {(maxWindow / 1e6).toFixed(0)} Mb at a time.
        {nGroups > MAX_GROUPS_SHOWN && ` This column has ${nGroups} groups; the ${MAX_GROUPS_SHOWN} largest are drawn.`}
      </p>
      <RequestError message={error} />
      {cov && figure && (
        <figure className="space-y-1" data-testid="atac-coverage">
          <Plot
            data={figure.data}
            layout={figure.layout}
            config={plotConfig(cov.gene ? `coverage_${cov.gene.name}` : `coverage_${cov.region.chrom}_${cov.region.start}`)}
            useResizeHandler
            style={{ width: "100%" }}
          />
          <Caption n={{ value: cov.n_fragments, unit: "fragments in the window" }}>
            {cov.region.chrom}:{cov.region.start.toLocaleString()}-{cov.region.end.toLocaleString()}, about{" "}
            {Math.round(cov.bin_size).toLocaleString()} bp per bin. {cov.normalisation}{" "}
            {cov.peaks.length > 0 ? `${cov.peaks.length} peaks marked. ` : "No peak table on this dataset. "}
            {cov.genes_note}
          </Caption>
          {cov.notes.map((n) => (
            <p key={n} className="text-[11px] leading-snug text-amber-700">{n}</p>
          ))}
        </figure>
      )}
    </section>
  );
}
