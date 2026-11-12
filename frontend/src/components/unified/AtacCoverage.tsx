import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { getCoverage, getFragments, type Coverage } from "@/api/atacFragments";
import type { LinksLocus } from "@/api/atacLinks";
import { Caption } from "@/components/common/Caption";
import { Help } from "@/components/common/Help";
import { HoverTip } from "@/components/common/InfoTip";
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

/** A locus of the peak-gene links in the shape the coverage figure draws: no tracks. */
function locusAsCoverage(locus: LinksLocus): Coverage {
  return {
    region: { chrom: locus.chrom, start: locus.start, end: locus.end },
    bins: 0,
    bin_size: 0,
    x: [],
    group_by: null,
    tracks: [],
    ymax: 0,
    normalisation: "",
    notes: [],
    n_fragments: 0,
    max_window: 0,
    peaks: locus.peaks,
    genes: locus.genes.map((g) => ({ ...g, id: g.name, type: "protein_coding" })),
    gene: null,
    genes_note: null,
    seconds: 0,
  };
}

/** A region typed as chr1:1000-5000 (commas fine), else a gene. */
export function asRequest(text: string): { gene?: string; region?: string } {
  const t = text.trim();
  return /^[^:\s]+:[\d,]+-[\d,]+$/.test(t) ? { region: t } : { gene: t };
}

/**
 * Coverage tracks from the ATAC fragments file (backend core/atac_fragments.py): one filled
 * track per group for a gene (its body +- 5 kb) or a region of up to 2 Mb, peak ticks and gene
 * arrows beneath. Renders nothing until a fragments file and its index are attached, or until a
 * row of the peak-gene links opens a ``locus``: its arcs go under the tracks, or on a peak track
 * alone without a fragments file.
 */
export function AtacCoverage({
  datasetId,
  locus = null,
  selectedPeak = null,
}: {
  datasetId: string;
  locus?: LinksLocus | null;
  selectedPeak?: string | null;
}) {
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
  const arcs = useMemo(() => {
    if (!locus || !cov || cov.region.chrom !== locus.chrom) return null;
    return { gene: locus.gene, arcs: locus.arcs, selectedPeak };
  }, [locus, cov, selectedPeak]);
  const figure = useMemo(() => {
    if (!cov) return null;
    return coverageFigure(cov, (g) => (cov.group_by ? rgbCss(colors(g)) : "#2563eb"), arcs);
  }, [cov, colors, arcs]);
  // Without a fragments file: the same figure with no tracks, peaks and genes from the links' own locus.
  const peakFigure = useMemo(() => {
    if (attached || !locus) return null;
    return coverageFigure(locusAsCoverage(locus), () => "#2563eb", { gene: locus.gene, arcs: locus.arcs, selectedPeak });
  }, [attached, locus, selectedPeak]);

  const draw = async (asked: string = text, column: string = groupBy) => {
    if (!asked.trim()) {
      setError("Type a gene or a region like chr1:1000-5000.");
      return;
    }
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setBusy(true);
    setError(null);
    try {
      setCov(await getCoverage(datasetId, { ...asRequest(asked), groupBy: column || null }, ctl.signal));
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setCov(null);
        setError(errorMessage(e));
      }
    } finally {
      if (abort.current === ctl) setBusy(false);
    }
  };

  // A link's row was clicked: draw its window with the tracks, once per locus opened.
  const drawRef = useRef(draw);
  drawRef.current = draw;
  useEffect(() => {
    if (!locus || !attached) return;
    const region = `${locus.chrom}:${locus.start}-${locus.end}`;
    setText(region);
    void drawRef.current(region);
  }, [locus, attached]);

  if (!attached) {
    if (!locus || !peakFigure) return null;
    return (
      <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Links at this locus">
        <h4 className="text-xs font-semibold text-slate-600">{locus.gene}: links at this locus</h4>
        <figure className="space-y-1" data-testid="atac-links-locus">
          <Plot
            data={peakFigure.data}
            layout={peakFigure.layout}
            config={plotConfig(`links_${locus.gene}`)}
            useResizeHandler
            style={{ width: "100%" }}
          />
          <Caption n={{ value: locus.arcs.length, unit: "links" }}>
            {locus.chrom}:{locus.start.toLocaleString()}-{locus.end.toLocaleString()}. Each arc joins a peak&rsquo;s middle to the
            gene&rsquo;s TSS: a correlation, a candidate and never proof of regulation. Attach a fragments file under Fragment QC to
            draw coverage tracks above these.
          </Caption>
          {locus.note && <p className="text-[11px] leading-snug text-amber-700">{locus.note}</p>}
        </figure>
      </section>
    );
  }
  const nGroups = columns.find((c) => c.name === groupBy)?.n_unique ?? 1;

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Coverage tracks">
      <h4 className="text-xs font-semibold text-slate-600">
        <HoverTip
          width={320}
          tip={<>
            How many fragments cover each base, per group of cells, drawn from the fragments file for the window you ask
            for. A peak in one group&rsquo;s track only is accessible chromatin in those cells. A gene shows its body{" "}
            {(genePad / 1000).toFixed(0)} kb either side, so the promoter is in view.
          </>}
        >
          Coverage tracks
        </HoverTip>
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
          <Help id="atac.coverage-group">Group by</Help>
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
            <Help id="atac.coverage-tracks">
              {cov.region.chrom}:{cov.region.start.toLocaleString()}-{cov.region.end.toLocaleString()}, about{" "}
              {Math.round(cov.bin_size).toLocaleString()} bp per bin.
            </Help>{" "}
            {cov.normalisation}{" "}
            {cov.peaks.length > 0 ? `${cov.peaks.length} peaks marked. ` : "No peak table on this dataset. "}
            {cov.genes_note}
            {arcs && " Arcs: a peak's correlation with the gene, a candidate and never proof of regulation."}
          </Caption>
          {cov.notes.map((n) => (
            <p key={n} className="text-[11px] leading-snug text-amber-700">{n}</p>
          ))}
        </figure>
      )}
    </section>
  );
}
