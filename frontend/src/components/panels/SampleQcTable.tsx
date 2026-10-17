import { useState } from "react";
import type { FilterGroupLoss, FilterPreview, MadThresholds, SampleFlag } from "@/api/qc";
import { bound, pct } from "@/lib/qcThresholds";

export const SAMPLE_FLAG_LABELS: Record<SampleFlag, string> = {
  few_cells: "Few cells",
  low_genes: "Few genes",
  high_genes: "Many genes",
  low_counts: "Low UMIs",
  high_counts: "High UMIs",
  high_mito: "High mito",
};

/** One sequential hue per column (Tailwind 600s), light to dark with the value. */
const HUES = {
  cells: [71, 85, 105], // slate
  genes: [37, 99, 235], // blue
  counts: [124, 58, 237], // violet
  pct_mt: [217, 119, 6], // amber
  removed: [220, 38, 38], // red
} as const;
type HeatColumn = keyof typeof HUES;
const MAX_ALPHA = 0.45; // dark ink stays readable on the darkest cell

/** A cell's shade: 0 to the column's highest value, so near-equal values look
 *  near-equal (the flags, not the shading, say which sample stands out). */
export function heat(column: HeatColumn, value: number | null | undefined, max: number): string | undefined {
  if (value === null || value === undefined || !isFinite(value) || !(max > 0)) return undefined;
  const [r, g, b] = HUES[column];
  const a = 0.04 + (MAX_ALPHA - 0.04) * Math.min(1, Math.max(0, value / max));
  return `rgba(${r}, ${g}, ${b}, ${a.toFixed(3)})`;
}

function columnMax(rows: FilterGroupLoss[], pick: (g: FilterGroupLoss) => number | null | undefined): number {
  return rows.reduce((m, g) => {
    const v = pick(g);
    return v !== null && v !== undefined && isFinite(v) ? Math.max(m, v) : m;
  }, 0);
}

/** A count range a sample keeps: "312–9,870", or "≥ 312" with no upper bound (the
 *  default: only the low tail is cut); a side without a bound reads "any". */
function keptRange(lo: number | null, hi: number | null): string {
  if (lo === null && hi === null) return "any";
  if (hi === null) return `≥ ${bound(lo)}`;
  return `${lo === null ? "any" : bound(lo)}–${bound(hi)}`;
}

/** MAD mode: each sample's bounds and what they remove, in the samples' own order. */
export function SampleThresholds({ mad }: { mad: MadThresholds }) {
  if (!mad.column) {
    const p = mad.pooled;
    return (
      <p className="text-[11px] text-slate-600" data-testid="mad-pooled">
        No sample column found, so all cells are one group: kept with {keptRange(p.min_counts, p.max_counts)} counts
        and {keptRange(p.min_genes, p.max_genes)} genes per cell
        {p.max_pct_mt !== null ? `, under ${bound(p.max_pct_mt, 1)}% mito` : ""}.
      </p>
    );
  }
  const hasMt = mad.samples.some((s) => s.max_pct_mt !== null);
  const skipped = (s: MadThresholds["samples"][number], metric: string) =>
    s.skipped.includes(metric) ? "Its MAD is 0 (most cells alike): no bound from it here." : undefined;
  return (
    <div>
      <table className="w-full text-[11px]" aria-label="Thresholds per sample">
        <thead className="sticky top-0 bg-white">
          <tr className="text-left text-slate-400">
            <th className="font-medium">{mad.column}</th>
            <th className="text-right font-medium">Cells</th>
            <th className="text-right font-medium">Counts kept</th>
            <th className="text-right font-medium">Genes kept</th>
            {hasMt && <th className="text-right font-medium">Max % mito</th>}
            <th className="text-right font-medium">Removed</th>
          </tr>
        </thead>
        <tbody className="text-slate-600">
          {mad.samples.map((s) => (
            <tr key={s.name}>
              <td
                className="max-w-[8rem] truncate pr-2"
                title={
                  s.pooled
                    ? `${s.name}: fewer than ${mad.min_sample_cells} cells, so judged against all cells pooled`
                    : s.name
                }
              >
                {s.name}
                {s.pooled && <span className="text-slate-400"> (pooled)</span>}
              </td>
              <td className="text-right">{s.n_cells.toLocaleString()}</td>
              <td className="text-right" title={skipped(s, "total_counts")}>
                {keptRange(s.min_counts, s.max_counts)}
              </td>
              <td className="text-right" title={skipped(s, "n_genes")}>
                {keptRange(s.min_genes, s.max_genes)}
              </td>
              {hasMt && (
                <td className="text-right" title={skipped(s, "pct_mt")}>
                  {s.max_pct_mt === null ? "any" : `< ${bound(s.max_pct_mt, 1)}`}
                </td>
              )}
              <td className="text-right">
                {s.n_removed === undefined
                  ? "—"
                  : `${s.n_removed.toLocaleString()} (${pct(s.n_cells ? s.n_removed / s.n_cells : 0)})`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {mad.n_unlabelled > 0 && (
        <p className="text-[11px] text-slate-400">
          {mad.n_unlabelled.toLocaleString()} cells without a {mad.column} value are judged against all cells pooled.
        </p>
      )}
    </div>
  );
}

/** Each sample's cells and median genes, UMIs and % mito, shaded per column, with
 *  what the current thresholds remove from it and flags for a sample whose median
 *  is far from the others' (or that has few cells). */
function SampleMedians({ grouping, hasMt }: { grouping: NonNullable<FilterPreview["samples"]>; hasMt: boolean }) {
  const rows = grouping.groups;
  const max = {
    cells: columnMax(rows, (g) => g.n_cells),
    genes: columnMax(rows, (g) => g.median_genes),
    counts: columnMax(rows, (g) => g.median_counts),
    pct_mt: columnMax(rows, (g) => g.median_pct_mt),
    removed: columnMax(rows, (g) => g.fraction),
  };
  const mito = hasMt && rows.some((g) => g.median_pct_mt !== undefined && g.median_pct_mt !== null);
  const t = grouping.typical ?? {};
  const alone = (g: FilterGroupLoss) =>
    Object.entries(g.by_criterion)
      .filter(([, n]) => n && n > 0)
      .map(([c, n]) => `${c.replace(/_/g, " ")} ${n!.toLocaleString()}`)
      .join(", ");
  return (
    <table className="w-full text-[11px]" aria-label="QC per sample">
      <thead className="sticky top-0 bg-white">
        <tr className="text-left text-slate-400">
          <th className="font-medium">{grouping.column}</th>
          <th className="px-1 text-right font-medium">Cells</th>
          <th className="px-1 text-right font-medium" title="Median genes detected per cell">
            Genes
          </th>
          <th className="px-1 text-right font-medium" title="Median counts (UMIs) per cell">
            UMIs
          </th>
          {mito && (
            <th className="px-1 text-right font-medium" title="Median % mitochondrial counts per cell">
              % mito
            </th>
          )}
          <th className="px-1 text-right font-medium" title="Cells the current thresholds remove">
            Removed
          </th>
          <th className="pl-1 font-medium">Flags</th>
        </tr>
      </thead>
      <tbody className="text-slate-700">
        {rows.map((g) => (
          <tr key={g.name} className="border-t border-white">
            <td className="max-w-[8rem] truncate pr-2" title={g.name}>
              {g.name}
            </td>
            <td className="px-1 text-right" style={{ backgroundColor: heat("cells", g.n_cells, max.cells) }}>
              {g.n_cells.toLocaleString()}
            </td>
            <td className="px-1 text-right" style={{ backgroundColor: heat("genes", g.median_genes, max.genes) }}>
              {bound(g.median_genes)}
            </td>
            <td className="px-1 text-right" style={{ backgroundColor: heat("counts", g.median_counts, max.counts) }}>
              {bound(g.median_counts)}
            </td>
            {mito && (
              <td className="px-1 text-right" style={{ backgroundColor: heat("pct_mt", g.median_pct_mt, max.pct_mt) }}>
                {bound(g.median_pct_mt, 1)}
              </td>
            )}
            <td
              className="px-1 text-right"
              style={{ backgroundColor: heat("removed", g.fraction, max.removed) }}
              title={alone(g) ? `Each threshold on its own: ${alone(g)}` : undefined}
            >
              {g.n_removed.toLocaleString()} ({pct(g.fraction)})
            </td>
            <td className="pl-1">
              <span className="flex flex-wrap gap-0.5">
                {(g.flags ?? []).map((f, i) => (
                  <span
                    key={f}
                    title={g.reasons?.[i] ?? grouping.flag_help?.[f]}
                    className="rounded border border-amber-200 bg-amber-50 px-1 text-[10px] text-amber-800"
                  >
                    {SAMPLE_FLAG_LABELS[f] ?? f}
                  </span>
                ))}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
      <tfoot>
        <tr className="text-slate-400">
          <td className="pr-2">Typical</td>
          <td />
          <td className="px-1 text-right">{bound(t.median_genes)}</td>
          <td className="px-1 text-right">{bound(t.median_counts)}</td>
          {mito && <td className="px-1 text-right">{bound(t.median_pct_mt, 1)}</td>}
          <td />
          <td />
        </tr>
      </tfoot>
    </table>
  );
}

/**
 * Per-sample QC beside the cell-loss summary, from the same filter preview (so the
 * samples are the ones the thresholds are worked out on): each sample's medians,
 * shaded per column, what the current thresholds remove from it, and range flags.
 * In MAD mode a second view lists each sample's own thresholds.
 */
export function SampleQcTable({ preview, mad }: { preview: FilterPreview; mad: MadThresholds | null }) {
  const [view, setView] = useState<"medians" | "thresholds">("medians");
  const grouping = preview.samples;
  const shown = mad ? view : "medians";
  const flagged = grouping?.groups.filter((g) => (g.flags ?? []).length > 0) ?? [];
  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3" data-testid="sample-qc">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-xs font-semibold text-slate-700">
          QC per sample
          {grouping && (
            <span className={`ml-2 font-normal ${flagged.length ? "text-amber-700" : "text-slate-500"}`}>
              {flagged.length
                ? `${flagged.length} of ${grouping.groups.length} samples stand out`
                : grouping.compared
                  ? `No sample stands out (${grouping.groups.length} samples)`
                  : `${grouping.groups.length} samples: too few to say which stands out`}
            </span>
          )}
        </h4>
        {mad?.column && (
          <div role="group" aria-label="Per-sample view" className="inline-flex rounded-md border border-slate-200 p-0.5">
            {(["medians", "thresholds"] as const).map((v) => (
              <button
                key={v}
                type="button"
                aria-pressed={shown === v}
                onClick={() => setView(v)}
                className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                  shown === v ? "bg-slate-700 text-white" : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {v === "medians" ? "QC medians" : "Thresholds"}
              </button>
            ))}
          </div>
        )}
      </div>
      {shown === "thresholds" && mad ? (
        <div className="max-h-72 overflow-auto">
          <SampleThresholds mad={mad} />
        </div>
      ) : !grouping ? (
        mad ? (
          <SampleThresholds mad={mad} />
        ) : (
          <p className="text-[11px] text-slate-500">No sample, donor or batch column found to compare samples by.</p>
        )
      ) : grouping.note ? (
        <p className="text-[11px] text-slate-400">{grouping.note}</p>
      ) : (
        <>
          <div className="max-h-72 overflow-auto">
            <SampleMedians grouping={grouping} hasMt={preview.has_mt} />
          </div>
          <p className="text-[10px] text-slate-400">
            Medians per cell. Shading runs from 0 to the column&apos;s highest value; a flag marks a median over 3
            MADs from the median of all samples (or a sample with few cells). Hover a flag for its numbers.
          </p>
        </>
      )}
    </div>
  );
}
