import type { AtacCriterion, AtacQcSuggestion, FilterPreview, FilterThresholds } from "@/api/qc";

type AtacField =
  | "max_counts"
  | "qc_min_atac_counts"
  | "qc_max_atac_counts"
  | "qc_max_nucleosome_signal"
  | "qc_min_tss_enrichment";

/** The filtering step's 10x Multiome cut-offs (backend qc_filter.ATAC_CRITERIA + max counts). */
export const ATAC_FIELDS: { field: AtacField; criterion: AtacCriterion; label: string; step: number }[] = [
  { field: "max_counts", criterion: "max_counts", label: "Max RNA counts", step: 1000 },
  { field: "qc_min_atac_counts", criterion: "min_atac_counts", label: "Min ATAC counts", step: 100 },
  { field: "qc_max_atac_counts", criterion: "max_atac_counts", label: "Max ATAC counts", step: 1000 },
  { field: "qc_max_nucleosome_signal", criterion: "max_nucleosome_signal", label: "Max nucleosome signal", step: 0.1 },
  { field: "qc_min_tss_enrichment", criterion: "min_tss_enrichment", label: "Min TSS enrichment", step: 0.1 },
];

/** The thresholds patch that applies the suggested (Signac vignette) cuts, or turns them all off. */
export function atacPatch(atac: AtacQcSuggestion, use: boolean): Partial<FilterThresholds> {
  const patch: Partial<FilterThresholds> = {};
  for (const f of ATAC_FIELDS) {
    const v = atac.suggested[f.criterion];
    if (v !== undefined) patch[f.field] = use ? v : 0;
  }
  return patch;
}

/**
 * Under the cell-loss summary, for data with ATAC QC columns (10x Multiome): Signac's
 * standard cut-offs on ATAC counts, nucleosome signal and TSS enrichment, plus an RNA counts
 * ceiling. Off until set; "Use Signac's cuts" fills in the vignette's values.
 */
export function AtacFilterControl({
  preview,
  thresholds,
  onChange,
}: {
  preview: FilterPreview;
  thresholds: FilterThresholds;
  onChange: (patch: Partial<FilterThresholds>) => void;
}) {
  const atac = preview.atac_qc;
  if (!atac) return null;
  const fields = ATAC_FIELDS.filter((f) => atac.suggested[f.criterion] !== undefined);
  const anyOn = fields.some((f) => Number(thresholds[f.field] ?? 0) > 0);
  return (
    <div className="space-y-1 rounded-md border border-slate-200 px-2 py-1.5" data-testid="atac-filter">
      <p className="text-[11px] font-medium text-slate-600">ATAC QC (10x Multiome), 0 = off</p>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 sm:grid-cols-3">
        {fields.map((f) => {
          const removed = preview.by_criterion[f.criterion]?.n_removed;
          return (
            <label key={f.field} className="flex flex-col text-[11px] text-slate-600">
              <span>{f.label}</span>
              <input
                type="number"
                min={0}
                step={f.step}
                aria-label={f.label}
                value={Number(thresholds[f.field] ?? 0)}
                onChange={(e) => onChange({ [f.field]: Math.max(0, parseFloat(e.target.value) || 0) } as Partial<FilterThresholds>)}
                className="w-full rounded border border-slate-200 px-1.5 py-0.5 text-[11px] text-slate-700"
              />
              {removed !== undefined && (
                <span className="text-slate-400">removes {removed.toLocaleString()} on its own</span>
              )}
            </label>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <button
          type="button"
          onClick={() => onChange(atacPatch(atac, true))}
          className="rounded border border-slate-200 bg-white px-2 py-0.5 text-slate-700 hover:bg-slate-50"
        >
          Use Signac&apos;s cuts
        </button>
        {anyOn && (
          <button type="button" onClick={() => onChange(atacPatch(atac, false))} className="text-primary hover:underline">
            Turn them off
          </button>
        )}
        <span className="text-slate-500" data-testid="atac-suggestion">
          Together they would remove {atac.n_removed.toLocaleString()} of {preview.n_cells.toLocaleString()} cells.
        </span>
      </div>
      <p className="text-[11px] text-slate-500">{atac.source}</p>
      {atac.missing_note && <p className="text-[11px] text-slate-400">{atac.missing_note}</p>}
    </div>
  );
}
