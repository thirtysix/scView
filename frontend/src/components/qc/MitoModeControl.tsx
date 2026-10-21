import { useEffect, useMemo, useState } from "react";
import { AlertTriangle } from "lucide-react";
import type { FilterThresholds } from "@/api/qc";
import type { MitoMode, MitoTypesTable } from "@/api/mitoFilter";
import { getMitoByType } from "@/api/mitoByType";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { useDatasetStore } from "@/stores/datasetStore";

/** As the backend says it (core/mito_by_type.py NOT_MEASURABLE): never "0 %". */
export const MITO_NOT_MEASURABLE =
  "No mitochondrial (MT-) genes in this dataset, so % mito can't be measured and no per-type mito bound can be set.";

const MAX_TYPES = 200; // backend qc_filter.MAX_TYPES
const TYPE_COLUMN = "scview_mito_type"; // kept by Reprocess from Scratch (backend mito_refilter.TYPE_COLUMN)

const MODE_LABELS: Record<MitoMode, string> = { global: "One rule", per_type: "Per cell type" };

const WHY: Record<string, string> = {
  too_few: "too few cells",
  no_spread: "no spread",
  damaged: "damaged",
  no_values: "no values",
};

/** The % mito params a dataset was filtered with, for the step's params: per-type
 *  bounds keep their type column; the global rule is the default, so nothing. */
export function recordedMitoParams(a: {
  qc_mito_mode?: string | null;
  qc_mito_type_column?: string | null;
}): Partial<FilterThresholds> {
  return a.qc_mito_mode === "per_type"
    ? { qc_mito_mode: "per_type", qc_mito_type_column: a.qc_mito_type_column ?? "" }
    : {};
}

function pct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || !isFinite(v) ? "—" : `${v.toFixed(digits)}%`;
}

/** Each type's rule and loss under both rules (the filter preview's per-type table). */
export function MitoTypesTableView({ table }: { table: MitoTypesTable }) {
  const lostGlobal = table.types.reduce((s, t) => s + (t.removed_global ?? 0), 0);
  const lost = table.types.reduce((s, t) => s + (t.removed ?? 0), 0);
  return (
    <div className="space-y-1" data-testid="mito-types-table">
      <div className="max-h-56 overflow-auto rounded border border-slate-100">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-slate-50 text-left text-slate-500">
            <tr>
              <th className="px-1.5 py-0.5 font-medium" title={`Groups of '${table.column}'`}>Type</th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Cells of the type with a % mito value">Cells</th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Median % of a cell's counts from MT- genes, over all the type's cells">Median</th>
              <th
                className="px-1.5 py-0.5 text-right font-medium"
                title={`The bound the type is judged by: its median + ${table.k} MADs (raw MAD, at most ${table.cap}%), or "global" when it has none (fewer than ${table.min_cells} cells, no spread, or a median of ${table.damaged_median}% or more)`}
              >
                Bound used
              </th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Cells the global % mito rule removes from the type (MAD mode: each sample's bound and the ceiling; fixed: the ceiling)">
                Global rule
              </th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Cells the per-type rule removes from the type for % mito">
                Per type
              </th>
            </tr>
          </thead>
          <tbody className="text-slate-600">
            {table.types.map((t) => {
              const more = (t.removed ?? 0) > (t.removed_global ?? 0);
              return (
                <tr key={t.name} className="border-t border-slate-100" data-testid={`mito-type-${t.name}`}>
                  <td className="max-w-[9rem] truncate px-1.5 py-0.5" title={[t.note, t.warning].filter(Boolean).join(" ") || t.name}>
                    {t.warning && <AlertTriangle className="mr-0.5 inline h-3 w-3 text-amber-500" aria-label="Warning" />}
                    {t.name}
                  </td>
                  <td className="px-1.5 py-0.5 text-right tabular-nums">{t.n_cells.toLocaleString()}</td>
                  <td className="px-1.5 py-0.5 text-right tabular-nums">{pct(t.median)}</td>
                  <td className="px-1.5 py-0.5 text-right tabular-nums" title={t.note || undefined}>
                    {t.bound !== null ? (
                      `${pct(t.bound)}${t.capped ? " (cap)" : ""}`
                    ) : (
                      <span className="text-slate-400">global{t.why ? ` (${WHY[t.why] ?? t.why})` : ""}</span>
                    )}
                  </td>
                  <td className="px-1.5 py-0.5 text-right tabular-nums">{(t.removed_global ?? 0).toLocaleString()}</td>
                  <td className={`px-1.5 py-0.5 text-right tabular-nums ${more ? "font-medium text-amber-700" : ""}`}>
                    {(t.removed ?? 0).toLocaleString()}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {table.warnings.length > 0 && (
        <ul className="space-y-0.5" aria-label="Per-type bound warnings">
          {table.warnings.map((w) => (
            <li key={w} className="flex items-start gap-1 rounded bg-amber-50 px-1.5 py-0.5 text-[11px] text-amber-800">
              <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
              <span>{w}</span>
            </li>
          ))}
        </ul>
      )}
      <Caption
        n={[
          { value: table.types.reduce((s, t) => s + t.n_cells, 0), unit: "cells" },
          { value: table.types.length, unit: `types of '${table.column}'` },
        ]}
      >
        Removed for % mito: {lostGlobal.toLocaleString()} by the global rule, {lost.toLocaleString()} per type.{" "}
        {table.n_own} type{table.n_own === 1 ? "" : "s"} have a bound of their own, {table.n_global} keep the global
        rule
        {table.n_unlabelled ? `, as do ${table.n_unlabelled.toLocaleString()} cells without a type` : ""}. A bound
        works both ways: a type with a low baseline is cut lower than the global rule would.
        {table.reused_from ? " The bounds are the last per-type filtering's, so filtering again removes no more." : ""}
      </Caption>
    </div>
  );
}

/**
 * The filtering step's % mito rule: one global rule (the default) or each cell
 * type's own bound (median + k MADs), and for per type the column of cell types.
 * Disabled with the reason when the dataset has no MT- genes. `table` (the filter
 * preview's) shows what per type does to each type.
 */
export function MitoModeControl({
  datasetId,
  values,
  onChange,
  hasMt,
  table,
}: {
  datasetId: string;
  values: FilterThresholds;
  onChange?: (patch: Partial<FilterThresholds>) => void;
  /** From the filter preview; undefined: asked of the backend once. */
  hasMt?: boolean;
  table?: MitoTypesTable | null;
}) {
  const obsColumns = useDatasetStore((s) => s.currentDataset?.obs_columns);
  const [asked, setAsked] = useState<{ id: string; measurable: boolean; column: string | null } | null>(null);
  useEffect(() => {
    if (hasMt !== undefined) return;
    let cancelled = false;
    getMitoByType(datasetId)
      .then((r) => !cancelled && setAsked({ id: datasetId, measurable: r.measurable, column: r.column }))
      .catch(() => {
        /* unknown: the control stays usable; the step says so if it can't measure */
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, hasMt]);
  const known = asked && asked.id === datasetId ? asked : null;
  const measurable = hasMt ?? known?.measurable ?? true;
  const mode: MitoMode = values.qc_mito_mode === "per_type" ? "per_type" : "global";
  const column = values.qc_mito_type_column ?? "";
  const detected = table?.column ?? known?.column ?? null;
  const options = useMemo(
    () =>
      (obsColumns ?? [])
        .filter((c) => (c.dtype === "category" || c.dtype === "object") && c.n_unique >= 2 && c.n_unique <= MAX_TYPES)
        .map((c) => c.name),
    [obsColumns],
  );
  const chosen = column || (mode === "per_type" ? detected : null) || "";
  const removedByReset = chosen.startsWith("scview_") && chosen !== TYPE_COLUMN;

  return (
    <div className="space-y-1 text-[11px] text-slate-600" data-testid="mito-mode-control">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-medium text-slate-500">% mito</span>
        <InfoTip label="About per-type % mito bounds" width={300}>
          One rule judges every cell by the same % mito threshold (MAD mode: each sample&apos;s median + MADs and
          the ceiling; fixed: the ceiling). Cell types differ in their healthy mitochondrial share, so one rule
          can remove most of a type with a high baseline. Per cell type, each type with enough cells gets its own
          bound, its median + {values.qc_mad_k_mt ?? 3} MADs (at most 50%); small, flat or damaged types keep the one
          rule. Counts and genes are judged as before.
        </InfoTip>
        <div role="group" aria-label="% mito rule" className="inline-flex rounded-md border border-slate-200 p-0.5">
          {(["global", "per_type"] as const).map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={mode === m}
              disabled={!onChange || !measurable}
              onClick={() => onChange?.({ qc_mito_mode: m })}
              className={`rounded px-2 py-0.5 text-[11px] font-medium disabled:cursor-not-allowed disabled:opacity-50 ${
                mode === m ? "bg-primary text-white" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {MODE_LABELS[m]}
            </button>
          ))}
        </div>
        {mode === "per_type" && measurable && (
          <select
            aria-label="Cell types for per-type % mito bounds"
            value={column}
            disabled={!onChange}
            onChange={(e) => onChange?.({ qc_mito_type_column: e.target.value })}
            className="min-w-0 max-w-[55%] rounded border border-slate-200 bg-white px-1.5 py-0.5 text-[11px] text-slate-600"
          >
            <option value="">{detected ? `Auto: ${detected}` : "Auto (author cell types, else the clustering)"}</option>
            {options.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        )}
      </div>
      {!measurable && <p className="text-slate-500">{MITO_NOT_MEASURABLE}</p>}
      {measurable && mode === "per_type" && removedByReset && (
        <p className="text-amber-700">
          Reprocess from Scratch removes &apos;{chosen}&apos; (a scView clustering) before filtering. To filter by
          it, use Re-filter on the Mito by cell type card, which starts again from the original upload.
        </p>
      )}
      {measurable && mode === "per_type" && table && <MitoTypesTableView table={table} />}
    </div>
  );
}
