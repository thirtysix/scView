import { useEffect, useMemo, useState } from "react";
import { AlertTriangle } from "lucide-react";
import type { FilterThresholds } from "@/api/qc";
import type { MitoMode, MitoTypeRule, MitoTypesTable } from "@/api/mitoFilter";
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
 *  bounds keep their type column (and the floor, when it was on); the global rule is
 *  the default, so nothing. */
export function recordedMitoParams(a: {
  qc_mito_mode?: string | null;
  qc_mito_type_column?: string | null;
  qc_mito_type_floor?: boolean | null;
}): Partial<FilterThresholds> {
  if (a.qc_mito_mode !== "per_type") return {};
  return {
    qc_mito_mode: "per_type",
    qc_mito_type_column: a.qc_mito_type_column ?? "",
    ...(a.qc_mito_type_floor ? { qc_mito_type_floor: true } : {}),
  };
}

/** What the floor does, in the words the checkbox's tooltip and the table's caption use. */
export function floorText(mad: boolean): string {
  return (
    "Never stricter than the global rule: a type's bound is the higher of its own and the global rule's" +
    (mad
      ? " for each cell (MAD mode: the bound of the cell's own sample, under the ceiling, so a type can keep" +
        " its own bound in one sample and take the global one in another)"
      : " (the ceiling)") +
    ". A type with a high baseline still gets a looser bound; one with a low baseline is cut no harder than" +
    " the global rule would."
  );
}

/** Which bound applied to a type with a bound of its own, under the floor. */
function boundUsed(t: MitoTypeRule, floor: boolean): { text: string; title?: string } {
  const own = `${pct(t.bound)}${t.capped ? " (cap)" : ""}`;
  if (!floor || !t.floor_uses || t.floor_uses === "type") {
    return {
      text: own,
      title: floor && t.floor_uses === "type" ? `Its own bound, above the global rule's (${range(t.global_bound)})` : t.note || undefined,
    };
  }
  const g = range(t.global_bound);
  if (t.floor_uses === "global")
    return { text: "global", title: `Its own bound, ${own}, is below the global rule's (${g}), so the global rule applies` };
  return {
    text: `${own} / global`,
    title: `Its own bound, ${own}, for ${(t.n_cells - (t.n_floored ?? 0)).toLocaleString()} cells; the global rule's (${g}, higher) for ${(t.n_floored ?? 0).toLocaleString()} cells in samples whose bound is above it`,
  };
}

function range(r: [number, number] | null | undefined): string {
  if (!r) return "no global % mito bound";
  return r[0] === r[1] ? pct(r[0]) : `${pct(r[0])}–${pct(r[1])}`;
}

function pct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || !isFinite(v) ? "—" : `${v.toFixed(digits)}%`;
}

/** Each type's rule and loss under both rules (the filter preview's per-type table), and
 *  with the floor off vs on when the backend gives both. `mad`: the global rule is MAD mode's. */
export function MitoTypesTableView({ table, mad = false }: { table: MitoTypesTable; mad?: boolean }) {
  const lostGlobal = table.types.reduce((s, t) => s + (t.removed_global ?? 0), 0);
  const lost = table.types.reduce((s, t) => s + (t.removed_type ?? t.removed ?? 0), 0);
  const floor = Boolean(table.floor);
  // Both per-type columns once the backend says what the floor removes (older ones don't).
  const both = table.types.some((t) => t.removed_floor !== undefined);
  const lostFloor = table.types.reduce((s, t) => s + (t.removed_floor ?? 0), 0);
  const applied = (on: boolean) => (both && floor === on ? "font-semibold" : "");
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
                title={
                  `The bound the type is judged by: its median + ${table.k} MADs (raw MAD, at most ${table.cap}%), or "global" when it has none (fewer than ${table.min_cells} cells, no spread, or a median of ${table.damaged_median}% or more)` +
                  (floor ? `. Never stricter than the global rule: "global" where the global rule's bound is higher, "own / global" where that differs by sample` : "")
                }
              >
                Bound used
              </th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Cells the global % mito rule removes from the type (MAD mode: each sample's bound and the ceiling; fixed: the ceiling)">
                Global rule
              </th>
              <th
                className={`px-1.5 py-0.5 text-right font-medium ${applied(false)}`}
                title={`Cells the per-type rule removes from the type for % mito${both ? ": its own bound, also where that is below the global rule's" : ""}`}
              >
                Per type
              </th>
              {both && (
                <th
                  className={`px-1.5 py-0.5 text-right font-medium ${applied(true)}`}
                  title="Cells the per-type rule removes with the floor: only those above both its own bound and the global rule's for that cell (never more than the global rule)"
                >
                  Never stricter
                </th>
              )}
            </tr>
          </thead>
          <tbody className="text-slate-600">
            {table.types.map((t) => {
              const perType = t.removed_type ?? t.removed ?? 0;
              const more = perType > (t.removed_global ?? 0);
              const used = boundUsed(t, floor);
              return (
                <tr key={t.name} className="border-t border-slate-100" data-testid={`mito-type-${t.name}`}>
                  <td className="max-w-[9rem] truncate px-1.5 py-0.5" title={[t.note, t.warning].filter(Boolean).join(" ") || t.name}>
                    {t.warning && <AlertTriangle className="mr-0.5 inline h-3 w-3 text-amber-500" aria-label="Warning" />}
                    {t.name}
                  </td>
                  <td className="px-1.5 py-0.5 text-right tabular-nums">{t.n_cells.toLocaleString()}</td>
                  <td className="px-1.5 py-0.5 text-right tabular-nums">{pct(t.median)}</td>
                  <td className="px-1.5 py-0.5 text-right tabular-nums" title={t.bound !== null ? used.title : t.note || undefined}>
                    {t.bound !== null ? (
                      used.text === "global" ? <span className="text-slate-400">global</span> : used.text
                    ) : (
                      <span className="text-slate-400">global{t.why ? ` (${WHY[t.why] ?? t.why})` : ""}</span>
                    )}
                  </td>
                  <td className="px-1.5 py-0.5 text-right tabular-nums">{(t.removed_global ?? 0).toLocaleString()}</td>
                  <td className={`px-1.5 py-0.5 text-right tabular-nums ${more ? "font-medium text-amber-700" : ""} ${applied(false)}`}>
                    {perType.toLocaleString()}
                  </td>
                  {both && (
                    <td className={`px-1.5 py-0.5 text-right tabular-nums ${applied(true)}`}>
                      {(t.removed_floor ?? 0).toLocaleString()}
                    </td>
                  )}
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
        Removed for % mito: {lostGlobal.toLocaleString()} by the global rule, {lost.toLocaleString()} per type
        {both ? `, ${lostFloor.toLocaleString()} per type never stricter than the global rule (${floor ? "applied" : "not applied"})` : ""}.{" "}
        {table.n_own} type{table.n_own === 1 ? "" : "s"} have a bound of their own, {table.n_global} keep the global
        rule
        {table.n_unlabelled ? `, as do ${table.n_unlabelled.toLocaleString()} cells without a type` : ""}.{" "}
        {floor
          ? floorText(mad)
          : "A bound works both ways: a type with a low baseline is cut lower than the global rule would."}
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
  const floor = Boolean(values.qc_mito_type_floor);
  const mad = values.qc_threshold_mode !== "fixed";
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
      {mode === "per_type" && measurable && (
        <div className="flex items-center gap-1.5 pl-1">
          <label className="inline-flex items-center gap-1.5 text-slate-600">
            <input
              type="checkbox"
              checked={floor}
              disabled={!onChange}
              onChange={(e) => onChange?.({ qc_mito_type_floor: e.target.checked })}
              className="h-3 w-3 rounded border-slate-300"
            />
            Never stricter than the global rule
          </label>
          <InfoTip label="About never stricter than the global rule" width={300}>
            {floorText(mad)} Off by default: per type, a type&apos;s own bound is used even where it is below the
            global rule&apos;s.
          </InfoTip>
        </div>
      )}
      {!measurable && <p className="text-slate-500">{MITO_NOT_MEASURABLE}</p>}
      {measurable && mode === "per_type" && removedByReset && (
        <p className="text-amber-700">
          Reprocess from Scratch removes &apos;{chosen}&apos; (a scView clustering) before filtering. To filter by
          it, use Re-filter on the Mito by cell type card, which starts again from the original upload.
        </p>
      )}
      {measurable && mode === "per_type" && table && <MitoTypesTableView table={table} mad={mad} />}
    </div>
  );
}
