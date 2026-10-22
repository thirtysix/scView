import { useMemo, useState } from "react";
import { AlertTriangle } from "lucide-react";
import type {
  FilterCriterion,
  FilterPreview,
  RemovalAudit as Audit,
  RemovalAuditColumns,
  RemovalRow,
  RemovalTest,
} from "@/api/qc";
import { useDatasetStore } from "@/stores/datasetStore";
import { qcGroupOptions } from "@/lib/qcGroups";
import { pct } from "@/lib/qcThresholds";
import { formatDate } from "@/lib/formatting";
import { Caption, capText } from "@/components/common/Caption";

const ROWS = 8; // flagged and most affected first, more behind "show all"
const FLAGS = 4;
const MAX_LABEL_GROUPS = 100; // backend qc_filter.MAX_AUDIT_GROUPS
const MAX_CONDITIONS = 20; // backend qc_filter.MAX_AUDIT_CONDITIONS

/** Backend qc_filter.CRITERIA. */
export const CRITERION_LABELS: Record<FilterCriterion, string> = {
  min_genes: "Min genes",
  max_genes: "Max genes",
  min_counts: "Min counts",
  max_counts: "Max counts",
  max_pct_mt: "Max % mito",
  doublets: "Predicted doublets",
  hash_negatives: "Hashing negatives",
};

/** A chi-square p for a table cell: "2e-21", "0.034"; "–" when there is none. */
export function formatP(p: number | null | undefined): string {
  if (p === null || p === undefined || !isFinite(p)) return "–";
  if (p < 1e-300) return "< 1e-300";
  return p < 0.001 ? p.toExponential(0) : p.toPrecision(2);
}

/** "Max % mito 250, Min genes 3": what each threshold removes on its own. */
export function criteriaText(by: Partial<Record<FilterCriterion, number>>): string {
  const parts = Object.entries(by)
    .filter(([, v]) => (v ?? 0) > 0)
    .map(([c, v]) => `${CRITERION_LABELS[c as FilterCriterion] ?? c} ${v!.toLocaleString()}`);
  return parts.length ? `Each threshold on its own: ${parts.join(", ")}.` : "No threshold removes any of these cells.";
}

/** The p cell's tooltip: the test, its numbers, and when it is approximate or absent. */
export function testTitle(test: RemovalTest | null | undefined, condition: string): string {
  if (!test) return `Fewer than two ${condition} groups have 10 or more cells here: not tested.`;
  if (test.p === null) return "Nothing (or everything) in this group is removed: there is nothing to test.";
  const yates = test.dof === 1 ? ", Yates-corrected" : "";
  return (
    `Chi-square test of removed vs kept across ${test.compared.join(", ")} ` +
    `(chi-square ${test.statistic}, df ${test.dof}${yates}): p = ${formatP(test.p)}. ` +
    `Removal rate ${pct(test.high_rate)} in ${test.high} vs ${pct(test.low_rate)} in ${test.low}.` +
    (test.low_expected ? " Some expected counts are below 5, so the p-value is approximate." : "")
  );
}

function reasonText(reason: string | null | undefined): string {
  switch (reason) {
    case "labels":
      return "the author labels";
    case "clustering":
      return "the clustering";
    case "condition":
      return "a condition column";
    case "sample":
      return "the samples (no condition column found)";
    default:
      return "";
  }
}

/** "65.0% 650/1,000" with the per-criterion counts on hover. */
function LossCell({
  r,
  n,
  by,
  strong,
}: {
  r: number;
  n: number;
  by: Partial<Record<FilterCriterion, number>>;
  strong?: boolean;
}) {
  if (!n) return <td className="text-right text-slate-300">–</td>;
  const f = r / n;
  return (
    <td className="whitespace-nowrap pl-2 text-right" title={`${r.toLocaleString()} of ${n.toLocaleString()} cells removed. ${criteriaText(by)}`}>
      <span className={strong || f > 0.3 ? "font-semibold text-amber-800" : ""}>{pct(f)}</span>{" "}
      <span className="text-slate-400">
        {r.toLocaleString()}/{n.toLocaleString()}
      </span>
    </td>
  );
}

function conditionBy(row: RemovalRow, i: number): Partial<Record<FilterCriterion, number>> {
  const out: Partial<Record<FilterCriterion, number>> = {};
  for (const [c, v] of Object.entries(row.conditions?.by_criterion ?? {})) out[c as FilterCriterion] = v?.[i] ?? 0;
  return out;
}

function medianText(row: RemovalRow): string {
  const m = row.removed_median;
  if (!m) return "";
  const parts = [
    m.total_counts != null ? `${Math.round(m.total_counts).toLocaleString()} counts` : "",
    m.n_genes != null ? `${Math.round(m.n_genes).toLocaleString()} genes` : "",
    m.pct_mt != null ? `${m.pct_mt.toFixed(1)}% mito` : "",
  ].filter(Boolean);
  return parts.length ? ` The removed cells' medians: ${parts.join(", ")}.` : "";
}

function AuditTable({ audit, flagged }: { audit: Audit; flagged: Map<string | null, "loss" | "condition"> }) {
  const [all, setAll] = useState(false);
  const cond = audit.condition;
  const names = cond?.names ?? [];
  const hasTest = audit.by_type.some((r) => r.test) || Boolean(audit.overall.test);
  const rows = useMemo(
    () =>
      [...audit.by_type].sort(
        (a, b) => Number(flagged.has(b.name)) - Number(flagged.has(a.name)) || b.fraction - a.fraction,
      ),
    [audit.by_type, flagged],
  );
  const shown = all ? rows : rows.slice(0, ROWS);
  const label = audit.label.column ?? "Group";
  const row = (r: RemovalRow, key: string, name: string, title: string) => {
    const flag = flagged.get(r.name);
    return (
      <tr key={key} className={flag ? "bg-amber-50" : ""} data-flagged={flag ?? undefined}>
        <td className="max-w-[10rem] truncate pr-2 font-medium" title={title}>
          {name}
        </td>
        <LossCell r={r.n_removed} n={r.n_cells} by={r.by_criterion} />
        {names.map((c, i) => (
          <LossCell
            key={c}
            r={r.conditions?.n_removed[i] ?? 0}
            n={r.conditions?.n_cells[i] ?? 0}
            by={conditionBy(r, i)}
            strong={flag === "condition" && (c === r.test?.high || c === r.test?.low)}
          />
        ))}
        {hasTest && (
          <td className="pl-2 text-right" title="Highest minus lowest removal rate between the conditions compared">
            {r.test ? r.test.max_diff_pp.toFixed(1) : "–"}
          </td>
        )}
        {hasTest && (
          <td
            className={`pl-2 text-right ${flag === "condition" ? "font-semibold text-amber-800" : ""}`}
            title={testTitle(r.test, cond?.column ?? "condition")}
          >
            {formatP(r.test?.p)}
          </td>
        )}
      </tr>
    );
  };
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px]" aria-label="Cells removed per group and condition">
        <thead>
          <tr className="text-left text-slate-400">
            <th className="font-medium" title={`The groups of '${label}'`}>
              {label}
            </th>
            <th className="pl-2 text-right font-medium" title="Removed as a share of all the group's cells, then removed / cells">
              All
            </th>
            {names.map((c) => (
              <th
                key={c}
                className="max-w-[7rem] truncate pl-2 text-right font-medium"
                title={`Removed as a share of the group's cells with ${cond!.column} = ${c}, then removed / cells`}
              >
                {c}
              </th>
            ))}
            {hasTest && (
              <th
                className="pl-2 text-right font-medium"
                title="Largest difference in removal rate between two conditions, in percentage points (conditions with 10+ cells in the group)"
              >
                Δ pts
              </th>
            )}
            {hasTest && (
              <th
                className="pl-2 text-right font-medium"
                title={`Chi-square test of removed vs kept by ${cond?.column ?? "condition"} within the group (Yates-corrected for two conditions; conditions with fewer than 10 cells in the group are left out). Small p: the removal depends on the condition.`}
              >
                p
              </th>
            )}
          </tr>
        </thead>
        <tbody className="text-slate-600">
          {shown.map((r) => row(r, `g:${r.name}`, r.name ?? "", `${r.name}: ${r.n_cells.toLocaleString()} cells.${medianText(r)}`))}
        </tbody>
        <tfoot className="border-t border-slate-100 text-slate-600">
          {row(audit.overall, "overall", "All cells", "Every cell, labelled or not, ignoring the groups")}
        </tfoot>
      </table>
      {rows.length > ROWS && (
        <button type="button" onClick={() => setAll((v) => !v)} className="text-[11px] text-primary hover:underline">
          {all ? "Show fewer" : `Show all ${rows.length}`}
        </button>
      )}
      {!all && <Caption>{capText({ shown: shown.length, total: rows.length, by: "flagged, then share removed" })}</Caption>}
    </div>
  );
}

function ColumnSelect({
  label,
  value,
  detected,
  options,
  onChange,
}: {
  label: string;
  value: string;
  detected: string | null;
  options: { name: string; n?: number; disabled?: boolean }[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="font-medium text-slate-500">{label}</span>
      <select
        aria-label={`${label} column for the removal audit`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="min-w-0 max-w-[12rem] rounded border border-slate-200 bg-white px-1.5 py-0.5 text-[11px] text-slate-600"
      >
        <option value="">{detected ? `Detected: ${detected}` : "None found"}</option>
        {options.map((o) => (
          <option key={o.name} value={o.name} disabled={o.disabled}>
            {o.name}
            {o.n !== undefined ? ` (${o.n})` : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

/**
 * "Who gets removed": the QC thresholds' loss per cell type x condition, with a
 * chi-square per cell type of removed vs condition, and flags for a cell type that
 * loses more than 30 % or loses at a rate that depends on the condition (a filter
 * that follows the treatment selects a cell state). After filtering, it shows the
 * record of what the filtering step removed; the preview stays one click away.
 */
export function RemovalAudit({
  preview,
  columns,
  onColumns,
}: {
  preview: FilterPreview;
  columns: RemovalAuditColumns;
  onColumns: (c: RemovalAuditColumns) => void;
}) {
  const obsColumns = useDatasetStore((s) => s.currentDataset?.obs_columns);
  const [choice, setChoice] = useState<"recorded" | "preview" | null>(null);
  const [allFlags, setAllFlags] = useState(false);
  const recorded = preview.recorded ?? null;
  const usable = recorded && !recorded.stale && recorded.by_type && recorded.overall ? recorded : null;
  const view = choice === "preview" || !usable ? "preview" : "recorded";
  const audit = (view === "recorded" ? usable : preview.audit) as Audit | null | undefined;

  const counts = useMemo(() => new Map((obsColumns ?? []).map((c) => [c.name, c.n_unique])), [obsColumns]);
  const labelOptions = useMemo(
    () =>
      view === "recorded"
        ? (usable?.columns ?? []).map((name) => ({ name, n: counts.get(name) }))
        : qcGroupOptions(obsColumns, MAX_LABEL_GROUPS),
    [view, usable, counts, obsColumns],
  );
  const conditionOptions = useMemo(
    () =>
      view === "recorded"
        ? labelOptions
        : qcGroupOptions(obsColumns, MAX_CONDITIONS),
    [view, labelOptions, obsColumns],
  );
  const flagged = useMemo(() => {
    const m = new Map<string | null, "loss" | "condition">();
    for (const f of audit?.flags ?? []) if (!m.has(f.group) || f.kind === "condition") m.set(f.group, f.kind);
    return m;
  }, [audit]);

  if (!audit) return null;
  const cond = audit.condition;
  const flags = audit.flags;
  const n = audit.overall.n_cells;
  const lcol = audit.label.column;
  const when = usable ? formatDate(usable.timestamp) : "";
  const intro =
    view === "recorded" && usable
      ? `What the filtering step removed${when ? ` on ${when}` : ""}: ${usable.n_removed.toLocaleString()} of ${usable.n_before.toLocaleString()} cells (${pct(usable.n_before ? usable.n_removed / usable.n_before : 0)}).`
      : `Who the current thresholds remove: ${audit.overall.n_removed.toLocaleString()} of ${n.toLocaleString()} cells (${pct(audit.overall.fraction)}).`;
  const split = lcol
    ? `Per ${lcol} group (${reasonText(audit.label.reason) || "as chosen"})` +
      (cond ? ` and per ${cond.column} (${reasonText(cond.reason) || "as chosen"}).` : ".")
    : "No label or clustering column, so only the totals are shown.";

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3" data-testid="removal-audit">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-xs font-semibold text-slate-700">Who gets removed</h4>
        {usable && (
          <div role="group" aria-label="Removal audit source" className="inline-flex rounded-md border border-slate-200 p-0.5">
            {(
              [
                ["recorded", "What was removed"],
                ["preview", "Current thresholds"],
              ] as const
            ).map(([v, text]) => (
              <button
                key={v}
                type="button"
                aria-pressed={view === v}
                onClick={() => setChoice(v)}
                className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                  view === v ? "bg-primary text-white" : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {text}
              </button>
            ))}
          </div>
        )}
      </div>
      <p className="text-[11px] text-slate-700">
        <span className="font-semibold">{intro}</span> {split}
      </p>
      <div className="flex flex-wrap items-center gap-3 text-[11px]">
        <ColumnSelect
          label="Rows"
          value={columns.label_column ?? ""}
          detected={columns.label_column ? null : lcol}
          options={labelOptions}
          onChange={(v) => onColumns({ ...columns, label_column: v })}
        />
        <ColumnSelect
          label="Columns"
          value={columns.condition_column ?? ""}
          detected={columns.condition_column ? null : (cond?.column ?? null)}
          options={conditionOptions}
          onChange={(v) => onColumns({ ...columns, condition_column: v })}
        />
      </div>
      {recorded?.stale && (
        <p className="rounded-md bg-slate-50 px-2 py-1 text-[11px] text-slate-600" data-testid="removal-audit-stale">
          {recorded.note}
        </p>
      )}
      {flags.length > 0 && (
        <ul className="space-y-1" aria-label="Removal audit flags">
          {(allFlags ? flags : flags.slice(0, FLAGS)).map((f) => (
            <li
              key={`${f.kind}:${f.group}`}
              className="flex items-start gap-1.5 rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800"
            >
              <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
              <span>{f.message}</span>
            </li>
          ))}
        </ul>
      )}
      {flags.length > FLAGS && (
        <button type="button" onClick={() => setAllFlags((v) => !v)} className="text-[11px] text-primary hover:underline">
          {allFlags ? "Fewer flags" : `Show all ${flags.length} flags`}
        </button>
      )}
      <AuditTable audit={audit} flagged={flagged} />
      <Caption
        n={[
          { value: n, unit: "cells" },
          ...(lcol ? [{ value: audit.by_type.length, unit: `${lcol} groups` }] : []),
          ...(cond ? [{ value: cond.names.length, unit: `${cond.column} groups` }] : []),
        ]}
      >
        {audit.rule}
        {!cond && " Without a condition or sample column, removal is not tested against one."}
        {audit.n_unlabelled > 0 &&
          ` ${audit.n_unlabelled.toLocaleString()} cells have no ${lcol} label and count only in All cells.`}
        {audit.note ? ` ${audit.note}` : ""}
      </Caption>
    </div>
  );
}
