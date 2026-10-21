import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { getMitoByType, type MitoByType as Report, type TypeMito } from "@/api/mitoByType";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { MitoRefilter } from "@/components/qc/MitoRefilter";
import { useDatasetStore } from "@/stores/datasetStore";

const MAX_GROUPS = 200;

function pctText(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || !isFinite(v) ? "—" : `${v.toFixed(digits)}%`;
}

function removed(n: number | null, frac: number | null): string {
  return n === null || frac === null ? "—" : `${n.toLocaleString()} (${Math.round(frac * 100)}%)`;
}

const SOURCE_TEXT: Record<string, string> = {
  requested: "the value typed here",
  recorded: "the dataset's last filtering",
  default: "the filtering step's default",
};

/** The ceiling input: commits on Enter or leaving the field; empty = recorded or default. */
function CeilingInput({ value, onCommit }: { value: string; onCommit: (v: string) => void }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  const commit = () => {
    const t = text.trim();
    if (t === "" || (isFinite(Number(t)) && Number(t) >= 0)) onCommit(t);
    else setText(value);
  };
  return (
    <label className="flex items-center gap-1 text-[11px] text-slate-500" title="Global % mito ceiling to compare against; empty uses the recorded one, else the default">
      Ceiling
      <input
        type="number"
        min={0}
        step={0.5}
        value={text}
        placeholder="auto"
        aria-label="Global % mito ceiling"
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && commit()}
        className="w-14 rounded border border-slate-200 px-1 py-0 text-[11px] text-slate-600"
      />
      %
    </label>
  );
}

function Row({ t, applied, filtered }: { t: TypeMito; applied: boolean; filtered: boolean }) {
  const tone = t.damaged ? "bg-red-50/50" : t.flagged ? "bg-amber-50/60" : "";
  return (
    <tr className={`border-t border-slate-100 ${tone}`} data-testid={`mito-row-${t.group}`}>
      <td className="px-2 py-1">
        <span className="flex items-center gap-1 font-medium text-slate-700">
          {(t.flagged || t.damaged) && (
            <AlertTriangle
              className={`h-3 w-3 flex-shrink-0 ${t.damaged ? "text-red-500" : "text-amber-500"}`}
              aria-label={t.damaged ? "Likely damaged" : "Flagged"}
            />
          )}
          <span title={[...t.reasons, t.note].filter(Boolean).join(" ") || undefined}>{t.group}</span>
        </span>
      </td>
      <td className="px-2 py-1 text-right tabular-nums">{t.n_cells.toLocaleString()}</td>
      <td
        className="px-2 py-1 text-right tabular-nums"
        title={t.quantiles.q05 !== undefined ? `5th-95th percentile ${pctText(t.quantiles.q05)}–${pctText(t.quantiles.q95)}; IQR ${pctText(t.quantiles.q25)}–${pctText(t.quantiles.q75)}` : undefined}
      >
        {pctText(t.median)}
      </td>
      <td className="px-2 py-1 text-right tabular-nums">{t.mad === null ? "—" : t.mad.toFixed(2)}</td>
      <td className="px-2 py-1 text-right tabular-nums" title={t.bound === null ? t.note || undefined : undefined}>
        {pctText(t.bound)}
      </td>
      <td className={`px-2 py-1 text-right tabular-nums ${t.flagged ? "font-medium text-amber-800" : ""}`}>
        {removed(t.removed_global, t.removed_global_frac)}
      </td>
      <td className="px-2 py-1 text-right tabular-nums">{removed(t.removed_bound, t.removed_bound_frac)}</td>
      {filtered && (
        <td
          className={`px-2 py-1 text-right tabular-nums ${t.flagged && (t.removed_filter_frac ?? 0) > t.removed_global_frac ? "font-medium text-amber-800" : ""}`}
        >
          {removed(t.removed_filter ?? null, t.removed_filter_frac ?? null)}
        </td>
      )}
      {applied && (
        <td className="px-2 py-1 text-right tabular-nums">
          {t.applied_rule === "own" ? pctText(t.applied_bound) : t.applied_rule ? <span className="text-slate-400">global</span> : "—"}
        </td>
      )}
    </tr>
  );
}

/** % mito per cell type against one global ceiling: which types the ceiling (or, after
 *  filtering, the rule it used) cuts hard, and what each type's own bound (median + 3
 *  MADs) would remove instead. A dataset filtered per type is shown with the bound each
 *  type had. When filtering cut a type hard, "Re-filter" applies the bounds from the
 *  original upload. Collapsed to one line when no type is flagged. */
export function MitoByType({
  datasetId,
  refreshKey = 0,
  onRefiltered,
}: {
  datasetId: string;
  refreshKey?: number;
  /** After a re-filter saved a new dataset (the panel reloads its views). */
  onRefiltered?: () => void;
}) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const [column, setColumn] = useState("");
  const [ceiling, setCeiling] = useState("");
  const [data, setData] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<boolean | null>(null); // null = follow the flags

  useEffect(() => {
    setColumn("");
    setCeiling("");
    setData(null);
  }, [datasetId]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getMitoByType(datasetId, column || undefined, ceiling === "" ? undefined : Number(ceiling))
      .then((d) => !cancelled && setData(d && typeof d === "object" && "measurable" in d ? d : null))
      .catch((e) => !cancelled && setError(String(e?.message ?? e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, column, ceiling, refreshKey]);

  const options = useMemo(
    () =>
      (dataset?.obs_columns ?? [])
        .filter((c) => (c.dtype === "category" || c.dtype === "object") && c.n_unique >= 2 && c.n_unique <= MAX_GROUPS)
        .map((c) => c.name),
    [dataset],
  );

  if (error) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-xs text-slate-400">
        Mito by cell type unavailable: {error}
      </div>
    );
  }
  const types = data?.types ?? [];
  const damaged = types.filter((t) => t.damaged);
  const expanded = open ?? ((data?.n_flagged ?? 0) > 0 || damaged.length > 0);
  const perType = data?.filter_mito_mode === "per_type" && types.some((t) => t.applied_rule);
  const filtered = types.some((t) => t.removed_filter !== null && t.removed_filter !== undefined);
  const headline = !data
    ? null
    : !data.column
      ? data.note
      : !data.measurable
        ? "% mito not measurable"
        : perType && data.ceiling_source !== "requested"
          ? `Filtered with per-type bounds (${types.filter((t) => t.applied_rule === "own").length} of ${types.length} types)` +
            (data.n_flagged ? `; ${data.n_flagged} cut hard by the global rule` : "")
          : data.n_flagged
            ? `${data.n_flagged} of ${types.length} types cut hard by ${filtered ? "filtering" : `the ${pctText(data.ceiling)} ceiling`}`
            : `The ${pctText(data.ceiling)} ceiling suits every type (${types.length} types)`;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3" data-testid="mito-by-type">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setOpen(!expanded)}
          aria-expanded={expanded}
          className="flex items-center gap-1 text-sm font-semibold text-slate-700"
        >
          {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          Mito by cell type
        </button>
        <InfoTip label="About per-type mito bounds" width={300}>
          Cell types differ in their normal mitochondrial share, so one % mito ceiling can remove most
          cells of a type with a high baseline and few of another. Each type&apos;s own bound here is its
          median + 3 MADs. The filtering step applies them when &quot;% mito: Per cell type&quot; is chosen; on a
          dataset already filtered, Re-filter applies them from the original upload. {data?.citations ?? ""}
        </InfoTip>
        {loading ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-slate-400" />
        ) : headline ? (
          <span className={`text-xs ${data?.n_flagged ? "text-amber-700" : "text-slate-500"}`}>{headline}</span>
        ) : null}
        <span className="flex-1" />
        {data?.measurable !== false && <CeilingInput value={ceiling} onCommit={setCeiling} />}
        {options.length > 0 && (
          <select
            aria-label="Label column for mito by cell type"
            value={column || data?.column || ""}
            onChange={(e) => setColumn(e.target.value)}
            className="rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600"
          >
            {!data?.column && <option value="">Choose a label column…</option>}
            {options.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        )}
      </div>

      {expanded && data?.column && !data.measurable && (
        <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{data.note}</p>
      )}

      {expanded && data?.column && data.measurable && (
        <div className="mt-2 space-y-2">
          {data.summary && <p className="text-xs text-slate-700">{data.summary}</p>}
          <div className="max-h-72 overflow-auto rounded border border-slate-100">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-50 text-left text-[11px] text-slate-500">
                <tr>
                  <th className="px-2 py-1 font-medium" title={`Groups of '${data.column}'; hover a name for why it is flagged`}>Type</th>
                  <th className="px-2 py-1 text-right font-medium" title="Cells of the type with a % mito value">Cells</th>
                  <th className="px-2 py-1 text-right font-medium" title="Median % of a cell's counts from mitochondrial (MT-) genes; hover a value for its percentiles">Median mito</th>
                  <th className="px-2 py-1 text-right font-medium" title="Median absolute deviation of % mito within the type (raw, not scaled to an SD)">MAD</th>
                  <th className="px-2 py-1 text-right font-medium" title={`The type's own bound: median + ${data.mad_k} MADs. None for fewer than ${data.min_cells} cells, no spread, or a type that looks damaged`}>Own bound</th>
                  <th className="px-2 py-1 text-right font-medium" title={`Cells above the global ${pctText(data.ceiling)} ceiling (${SOURCE_TEXT[data.ceiling_source ?? "default"]})`}>Removed at {pctText(data.ceiling)}</th>
                  <th className="px-2 py-1 text-right font-medium" title="Cells above the type's own bound (median + 3 MADs)">Removed by own bound</th>
                  {filtered && (
                    <th
                      className="px-2 py-1 text-right font-medium"
                      title="Cells the filtering step's % mito rule actually removed, from its record. In MAD mode each sample's bound (median + 3 MADs) can sit well below the ceiling, so this can exceed the ceiling's count"
                    >
                      Removed by filtering
                    </th>
                  )}
                  {perType && (
                    <th className="px-2 py-1 text-right font-medium" title="The % mito bound the per-type filtering judged the type by; 'global' when it kept the global rule">
                      Filtered at
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {types.map((t) => (
                  <Row key={t.group} t={t} applied={perType} filtered={filtered} />
                ))}
              </tbody>
            </table>
          </div>
          {types.some((t) => t.flagged || t.damaged) && (
            <ul className="space-y-1 text-xs text-slate-600">
              {types
                .filter((t) => t.flagged || t.damaged)
                .map((t) => (
                  <li key={t.group} className="flex items-start gap-1.5">
                    <AlertTriangle className={`mt-0.5 h-3.5 w-3.5 flex-shrink-0 ${t.damaged ? "text-red-500" : "text-amber-500"}`} />
                    <span>
                      <span className="font-medium text-slate-700">{t.group}:</span> {t.damaged ? t.note : t.reasons.join(" ")}
                    </span>
                  </li>
                ))}
            </ul>
          )}
          <Caption
            n={[
              { value: data.n_labelled, of: data.n_cells, unit: "cells" },
              { value: types.length, unit: `types of '${data.column}'` },
            ]}
          >
            % mito per {data.reason === "cell type" ? "author cell type" : "group"}, against the{" "}
            {pctText(data.ceiling)} ceiling from {SOURCE_TEXT[data.ceiling_source ?? "default"]}. {data.rule} A type&apos;s
            own bound can be stricter than the ceiling; it assumes the type&apos;s typical cell is healthy.{" "}
            {perType
              ? `Filtered with per-type bounds on '${data.filter_type_column ?? data.column}'.`
              : data.already_filtered
                ? "Filtered with one global rule."
                : "Not filtered by scView yet."}
            {data.note ? ` ${data.note}` : ""}
          </Caption>
          {data.n_flagged > 0 && data.already_filtered && data.column && (
            <MitoRefilter
              datasetId={datasetId}
              column={data.column}
              perCluster={data.reason === "clustering" || data.column.startsWith("scview_")}
              onDone={onRefiltered}
            />
          )}
          {data.n_flagged > 0 && !data.already_filtered && (
            <p className="text-[11px] text-slate-600" data-testid="mito-per-type-hint">
              Not filtered yet: choose &quot;% mito: Per cell type&quot; in the Cell/Gene Filtering step to apply
              these bounds when it runs.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
