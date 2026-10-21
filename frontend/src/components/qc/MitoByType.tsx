import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { getMitoByType, type MitoByType as Report, type TypeMito } from "@/api/mitoByType";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
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

function Row({ t }: { t: TypeMito }) {
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
    </tr>
  );
}

/** % mito per cell type against one global ceiling: which types the ceiling cuts hard,
 *  and what each type's own bound (median + 3 MADs) would remove instead. Advisory:
 *  the filtering step is unchanged. Collapsed to one line when no type is flagged. */
export function MitoByType({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
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
  const headline = !data
    ? null
    : !data.column
      ? data.note
      : !data.measurable
        ? "% mito not measurable"
        : data.n_flagged
          ? `${data.n_flagged} of ${types.length} types cut hard by the ${pctText(data.ceiling)} ceiling`
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
          median + 3 MADs. Advisory: the filtering step still uses one ceiling.{" "}
          {data?.citations ?? ""}
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
                </tr>
              </thead>
              <tbody>
                {types.map((t) => (
                  <Row key={t.group} t={t} />
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
            own bound can be stricter than the ceiling; it assumes the type&apos;s typical cell is healthy.
            Advisory: filtering is unchanged.{data.note ? ` ${data.note}` : ""}
          </Caption>
        </div>
      )}
    </div>
  );
}
