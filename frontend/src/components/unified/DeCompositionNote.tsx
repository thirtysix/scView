import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { fetchDEComposition, type DEComposition, type DECompositionParams } from "@/api/de";
import { CapNote, Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { balancedText, compositionRule, flaggedGroups, formatPoints, formatShare } from "@/lib/deComposition";
import { prettyObsLabel } from "@/lib/formatting";

/** Flagged groups listed in the note; the table has every group. */
const SHOWN_FLAGGED = 4;

interface Props {
  datasetId: string;
  params: DECompositionParams;
  /** Label columns the check can use (the subtab's "Within" columns). */
  columns: string[];
  /** Picked label column; "" = Auto (a cell-type column, else the active clustering). */
  column: string;
  onColumnChange: (column: string) => void;
  /** Restricts the DE to one group (sets Within); absent when the column can't be a scope. */
  onCompareWithin?: (column: string, group: string) => void;
  /** Columns the Within picker offers (a chip only for these). */
  scopeColumns: string[];
}

/**
 * Before a DE between groups: each side's share of every cluster/state among the
 * compared cells (GET /de/composition), with the groups whose shares differ by
 * more than the rule's points flagged, since the DE then partly measures which
 * cells each side holds. A chip restricts the comparison to one flagged group.
 */
export function DeCompositionNote({
  datasetId,
  params,
  columns,
  column,
  onColumnChange,
  onCompareWithin,
  scopeColumns,
}: Props) {
  const [data, setData] = useState<DEComposition | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const { split_by, group_a, group_b, within_column, within_value } = params;

  useEffect(() => {
    if (!split_by || !group_a || !group_b || group_a === group_b) {
      setData(null);
      return;
    }
    const ctl = new AbortController();
    setLoading(true);
    setError(null);
    fetchDEComposition(
      datasetId,
      { split_by, group_a, group_b, within_column, within_value, column: column || undefined },
      ctl.signal,
    )
      .then((res) => setData(res))
      .catch((e: unknown) => {
        if (ctl.signal.aborted) return;
        setData(null);
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!ctl.signal.aborted) setLoading(false);
      });
    return () => ctl.abort();
  }, [datasetId, split_by, group_a, group_b, within_column, within_value, column]);

  if (!split_by || !group_a || !group_b || group_a === group_b) return null;

  const flagged = data ? flaggedGroups(data) : [];
  const canScope = !!data?.column && !!onCompareWithin && scopeColumns.includes(data.column) && data.column !== split_by;
  const a = data?.label_a ?? group_a;
  const b = data?.label_b ?? group_b;
  const picker = (
    <label className="flex items-center gap-1">
      <span title="The label column whose groups' shares are compared between the two sides. Auto: an author cell-type column if the dataset has one, else the active clustering; within one of its groups, a finer column (a subclustering or QC state).">
        by
      </span>
      <select
        aria-label="Composition by"
        value={column}
        onChange={(e) => onColumnChange(e.target.value)}
        className="min-w-0 max-w-[10rem] rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700"
      >
        <option value="">Auto{data?.auto && data.column ? ` (${prettyObsLabel(data.column)})` : " (cell type or clustering)"}</option>
        {columns
          .filter((c) => c !== split_by)
          .map((c) => (
            <option key={c} value={c}>
              {prettyObsLabel(c)}
            </option>
          ))}
      </select>
    </label>
  );

  return (
    <div
      data-testid="de-composition"
      className={`rounded-lg border p-2 text-[11px] leading-snug ${
        data?.warn ? "border-amber-200 bg-amber-50 text-amber-900" : "border-slate-200 bg-white text-slate-600"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-1.5">
        <span className="flex items-center gap-1 font-medium">
          {data?.warn ? "Composition differs between the sides" : "Composition of the two sides"}
          <InfoTip label="About the composition check" width={280}>
            {compositionRule(data ?? { threshold_points: 10, min_cells: 10 })}
          </InfoTip>
          {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
        </span>
        {picker}
      </div>

      {error && <p className="mt-1 text-red-700">Composition check failed: {error}</p>}

      {data?.warn && (
        <ul className="mt-1 flex flex-col gap-0.5">
          {flagged.slice(0, SHOWN_FLAGGED).map((g) => (
            <li key={g.group} className="flex flex-wrap items-center gap-1.5">
              <span>
                <span className="font-semibold">{g.group}</span>: {formatShare(g.share_a)} of {a} vs{" "}
                {formatShare(g.share_b)} of {b} ({formatPoints(g.diff_points)})
                {g.one_sided && (
                  <span
                    className="ml-1 text-amber-700"
                    title="80% or more of this group's share is on one side: it may be a state the comparison itself creates (then its genes belong to the difference) rather than a separate population. Check by a cell-type column too."
                  >
                    nearly absent from {g.share_a > g.share_b ? b : a}
                  </span>
                )}
              </span>
              {canScope && (
                <button
                  type="button"
                  onClick={() => onCompareWithin!(data.column!, g.group)}
                  title={`Restrict the DE to the ${g.group} cells (Within: ${prettyObsLabel(data.column!)} = ${g.group}), so the two sides hold the same kind of cell`}
                  className="rounded-full border border-amber-300 bg-white px-1.5 py-px text-[10px] text-amber-800 hover:bg-amber-100"
                >
                  Compare within {g.group}
                </button>
              )}
            </li>
          ))}
          {flagged.length > SHOWN_FLAGGED && (
            <li className="text-amber-800">
              {flagged.length - SHOWN_FLAGGED} more flagged group{flagged.length - SHOWN_FLAGGED > 1 ? "s" : ""} in the table.
            </li>
          )}
        </ul>
      )}
      {data?.warn && (
        <p className="mt-1">
          Genes specific to these groups will look like {a} vs {b} differences. Compare within one group,
          or test composition (Compare &gt; Composition).
        </p>
      )}
      {data && !data.warn && balancedText(data) && <p className="mt-1">{balancedText(data)}</p>}
      {data?.note && <p className="mt-1 text-slate-500">{data.note}</p>}

      {data?.column && data.groups.length > 0 && (
        <>
          <Caption className="mt-1" n={[{ value: data.n_a, unit: `${a} cells` }, { value: data.n_b, unit: `${b} cells` }]}>
            Shares of each side&apos;s compared cells by {prettyObsLabel(data.column)}.
            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              className="ml-1 text-primary underline-offset-2 hover:underline"
            >
              {showAll ? "Hide table" : `Show the table (${data.n_groups.toLocaleString()} groups)`}
            </button>
          </Caption>
          {showAll && (
            <div className="mt-1 max-h-48 overflow-auto rounded border border-slate-200 bg-white text-slate-700">
              <table className="w-full text-[10px] tabular-nums">
                <thead className="sticky top-0 bg-slate-50 text-slate-500">
                  <tr>
                    <th className="px-1.5 py-0.5 text-left" title={`The ${prettyObsLabel(data.column)} group`}>Group</th>
                    <th className="px-1.5 py-0.5 text-right" title={`This group's cells as a share of the ${a} cells compared`}>{a}</th>
                    <th className="px-1.5 py-0.5 text-right" title={`This group's cells as a share of the ${b} cells compared`}>{b}</th>
                    <th className="px-1.5 py-0.5 text-right" title={`${a} share minus ${b} share, in percentage points; flagged past ${data.threshold_points}`}>Points</th>
                    <th className="px-1.5 py-0.5 text-right" title={`Cells of this group: ${a} / ${b}`}>n</th>
                  </tr>
                </thead>
                <tbody>
                  {data.groups.map((g) => (
                    <tr key={g.group} className={g.flagged ? "bg-amber-50 font-medium" : ""}>
                      <td className="px-1.5 py-0.5">{g.group}</td>
                      <td className="px-1.5 py-0.5 text-right">{formatShare(g.share_a)}</td>
                      <td className="px-1.5 py-0.5 text-right">{formatShare(g.share_b)}</td>
                      <td className="px-1.5 py-0.5 text-right">{formatPoints(g.diff_points).replace(" points", "")}</td>
                      <td className="px-1.5 py-0.5 text-right">
                        {g.n_a.toLocaleString()} / {g.n_b.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <CapNote className="px-1.5 py-0.5" shown={data.groups.length} total={data.n_groups} by="the difference in points" />
            </div>
          )}
        </>
      )}
    </div>
  );
}
