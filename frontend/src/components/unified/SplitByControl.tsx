import { useEffect, useMemo } from "react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { prettyObsLabel } from "@/lib/formatting";
import { SPLIT_MAX_VALUES, splitColumnOptions } from "@/lib/splitView";

/**
 * Toolbar "Split" select: off, or a categorical column with 2–6 values to show
 * one scatter per value. Columns with more values are listed disabled, with
 * the reason, so it is clear why e.g. cell type is not offered.
 */
export function SplitByControl({ disabledReason = null }: { disabledReason?: string | null } = {}) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const splitBy = useSplitViewStore((s) => s.splitBy);
  const setSplitBy = useSplitViewStore((s) => s.setSplitBy);

  const options = useMemo(() => splitColumnOptions(dataset?.obs_columns ?? []), [dataset]);
  const allowed = options.filter((o) => o.disabledReason == null);
  const tooMany = options.filter((o) => o.disabledReason != null);

  // A split carried over from another dataset (or a renamed column) only
  // stays on when the current dataset can split by that column too.
  useEffect(() => {
    if (splitBy && !options.some((o) => o.name === splitBy && o.disabledReason == null)) {
      setSplitBy(null);
    }
  }, [splitBy, options, setSplitBy]);

  return (
    <div className="flex items-center gap-1.5">
      <label htmlFor="split-by" className="text-[11px] font-medium text-slate-500">
        Split
      </label>
      <select
        id="split-by"
        aria-label="Split the plot by"
        value={splitBy ?? ""}
        onChange={(e) => setSplitBy(e.target.value || null)}
        disabled={allowed.length === 0 || !!disabledReason}
        title={
          disabledReason
            ? disabledReason
            : allowed.length === 0
            ? `No column with 2–${SPLIT_MAX_VALUES} values to split by`
            : `One plot per value (2–${SPLIT_MAX_VALUES} values), sharing the camera, colours and colour scale`
        }
        className={`max-w-[140px] rounded border px-2 py-1 text-xs focus:border-blue-400 focus:outline-none disabled:opacity-50 ${
          splitBy
            ? "border-blue-300 bg-blue-50 text-blue-700"
            : "border-slate-300 bg-white text-slate-700"
        }`}
      >
        <option value="">Off</option>
        {allowed.map((o) => (
          <option key={o.name} value={o.name}>
            {prettyObsLabel(o.name)} ({o.nValues})
          </option>
        ))}
        {tooMany.length > 0 && (
          <optgroup label={`More than ${SPLIT_MAX_VALUES} values`}>
            {tooMany.map((o) => (
              <option key={o.name} value={o.name} disabled title={o.disabledReason ?? undefined}>
                {prettyObsLabel(o.name)} ({o.nValues})
              </option>
            ))}
          </optgroup>
        )}
      </select>
    </div>
  );
}
