import { Tags } from "lucide-react";
import { LABELS_AUTO_MAX_GROUPS, compareLabelKey, labelsShown } from "@/lib/centroidLabels";
import { isCategoricalDtype } from "@/lib/obsColumns";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

/**
 * Toolbar "Labels": each group's name drawn at its centre on the plot (and in
 * every split or compare panel). On by default up to 40 groups; a click sets it
 * for the current colouring only.
 */
export function GroupLabelsToggle() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const overlay = useUnifiedViewStore((s) => s.scatterOverlay);
  const choice = useUnifiedViewStore((s) => s.centroidLabels);
  const setChoice = useUnifiedViewStore((s) => s.setCentroidLabels);
  const compareColumns = useSplitViewStore((s) => s.compareColumns);

  const groups = (name: string) => {
    const col = dataset?.obs_columns.find((c) => c.name === name);
    return col && isCategoricalDtype(col.dtype) ? col.values?.length ?? 0 : 0;
  };
  // What the labels belong to: the compare set, or the colour-by column.
  const target = compareColumns
    ? { key: compareLabelKey(compareColumns), counts: compareColumns.map(groups) }
    : colorBy && !overlay && groups(colorBy) > 0
      ? { key: colorBy, counts: [groups(colorBy)] }
      : null;
  const on = !!target && target.counts.some((n) => labelsShown(choice, target.key, n));

  return (
    <button
      onClick={() => target && setChoice({ key: target.key, show: !on })}
      disabled={!target}
      aria-pressed={on}
      title={
        !target
          ? "Group labels need a categorical colouring"
          : on
            ? "Hide the group labels on the plot"
            : `Show each group's name at its centre (on by default up to ${LABELS_AUTO_MAX_GROUPS} groups)`
      }
      className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        on ? "bg-blue-50 text-blue-700 ring-1 ring-blue-300" : "text-slate-600 hover:bg-slate-100"
      }`}
    >
      <Tags className="h-3.5 w-3.5" />
      Labels
    </button>
  );
}
