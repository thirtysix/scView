import { useEffect, useState } from "react";
import { getSelectionByGroup, type SelectionStats } from "@/api/selection";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { selectedSorted } from "@/lib/linkedSelection";

/**
 * The current selection's per-gene summary for a marker plot's "Selection" row
 * (C035), fetched when the selection or the genes change; null without a
 * selection, while loading, or when the request fails (the plot then simply
 * has no extra row).
 */
export function useSelectionByGroup(
  datasetId: string,
  genes: readonly string[],
  layer?: string,
): SelectionStats | null {
  const { selected } = useLinkedSelection();
  const [stats, setStats] = useState<{ key: unknown; value: SelectionStats } | null>(null);
  const key = genes.join("\u0000") + "\u0001" + (layer ?? "");
  useEffect(() => {
    if (!selected || genes.length === 0) return;
    let cancelled = false;
    getSelectionByGroup(datasetId, [...genes], selectedSorted(selected), layer)
      .then((value) => { if (!cancelled) setStats({ key: [datasetId, key, selected], value }); })
      .catch(() => { if (!cancelled) setStats(null); });
    return () => { cancelled = true; };
  }, [datasetId, key, selected]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!selected || !stats) return null;
  const [d, k, s] = stats.key as [string, string, ReadonlySet<number>];
  return d === datasetId && k === key && s === selected ? stats.value : null;
}
