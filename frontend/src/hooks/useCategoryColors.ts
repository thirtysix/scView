import { useMemo } from "react";
import { categoryColor } from "@/lib/categoryColors";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * Colour lookup for the categories of one obs column of the current dataset,
 * matching the embedding scatter and its legend (lib/categoryColors.ts).
 */
export function useCategoryColors(
  column: string | null | undefined,
): (category: string) => readonly [number, number, number] {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const paletteName = useSettingsStore((s) => s.categoricalPalette);
  return useMemo(() => {
    const datasetId = dataset?.id ?? "";
    const col = column ?? "";
    const order = dataset?.obs_columns.find((c) => c.name === col)?.values ?? [];
    return (category: string) => categoryColor(datasetId, col, order, category, paletteName);
  }, [dataset, column, paletteName]);
}

/** The column's category order from the backend (the scatter's order), or []. */
export function useCategoryOrder(column: string | null | undefined): readonly string[] {
  const dataset = useDatasetStore((s) => s.currentDataset);
  return useMemo(
    () => dataset?.obs_columns.find((c) => c.name === column)?.values ?? [],
    [dataset, column],
  );
}
