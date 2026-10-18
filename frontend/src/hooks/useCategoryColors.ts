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
    const info = dataset?.obs_columns.find((c) => c.name === col);
    const order = info?.values ?? [];
    return (category: string) => categoryColor(datasetId, col, order, category, paletteName, info?.color_slots);
  }, [dataset, column, paletteName]);
}

/** The column's palette slot per category code once colours are pinned to labels, else undefined. */
export function useCategorySlots(column: string | null | undefined): readonly number[] | undefined {
  const dataset = useDatasetStore((s) => s.currentDataset);
  return useMemo(
    () => dataset?.obs_columns.find((c) => c.name === column)?.color_slots,
    [dataset, column],
  );
}

/** The column's category order from the backend (the scatter's order), or []. */
export function useCategoryOrder(column: string | null | undefined): readonly string[] {
  const dataset = useDatasetStore((s) => s.currentDataset);
  return useMemo(
    () => dataset?.obs_columns.find((c) => c.name === column)?.values ?? [],
    [dataset, column],
  );
}
