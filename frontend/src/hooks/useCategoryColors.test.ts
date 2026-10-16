import { describe, it, expect, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useCategoryColors, useCategoryOrder } from "./useCategoryColors";
import { resetCategoryColors } from "@/lib/categoryColors";
import { mapCategoryToColor } from "@/lib/colors";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import type { DatasetInfo } from "@/api/types";

const dataset = {
  id: "ds1",
  obs_columns: [{ name: "leiden", dtype: "category", n_unique: 3, values: ["0", "1", "2"] }],
} as unknown as DatasetInfo;

describe("useCategoryColors", () => {
  beforeEach(() => {
    resetCategoryColors();
    useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
    useSettingsStore.setState({ categoricalPalette: "classic" });
  });

  it("colours a category as the scatter does, from the current dataset's order", () => {
    const { result } = renderHook(() => useCategoryColors("leiden"));
    expect([...result.current("2")]).toEqual(mapCategoryToColor(2, "classic").slice(0, 3));
    expect([...result.current("0")]).toEqual(mapCategoryToColor(0, "classic").slice(0, 3));
  });

  it("returns the column's category order", () => {
    const { result } = renderHook(() => useCategoryOrder("leiden"));
    expect(result.current).toEqual(["0", "1", "2"]);
  });
});
