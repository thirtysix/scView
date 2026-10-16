import { describe, it, expect, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import { compareLabelKey } from "@/lib/centroidLabels";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { GroupLabelsToggle } from "./GroupLabelsToggle";

const groups = (name: string, n: number) => ({
  name,
  dtype: "category",
  n_unique: n,
  values: Array.from({ length: n }, (_, i) => `${name}${i}`),
});

const DATASET = {
  id: "ds",
  n_cells: 5000,
  obs_columns: [groups("cell_type", 13), groups("fine", 60), { name: "n_genes", dtype: "int64", n_unique: 900 }],
} as unknown as DatasetInfo;

const button = () => screen.getByRole("button", { name: /Labels/ });

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "ds" });
  useUnifiedViewStore.setState({ centroidLabels: null, scatterOverlay: null });
  useSplitViewStore.setState({ splitBy: null, compareColumns: null });
});

describe("GroupLabelsToggle", () => {
  it("is on by default for a colouring with up to 40 groups, and turns off for it alone", () => {
    useSettingsStore.setState({ colorBy: "cell_type" });
    render(<GroupLabelsToggle />);
    expect(button()).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(button());
    expect(useUnifiedViewStore.getState().centroidLabels).toEqual({ key: "cell_type", show: false });
    expect(button()).toHaveAttribute("aria-pressed", "false");
  });

  it("is off by default above 40 groups, and can be turned on", () => {
    useSettingsStore.setState({ colorBy: "fine" });
    render(<GroupLabelsToggle />);
    expect(button()).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(button());
    expect(useUnifiedViewStore.getState().centroidLabels).toEqual({ key: "fine", show: true });
  });

  it("is disabled for a numeric colouring or a gene overlay", () => {
    useSettingsStore.setState({ colorBy: "n_genes" });
    const { unmount } = render(<GroupLabelsToggle />);
    expect(button()).toBeDisabled();
    unmount();
    useSettingsStore.setState({ colorBy: "cell_type" });
    useUnifiedViewStore.setState({ scatterOverlay: { type: "expression", gene: "CD3E" } });
    render(<GroupLabelsToggle />);
    expect(button()).toBeDisabled();
  });

  it("sets the labels of a compare set as a whole", () => {
    useSettingsStore.setState({ colorBy: "n_genes" });
    useSplitViewStore.setState({ compareColumns: ["cell_type", "fine"] });
    render(<GroupLabelsToggle />);
    expect(button()).toHaveAttribute("aria-pressed", "true"); // cell_type's panel has them
    fireEvent.click(button());
    expect(useUnifiedViewStore.getState().centroidLabels).toEqual({
      key: compareLabelKey(["cell_type", "fine"]),
      show: false,
    });
  });
});
