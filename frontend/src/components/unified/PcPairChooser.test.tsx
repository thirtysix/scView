import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { PcPairChooser } from "./PcPairChooser";

type Dataset = ReturnType<typeof useDatasetStore.getState>["currentDataset"];
const dataset = (dims: Record<string, number>) =>
  ({ id: "d1", embedding_dimensions: dims }) as unknown as Dataset;

describe("PcPairChooser", () => {
  beforeEach(() => {
    useSettingsStore.setState({ embedding: "X_pca", pcaDims: [0, 1] });
    useDatasetStore.setState({ currentDataset: dataset({ X_pca: 5, X_umap: 2 }) });
  });

  it("shows only for a PCA with more than two columns", () => {
    const { container, rerender } = render(<PcPairChooser />);
    expect(screen.getByLabelText("PC on the x axis")).toBeTruthy();
    useSettingsStore.setState({ embedding: "X_umap" });
    rerender(<PcPairChooser />);
    expect(container.textContent).toBe("");
    useSettingsStore.setState({ embedding: "X_pca_harmony" });
    useDatasetStore.setState({ currentDataset: dataset({ X_pca_harmony: 30 }) });
    rerender(<PcPairChooser />);
    expect(screen.getByLabelText("PC on the y axis").querySelectorAll("option")).toHaveLength(30);
  });

  it("sets the pair, and swaps rather than plotting one PC against itself", () => {
    render(<PcPairChooser />);
    fireEvent.change(screen.getByLabelText("PC on the x axis"), { target: { value: "2" } });
    expect(useSettingsStore.getState().pcaDims).toEqual([2, 1]);
    fireEvent.change(screen.getByLabelText("PC on the y axis"), { target: { value: "2" } });
    expect(useSettingsStore.getState().pcaDims).toEqual([1, 2]);
  });

  it("falls back to PC1 vs PC2 when the pair is beyond this dataset's PCs", () => {
    useSettingsStore.setState({ pcaDims: [10, 11] });
    render(<PcPairChooser />);
    expect((screen.getByLabelText("PC on the x axis") as HTMLSelectElement).value).toBe("0");
  });
});
