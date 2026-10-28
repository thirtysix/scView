import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import type { DatasetInfo, SpatialInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useTissueViewStore } from "@/stores/tissueViewStore";

// The tissue plot's toolbar, its section choice and the cluster map beside it.
const { fetchEmbeddingBinary, scatterProps } = vi.hoisted(() => ({
  fetchEmbeddingBinary: vi.fn(),
  scatterProps: { current: null as Record<string, unknown> | null },
}));
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary }));
vi.mock("@/components/plots/EmbeddingScatter", () => ({
  EmbeddingScatter: (props: Record<string, unknown>) => {
    scatterProps.current = props;
    return <div data-testid="scatter" />;
  },
}));

import { UnifiedToolbar } from "./UnifiedToolbar";
import { ClusterReference } from "./ClusterReference";
import { useTissuePositions } from "@/hooks/useTissuePositions";

const lib = (id: string, n: number) => ({
  id,
  n_obs: n,
  images: { lowres: { width: 600, height: 600, scalef: 0.02 } },
  spot_diameter_fullres: 90,
  bounds: [0, 0, 100, 100] as [number, number, number, number],
});
const SPATIAL: SpatialInfo = {
  technology: "visium",
  unit: "spot",
  library_column: "library_id",
  libraries: [lib("A1", 2), lib("B1", 1)],
};
const DATASET: DatasetInfo = {
  id: "vis",
  name: "Visium",
  filename: "v.h5",
  n_cells: 3,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_pca", "X_umap"],
  obs_columns: [
    { name: "leiden", dtype: "category", n_unique: 2, values: ["0", "1"] },
    { name: "library_id", dtype: "category", n_unique: 2, values: ["A1", "B1"] },
  ],
  created_at: "",
  spatial: SPATIAL,
};

const arrow = (cols: Record<string, Float32Array | Int32Array>) => tableToIPC(tableFromArrays(cols)).buffer as ArrayBuffer;

const toolbar = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <UnifiedToolbar numCells={3} />
    </QueryClientProvider>,
  );

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "vis" });
  useSettingsStore.setState({ embedding: "X_umap", colorBy: "leiden" });
  useTissueViewStore.setState({ library: null, showImage: true, imageOpacity: 1 });
  useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
  fetchEmbeddingBinary.mockReset();
  scatterProps.current = null;
});

describe("Embedding chooser", () => {
  it("offers Tissue after the real embeddings, only with spatial data", () => {
    toolbar();
    const options = within(screen.getByLabelText("Embedding")).getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["PCA", "UMAP", "Tissue"]);
    expect((options[2] as HTMLOptionElement).value).toBe("spatial");
    expect(screen.getByText("3 spots")).toBeInTheDocument();
  });

  it("has no Tissue without spatial data", () => {
    useDatasetStore.setState({ currentDataset: { ...DATASET, spatial: null } });
    toolbar();
    const options = within(screen.getByLabelText("Embedding")).getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["PCA", "UMAP"]);
    expect(screen.getByText("3 cells")).toBeInTheDocument();
  });

  it("on the tissue: Section, Image and its opacity; Split and Compare off with the reason", () => {
    useSettingsStore.setState({ embedding: "spatial" });
    toolbar();
    const section = screen.getByLabelText("Section");
    expect(within(section).getAllByRole("option").map((o) => o.textContent)).toEqual(["A1 (2 spots)", "B1 (1 spot)"]);
    fireEvent.change(section, { target: { value: "B1" } });
    expect(useTissueViewStore.getState().library).toBe("B1");
    fireEvent.click(screen.getByLabelText("Image"));
    expect(useTissueViewStore.getState().showImage).toBe(false);
    expect(screen.queryByLabelText("Image opacity")).toBeNull();
    const split = screen.getByLabelText("Split the plot by");
    expect(split).toBeDisabled();
    expect(split.title).toMatch(/not the tissue/);
  });
});

describe("one section at a time", () => {
  it("masks the other sections' cells once the library column is in", async () => {
    fetchEmbeddingBinary.mockResolvedValue(
      arrow({ x: Float32Array.from([1, 2, 3]), y: Float32Array.from([1, 2, 3]), library_id: Int32Array.from([0, 1, 0]) }),
    );
    const world = Float32Array.from([1, -1, 2, -2, 3, -3]);
    const { result, rerender } = renderHook(() => useTissuePositions(DATASET, "spatial", world));
    expect(result.current.positions).toBeNull(); // not before the sections are known
    await waitFor(() => expect(result.current.positions).not.toBeNull());
    expect(fetchEmbeddingBinary).toHaveBeenCalledWith("vis", "spatial", "library_id");
    expect(result.current.library?.id).toBe("A1");
    const a = result.current.positions!;
    expect([a[0], a[4]]).toEqual([1, 3]);
    expect(Number.isNaN(a[2])).toBe(true);

    useTissueViewStore.setState({ library: "B1" });
    rerender();
    const b = result.current.positions!;
    expect(b[2]).toBe(2);
    expect(Number.isNaN(b[0])).toBe(true);
    expect(Number.isNaN(b[4])).toBe(true);
  });

  it("leaves a single section and other embeddings as they are", () => {
    const pos = Float32Array.from([1, 1]);
    const one = { ...DATASET, spatial: { ...SPATIAL, libraries: [lib("A1", 1)] } };
    expect(renderHook(() => useTissuePositions(one, "spatial", pos)).result.current.positions).toBe(pos);
    expect(renderHook(() => useTissuePositions(DATASET, "X_umap", pos)).result.current).toEqual({ positions: pos, library: null });
    expect(fetchEmbeddingBinary).not.toHaveBeenCalled();
  });
});

describe("Cluster map beside the tissue", () => {
  it("draws its own embedding's coordinates and shows the lassoed spots", async () => {
    fetchEmbeddingBinary.mockResolvedValue(
      arrow({ x: Float32Array.from([5, 6, 7]), y: Float32Array.from([8, 9, 10]), leiden: Int32Array.from([0, 1, 1]) }),
    );
    useSelectionStore.setState({ selectedCellIndices: new Set([2]) });
    render(
      <ClusterReference datasetId="vis" embedding="X_umap" column="leiden" positions={null} dimensions={2} categories={["0", "1"]} />,
    );
    await screen.findByTestId("scatter");
    expect(fetchEmbeddingBinary).toHaveBeenCalledWith("vis", "X_umap", "leiden");
    expect(Array.from(scatterProps.current!.positions as Float32Array)).toEqual([5, 8, 6, 9, 7, 10]);
    expect(scatterProps.current!.selectedIndices).toEqual(new Set([2]));
    // A cluster chip highlights the cluster's spots instead.
    fireEvent.click(screen.getByRole("button", { name: "1" }));
    expect(useSelectionStore.getState().highlightedGroup).toEqual({ column: "leiden", value: "1" });
    expect(scatterProps.current!.selectedIndices).toEqual(new Set([1, 2]));
  });
});
