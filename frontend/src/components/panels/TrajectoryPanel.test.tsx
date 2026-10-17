import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import type { PagaRecord, PagaView, TrendGenesResult } from "@/api/trajectory";

const { apiFetch, apiFetchBinary, getPaga, computePaga, rankTrendGenes, scatterProps } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  apiFetchBinary: vi.fn(),
  getPaga: vi.fn(),
  computePaga: vi.fn(),
  rankTrendGenes: vi.fn(),
  scatterProps: { current: null as Record<string, unknown> | null },
}));
vi.mock("@/api/client", () => ({ apiFetch, apiFetchBinary }));
vi.mock("@/api/trajectory", () => ({ getPaga, computePaga, rankTrendGenes }));
vi.mock("@/lib/arrow", () => ({
  decodeArrowBuffer: () => ({ "Hba-a2": Float32Array.from([0, 2, 4]) }),
}));
vi.mock("@/hooks/useEmbedding", () => ({
  useEmbedding: () => ({ positions: Float32Array.from([0, 0, 1, 1, 2, 2]), numCells: 3, dimensions: 2, isLoading: false }),
}));
// jsdom has no WebGL or canvas: record what the panel hands the scatter and plots.
vi.mock("@/components/plots/EmbeddingScatter", () => ({
  EmbeddingScatter: (props: Record<string, unknown>) => {
    scatterProps.current = props;
    return <div data-testid="scatter" />;
  },
}));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data, onClick }: { data: { type: string; y?: unknown[] }[]; onClick?: (e: unknown) => void }) => (
    <div data-testid={`plot-${data[0]?.type}`}>
      {data[0]?.type === "heatmap" &&
        (data[0].y as string[]).map((g) => (
          <button key={g} onClick={() => onClick?.({ points: [{ y: g }] })}>
            row {g}
          </button>
        ))}
    </div>
  ),
}));

import { TrajectoryPanel } from "./TrajectoryPanel";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const RECORD: PagaRecord = {
  groupby: "cluster",
  groups: ["MEP", "Ery1", "Ery2", "GMP"],
  sizes: [10, 20, 30, 40],
  edges: [[0, 1, 0.9], [1, 2, 0.8], [0, 3, 0.7]],
  tree: [[0, 1], [1, 2], [0, 3]],
  graph: "existing neighbour graph (use_rep=X_pca, k=15)",
  n_cells: 100,
};
const VIEW: PagaView = { paga: RECORD, positions: [[0, 0], [1, 1], [2, 2], [0, 2]], stale: null, root_group: "MEP" };

const trendResult = (n: number): TrendGenesResult => ({
  column: "scview_dpt_pseudotime",
  n_cells: n,
  n_genes_tested: 3000,
  min_detected: 10,
  expression: "log-normalised .raw",
  method: "spline",
  restrict: null,
  bins: { pseudotime: [0.1, 0.2, 0.3], n_cells: [10, 10, 10] },
  genes: [
    { gene: "Hba-a2", f_stat: 380, pval: 0, pval_adj: 0, r2: 0.6, amplitude: 2.6, peak_pseudotime: 0.3, pattern: "rising",
      n_detected: 900, mean: [0, 1, 3], fitted: [0, 1.2, 2.8] },
    { gene: "Apoe", f_stat: 200, pval: 0, pval_adj: 0, r2: 0.4, amplitude: 3.7, peak_pseudotime: 0.1, pattern: "falling",
      n_detected: 800, mean: [3, 1, 0], fitted: [2.9, 1, 0.1] },
  ],
});

function renderPanel() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <TrajectoryPanel />
    </QueryClientProvider>,
  );
}

describe("TrajectoryPanel: PAGA graph and genes along pseudotime", () => {
  beforeEach(() => {
    scatterProps.current = null;
    apiFetch.mockReset().mockImplementation((path: string) => {
      if (path.endsWith("/trajectory")) {
        return Promise.resolve({
          pseudotime_columns: ["scview_dpt_pseudotime"],
          n_columns: 1,
          computed: null,
          velocity_layers: true,
        });
      }
      if (path.endsWith("/trajectory/scview_dpt_pseudotime")) {
        return Promise.resolve({ column: "scview_dpt_pseudotime", n_cells: 3, values: [0, 0.5, 1], min: 0, max: 1 });
      }
      return Promise.resolve([]);
    });
    apiFetchBinary.mockReset().mockResolvedValue(new ArrayBuffer(8));
    getPaga.mockReset().mockResolvedValue(VIEW);
    computePaga.mockReset().mockResolvedValue(RECORD);
    rankTrendGenes.mockReset().mockImplementation((_d: string, _c: string, body: { groups?: string[] }) =>
      Promise.resolve(trendResult(body.groups ? 60 : 100)),
    );
    useSettingsStore.setState({ embedding: "X_umap" });
    useDatasetStore.setState({
      currentDatasetId: "d1",
      currentDataset: {
        id: "d1",
        name: "paul15",
        active_clustering: "cluster",
        obs_columns: [{ name: "cluster", dtype: "category", n_unique: 4, values: RECORD.groups }],
      } as unknown as DatasetInfo,
    });
  });

  it("draws the graph over the embedding and ranks genes on all cells, then on a branch", async () => {
    renderPanel();
    await waitFor(() => expect(getPaga).toHaveBeenCalledWith("d1", "X_umap"));
    await waitFor(() => {
      const ids = ((scatterProps.current?.overlayLayers as { id: string }[] | null) ?? []).map((l) => l.id);
      expect(ids).toEqual(["paga-edges-halo", "paga-edges", "paga-nodes", "paga-labels"]);
    });
    expect(screen.getByTestId("paga-summary")).toHaveTextContent("3 of 3 edges shown");
    expect(screen.getByTestId("velocity-note")).toHaveTextContent("spliced");

    await waitFor(() => expect(rankTrendGenes).toHaveBeenCalledWith("d1", "scview_dpt_pseudotime", {}, expect.anything()));
    // Heatmap rows: ordered by where each trend peaks (Apoe early, Hba-a2 late).
    const rows = await screen.findAllByRole("button", { name: /^row / });
    expect(rows.map((r) => r.textContent)).toEqual(["row Apoe", "row Hba-a2"]);

    // A branch: the tree path from the root group to the chosen end.
    fireEvent.change(screen.getByLabelText("Cells:"), { target: { value: "Ery2" } });
    await waitFor(() =>
      expect(rankTrendGenes).toHaveBeenLastCalledWith(
        "d1",
        "scview_dpt_pseudotime",
        { groups_column: "cluster", groups: ["MEP", "Ery1", "Ery2"] },
        expect.anything(),
      ),
    );
    expect(screen.getByTestId("trend-path")).toHaveTextContent("MEP → Ery1 → Ery2");
    // The branch is what the graph highlights.
    const edges = (scatterProps.current!.overlayLayers as { id: string; props: { data: { onPath: boolean }[] } }[])[1]!;
    expect(edges.props.data.filter((e) => e.onPath)).toHaveLength(2);

    // Clicking a gene plots its trend and colours the embedding by it.
    fireEvent.click(await screen.findByRole("button", { name: "row Hba-a2" }));
    await waitFor(() => expect(scatterProps.current!.colorLabel).toBe("Hba-a2 expression"));
    expect(apiFetchBinary).toHaveBeenCalledWith("/datasets/d1/expression?genes=Hba-a2");
    expect(screen.getByTestId("trend-focus")).toHaveTextContent("Hba-a2 rises");
    fireEvent.click(screen.getByRole("button", { name: /Back to pseudotime/ }));
    await waitFor(() => expect(scatterProps.current!.colorLabel).not.toBe("Hba-a2 expression"));
  });

  it("computes PAGA for the active clustering and hides a stale graph", async () => {
    getPaga.mockResolvedValue({ paga: null, positions: null, stale: null, root_group: null });
    renderPanel();
    const button = await screen.findByRole("button", { name: "Compute PAGA" });
    expect(screen.getByLabelText("PAGA grouping")).toHaveValue("cluster");
    getPaga.mockResolvedValue({ ...VIEW, stale: "'cluster' has changed since PAGA was computed; compute it again." });
    fireEvent.click(button);
    await waitFor(() => expect(computePaga).toHaveBeenCalledWith("d1", "cluster"));
    await screen.findByText(/has changed since PAGA/);
    expect(scatterProps.current!.overlayLayers).toBeNull();
    expect(screen.getByLabelText("Cells:")).toBeDisabled();
  });
});
