import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { DatasetInfo, ProteinInfo } from "@/api/types";

const { apiFetchBinary, decodeArrowBuffer } = vi.hoisted(() => ({
  apiFetchBinary: vi.fn(),
  decodeArrowBuffer: vi.fn(),
}));
vi.mock("@/api/client", async (orig) => ({
  ...(await orig<typeof import("@/api/client")>()),
  apiFetchBinary,
}));
// The "buffer" is the decoded table itself: the mock decoder hands it back.
vi.mock("@/lib/arrow", () => ({ decodeArrowBuffer }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; x?: number[] }[] }) => (
    <div data-testid="plot">{data.map((t) => `${t.type}:${(t.x ?? []).length}`).join(";")}</div>
  ),
}));
vi.mock("@/components/plots/MarkerPlot", () => ({
  MarkerPlot: ({ genes, layer, kind }: { genes: string[]; layer: string; kind: string }) => (
    <div data-testid="marker-plot">{`${kind}:${layer}:${genes.join(",")}`}</div>
  ),
}));
vi.mock("@/components/common/GeneSearch", () => ({
  GeneSearch: ({ onSelect }: { onSelect: (g: string) => void }) => (
    <button onClick={() => onSelect("MS4A1")}>pick MS4A1</button>
  ),
}));

import { UnifiedProteinSubtab } from "./UnifiedProteinSubtab";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const PROTEIN: ProteinInfo = {
  names: ["CD3", "CD19", "IgG1_control"],
  controls: ["IgG1_control"],
  layer: "protein",
  counts_layer: "protein_counts",
  normalisation: "CLR per cell",
};

describe("UnifiedProteinSubtab", () => {
  beforeEach(() => {
    decodeArrowBuffer.mockReset().mockImplementation((b: unknown) => b);
    apiFetchBinary.mockReset().mockImplementation(async (path: string) => {
      if (path.includes("layer=protein")) return { CD19: new Float32Array([0, 1, 3, 4]) };
      if (path.includes("CD19")) return { CD19: new Float32Array([0, 0, 1, 2]) }; // the gene CD19
      if (path.includes("MS4A1")) return { MS4A1: new Float32Array([0, 0, 0, 0]) };
      throw new Error("404");
    });
    useSettingsStore.setState({ expressionLayer: "raw" });
    useDatasetStore.setState({
      currentDatasetId: "d1",
      currentDataset: {
        id: "d1", name: "d1", filename: "", n_cells: 4, n_genes: 900, status: "ready", available_embeddings: [],
        created_at: "", obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "T"] }],
        protein: PROTEIN,
      } as DatasetInfo,
    });
  });

  const renderTab = (props: Partial<Parameters<typeof UnifiedProteinSubtab>[0]> = {}) =>
    render(
      <UnifiedProteinSubtab
        datasetId="d1"
        protein={PROTEIN}
        onProteinSelect={() => {}}
        activeProtein={null}
        groupByColumn="cell_type"
        setGroupByColumn={() => {}}
        proteinScale={null}
        {...props}
      />,
    );

  it("lists the targets apart from the isotype controls and colours the cells by a click", () => {
    const onProteinSelect = vi.fn();
    renderTab({ onProteinSelect });
    expect(screen.getByText(/2 antibodies measured with the RNA/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "CD19" }));
    expect(onProteinSelect).toHaveBeenCalledWith("CD19");
    expect(screen.getByRole("button", { name: "IgG1_control" }).title).toMatch(/isotype control/);
    // The heatmap reads the protein layer, controls left out.
    expect(screen.getByTestId("marker-plot").textContent).toBe("heatmap:protein:CD3,CD19");
  });

  it("sets the clicked protein against the gene of its target", async () => {
    renderTab({ activeProtein: "CD19" });
    await waitFor(() => screen.getByTestId("protein-rna-stats"));
    expect(apiFetchBinary).toHaveBeenCalledWith("/datasets/d1/expression?genes=CD19&layer=protein");
    expect(apiFetchBinary).toHaveBeenCalledWith("/datasets/d1/expression?genes=CD19,Cd19&layer=raw");
    expect(screen.getByTestId("plot").textContent).toBe("scattergl:4");
    const stats = screen.getByTestId("protein-rna-stats").textContent!;
    expect(stats).toMatch(/Spearman ρ 0\.95 over 4 cells/);
    expect(stats).toMatch(/CD19 RNA is not detected in 50% of cells/);
  });

  it("takes another gene, and says when none is found", async () => {
    renderTab({ activeProtein: "CD19" });
    await waitFor(() => screen.getByTestId("protein-rna-stats"));
    fireEvent.click(screen.getByText("pick MS4A1"));
    await waitFor(() => expect(screen.getByTestId("protein-rna-stats").textContent).toMatch(/MS4A1 RNA is not detected in 100%/));
    fireEvent.change(screen.getByLabelText("Protein to compare"), { target: { value: "CD3" } });
    await waitFor(() => screen.getByText(/No gene for CD3 found in the RNA \(tried CD3E/));
  });
});
