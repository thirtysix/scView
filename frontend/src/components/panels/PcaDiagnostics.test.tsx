import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { PcaDiagnostics as Result } from "@/api/pca";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";

const { getPcaDiagnostics, apiFetchBinary, decodeArrowBuffer } = vi.hoisted(() => ({
  getPcaDiagnostics: vi.fn(),
  apiFetchBinary: vi.fn(),
  decodeArrowBuffer: vi.fn(),
}));
vi.mock("@/api/pca", () => ({ getPcaDiagnostics }));
vi.mock("@/api/client", () => ({ apiFetchBinary }));
vi.mock("@/lib/arrow", () => ({ decodeArrowBuffer }));
// Plotly needs a real canvas; what it is handed is what matters here.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data, layout }: { data: { y?: (string | number)[] }[]; layout: { shapes?: { x0: number }[] } }) => (
    <div data-testid="plot" data-shapes={(layout.shapes ?? []).map((s) => s.x0).join(",")}>
      {data.map((t) => (t.y ?? []).join(" ")).join("|")}
    </div>
  ),
}));

import { PcaDiagnostics } from "./PcaDiagnostics";

const DATA: Result = {
  available: true,
  n_cells: 13836,
  n_pcs: 3,
  n_genes_total: 14053,
  n_genes_used: 2000,
  params: { n_comps: 3, use_highly_variable: true, mask_var: "highly_variable", zero_center: true },
  variance: [3, 2, 1],
  // Exact binary fractions, so 100 x the share prints without float noise.
  variance_ratio: [0.5, 0.25, 0.03125],
  cumulative: [0.5, 0.75, 0.78125],
  ratio_basis: "total",
  retained: 0.78125,
  elbow: 2,
  pcs_90: 3,
  neighbors: { use_rep: "X_pca_harmony", n_pcs: 3, n_neighbors: 15, from_pca: true },
  loadings: [
    { pc: 1, positive: [{ gene: "TYROBP", loading: 0.2 }], negative: [{ gene: "CD3D", loading: -0.15 }] },
    { pc: 2, positive: [{ gene: "MS4A1", loading: 0.3 }], negative: [{ gene: "NKG7", loading: -0.1 }] },
    { pc: 3, positive: [], negative: [] },
  ],
  method: "Kneedle…",
};

describe("PcaDiagnostics", () => {
  beforeEach(() => {
    getPcaDiagnostics.mockReset().mockResolvedValue(DATA);
    apiFetchBinary.mockReset();
    decodeArrowBuffer.mockReset();
    useUnifiedViewStore.getState().clearOverlayState();
    useViewStore.getState().setPanel("assessment");
  });

  it("loads on Show only, marks the elbow and the neighbours' PCs, and captions n", async () => {
    render(<PcaDiagnostics datasetId="d1" />);
    expect(getPcaDiagnostics).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await waitFor(() => expect(screen.getByText("Variance per PC")).toBeTruthy());
    expect(getPcaDiagnostics).toHaveBeenCalledWith("d1");
    expect(screen.getByText("PC2")).toBeTruthy(); // the elbow metric
    expect(screen.getByText("3 PCs of X_pca_harmony")).toBeTruthy();
    const plots = screen.getAllByTestId("plot");
    expect(plots[0]!.dataset.shapes).toBe("2,3"); // elbow line, neighbours line
    expect(plots[0]!.textContent).toBe("50 25 3.125|50 75 78.125"); // percentages, then cumulative
    expect(plots[1]!.textContent).toBe("CD3D TYROBP"); // PC1 loadings, negative first
    expect(screen.getByText(/3 PCs from 13,836 cells and 2,000 of 14,053 genes \(variable genes\)/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Hide" }));
    expect(screen.queryByText("Variance per PC")).toBeNull();
  });

  it("switches the loadings to the chosen PC", async () => {
    render(<PcaDiagnostics datasetId="d1" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await waitFor(() => screen.getByText("Loadings"));
    const select = screen.getByLabelText("Principal component") as HTMLSelectElement;
    expect(select.options[1]!.textContent).toBe("PC2 (25%)");
    expect(select.options[2]!.textContent).toBe("PC3 (3.1%)");
    fireEvent.change(select, { target: { value: "2" } });
    expect(screen.getAllByTestId("plot")[1]!.textContent).toBe("NKG7 MS4A1");
    expect(screen.getByRole("button", { name: "MS4A1" })).toBeTruthy();
  });

  it("colours the scatter by a clicked gene and opens the Unified View", async () => {
    apiFetchBinary.mockResolvedValue(new ArrayBuffer(8));
    decodeArrowBuffer.mockReturnValue({ TYROBP: new Float32Array([0, 1, 2]) });
    render(<PcaDiagnostics datasetId="d1" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await waitFor(() => screen.getByRole("button", { name: "TYROBP" }));
    fireEvent.click(screen.getByRole("button", { name: "TYROBP" }));
    await waitFor(() => expect(useViewStore.getState().activePanel).toBe("unified"));
    expect(apiFetchBinary).toHaveBeenCalledWith("/datasets/d1/expression?genes=TYROBP");
    const s = useUnifiedViewStore.getState();
    expect(s.scatterOverlay).toEqual({ type: "expression", gene: "TYROBP" });
    expect(s.overlayLabel).toBe("TYROBP");
    expect(s.overlayValues).toEqual(new Float32Array([0, 1, 2]));
    expect(s.overlayDatasetId).toBe("d1");
  });

  it("says why when the dataset has no PCA, and reports a failed request", async () => {
    getPcaDiagnostics.mockResolvedValue({ available: false, reason: "No PCA yet: run the pipeline's PCA step first." });
    render(<PcaDiagnostics datasetId="d1" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await waitFor(() => expect(screen.getByText(/No PCA yet/)).toBeTruthy());
    expect(screen.queryByTestId("plot")).toBeNull();
    getPcaDiagnostics.mockRejectedValue(new Error("Backend is down"));
    render(<PcaDiagnostics datasetId="d2" />);
    // The first panel now reads "Hide": the only "Show" is the new one.
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/Backend is down/));
  });
});
