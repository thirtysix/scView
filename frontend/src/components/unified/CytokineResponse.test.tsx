import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { CytokineResult } from "@/api/cytokines";
import type { DatasetInfo } from "@/api/types";
import { comparisonText, cytokineHeatmap } from "@/lib/cytokines";

const { getCytokineActivity, getActivitySummary } = vi.hoisted(() => ({
  getCytokineActivity: vi.fn(),
  getActivitySummary: vi.fn(),
}));
vi.mock("@/api/cytokines", async (orig) => ({
  ...(await orig<typeof import("@/api/cytokines")>()),
  getCytokineActivity,
}));
vi.mock("@/api/activity", async (orig) => ({
  ...(await orig<typeof import("@/api/activity")>()),
  getActivitySummary,
}));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; y?: string[]; zmin?: number; zmax?: number }[] }) => (
    <div data-testid="plot">
      {data.map((t) => `${t.type}:${(t.y ?? []).join(",")}:${t.zmin}..${t.zmax}`).join(";")}
    </div>
  ),
}));

import { CytokineResponse } from "./CytokineResponse";
import { UnifiedActivitySubtab } from "./UnifiedActivitySubtab";
import { useDatasetStore } from "@/stores/datasetStore";

const RESULT: CytokineResult = {
  mode: "conditions",
  groupby: "cell_type",
  reference: "rest",
  condition: "stim",
  a: "STIM",
  b: "CTRL",
  cytokines: ["IFN1", "IFNG", "TNFA"],
  groups: ["CD14 Mono", "B"],
  group_cells: [
    [300, 310],
    [200, 190],
  ],
  skipped: [{ group: "pDC", reason: "4 vs 3 cells (need 10 each)" }],
  n_shared_genes: 4669,
  n_signature_genes: 4800,
  z: [
    [41.9, 25.5, -1],
    [48.6, 26.2, 0.5],
  ],
  p: [
    [0.001, 0.001, 0.3],
    [0.001, 0.001, 0.6],
  ],
  coef: [
    [0.1, 0.05, 0],
    [0.1, 0.05, 0],
  ],
  top: [
    { group: "CD14 Mono", cytokines: [{ cytokine: "IFN1", z: 41.9, p: 0.001 }, { cytokine: "IFNG", z: 25.5, p: 0.001 }, { cytokine: "TNFA", z: -1, p: 0.3 }] },
    { group: "B", cytokines: [{ cytokine: "IFN1", z: 48.6, p: 0.001 }, { cytokine: "TNFA", z: 0.5, p: 0.6 }] },
  ],
  lambda: 10000,
  n_permutations: 1000,
  species: "human",
  mapped_by: "human gene symbols",
  matrix_source: "log-normalised X",
  citation: "Jiang et al. 2021",
  source_url: "https://example.org/signature.centroid",
  licence: "GPL-3.0 (data file, fetched at runtime, not redistributed)",
  elapsed_seconds: 0.3,
  reused: false,
};

describe("cytokine helpers", () => {
  it("orders rows by the strongest |z| and keeps the range symmetric", () => {
    const h = cytokineHeatmap(RESULT)!;
    expect(h.y).toEqual(["TNFA", "IFNG", "IFN1"]); // strongest drawn on top (last)
    expect(h.zmax).toBe(48.6);
    expect(h.z[2]).toEqual([41.9, 48.6]);
    expect(comparisonText(RESULT)).toBe("STIM vs CTRL within each cell_type group");
    expect(comparisonText({ ...RESULT, mode: "groups", reference: "all", groupby: "leiden" })).toBe(
      "each leiden group vs all cells",
    );
  });
});

describe("CytokineResponse", () => {
  beforeEach(() => {
    getCytokineActivity.mockReset().mockResolvedValue(RESULT);
    getActivitySummary.mockReset().mockResolvedValue(null);
    useDatasetStore.setState({
      currentDatasetId: "d1",
      currentDataset: {
        id: "d1", name: "d1", filename: "", n_cells: 1000, n_genes: 900, status: "ready", available_embeddings: [],
        created_at: "",
        obs_columns: [
          { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
          { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "CD14 Mono", "pDC"] },
        ],
      } as DatasetInfo,
    });
  });

  it("scores STIM vs CTRL within cell types and shows the heatmap, top list and caption", async () => {
    render(<CytokineResponse groupByColumn="cell_type" setGroupByColumn={() => {}} />);
    fireEvent.click(screen.getByLabelText("Two conditions"));
    fireEvent.change(screen.getByLabelText("Condition column"), { target: { value: "stim" } });
    expect((screen.getByLabelText("Condition A") as HTMLSelectElement).value).toBe("STIM");
    expect((screen.getByLabelText("vs condition B") as HTMLSelectElement).value).toBe("CTRL");
    fireEvent.click(screen.getByRole("button", { name: "Score cytokine response" }));
    await waitFor(() => screen.getByTestId("plot"));
    expect(getCytokineActivity).toHaveBeenCalledWith("d1", {
      mode: "conditions", groupby: "cell_type", condition: "stim", a: "STIM", b: "CTRL",
    });
    expect(screen.getByTestId("plot").textContent).toBe("heatmap:TNFA,IFNG,IFN1:-48.6..48.6");
    expect(screen.getByTestId("cytokine-top").textContent).toContain("IFN1 (z 41.9), IFNG (z 25.5)");
    expect(screen.getByTestId("cytokine-top").textContent).toContain("IFN1 (z 48.6)");
    const caption = screen.getByTestId("cytokine-caption").textContent ?? "";
    expect(caption).toContain("STIM vs CTRL within each cell_type group");
    expect(caption).toContain("n = 2 groups; 4,669 of 4,800 signature genes.");
    expect(screen.getByText(/Left out: pDC/)).toBeTruthy();
  });

  it("needs a group column before scoring groups, and shows a server refusal", async () => {
    render(<CytokineResponse groupByColumn="" setGroupByColumn={() => {}} />);
    const run = screen.getByRole("button", { name: "Score cytokine response" }) as HTMLButtonElement;
    expect(run.disabled).toBe(true);
    getCytokineActivity.mockRejectedValueOnce(new Error("Only 120 of the 4800 CytoSig genes matched"));
    const { unmount } = render(<CytokineResponse groupByColumn="stim" setGroupByColumn={() => {}} />);
    fireEvent.click(screen.getAllByRole("button", { name: "Score cytokine response" })[1]!);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/Only 120/));
    expect(getCytokineActivity).toHaveBeenCalledWith("d1", { mode: "groups", groupby: "stim", reference: "rest" });
    unmount();
  });

  it("opens from the activity subtab", async () => {
    render(<UnifiedActivitySubtab onScore={vi.fn()} groupByColumn="" setGroupByColumn={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Cytokine response" }));
    expect(screen.getByRole("button", { name: "Score cytokine response" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "About cytokine response" })).toBeTruthy();
  });
});
