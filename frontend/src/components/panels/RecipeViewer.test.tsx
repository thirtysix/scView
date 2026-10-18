import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

const prov = vi.hoisted(() => ({ getRecipe: vi.fn() }));
vi.mock("@/api/provenance", () => prov);
const exp = vi.hoisted(() => ({ downloadRecipeFile: vi.fn() }));
vi.mock("@/api/export", () => exp);

import { RecipeViewer } from "./RecipeViewer";
import { useViewStore } from "@/stores/viewStore";
import { replayCounts, replayStatus } from "@/lib/replayStatus";

const RECIPE = {
  source: { original_filename: "kang.h5ad" },
  recipe: [
    {
      commit_id: "a1", step: "normalization", params: { target_sum: 10000 }, tool: "scanpy.pp.normalize_total",
      timestamp: "2026-10-18T10:00:00+00:00", scview_version: "0.21.0", effect: { n_cells: 13666, n_genes: 13873 },
      replay: { status: "exact", note: "" },
    },
    {
      commit_id: "a2", step: "enrichment", params: { enrichment_columns: ["cluster"] }, tool: "scview.enrichment",
      timestamp: "2026-10-18T10:01:00+00:00", scview_version: "0.21.0",
      replay: { status: "approximate", note: "MSigDB over-representation: not replayed (needs the gene sets)" },
    },
    {
      commit_id: "a3", step: "label_cells", params: { column: "mine", n_cells: 3 }, tool: "scview.label_selection",
      timestamp: "2026-10-18T10:02:00+00:00", scview_version: "0.21.0",
      replay: { status: "not_replayable", note: "manual labels: cells not recorded" },
    },
  ],
};

describe("RecipeViewer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useViewStore.setState({ copilotOpen: false });
  });

  it("summarises how the steps replay and lists them with params, time and version", async () => {
    prov.getRecipe.mockResolvedValue(RECIPE);
    render(<RecipeViewer datasetId="abc" />);
    const counts = await screen.findByTestId("replay-counts");
    expect(counts).toHaveTextContent("1 exact");
    expect(counts).toHaveTextContent("1 approximate");
    expect(counts).toHaveTextContent("1 not replayable");
    expect(prov.getRecipe).toHaveBeenCalledWith("abc");

    fireEvent.click(screen.getByRole("button", { name: /Recipe \(3 steps\)/ }));
    const list = screen.getByRole("list", { name: "Recipe steps" });
    const rows = within(list).getAllByRole("listitem");
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveTextContent("Normalization");
    expect(rows[0]).toHaveTextContent("target_sum=10000");
    expect(rows[0]).toHaveTextContent("scView 0.21.0");
    expect(rows[0]).toHaveTextContent("13,666 cells");
    expect(rows[1]).toHaveTextContent("needs the gene sets");
    expect(rows[2]).toHaveTextContent("not replayable");
  });

  it("downloads the script and the manifest, and shows a failure", async () => {
    prov.getRecipe.mockResolvedValue(RECIPE);
    exp.downloadRecipeFile.mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error("Dataset not found."));
    render(<RecipeViewer datasetId="abc" />);
    fireEvent.click(await screen.findByRole("button", { name: /Replay script/ }));
    await waitFor(() => expect(exp.downloadRecipeFile).toHaveBeenCalledWith("abc", "script"));
    await waitFor(() => expect(screen.getByRole("button", { name: /Run manifest/ })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: /Run manifest/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Dataset not found.");
    expect(exp.downloadRecipeFile).toHaveBeenLastCalledWith("abc", "manifest");
  });

  it("opens the co-pilot for a methods paragraph", async () => {
    prov.getRecipe.mockResolvedValue(RECIPE);
    render(<RecipeViewer datasetId="abc" />);
    fireEvent.click(await screen.findByRole("button", { name: /Methods text/ }));
    expect(useViewStore.getState().copilotOpen).toBe(true);
  });

  it("says when nothing is recorded, and disables the script", async () => {
    prov.getRecipe.mockResolvedValue({ source: {}, recipe: [] });
    render(<RecipeViewer datasetId="abc" />);
    expect(await screen.findByText(/No steps are recorded yet/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Replay script/ })).toBeDisabled();
  });
});

describe("replayStatus", () => {
  it("treats a missing or newer status as unknown and counts in a fixed order", () => {
    const steps = [
      { step: "a", params: {}, replay: { status: "approximate" as const, note: "" } },
      { step: "b", params: {} },
      { step: "c", params: {}, replay: { status: "exact" as const, note: "" } },
      { step: "d", params: {}, replay: { status: "brand_new" as never, note: "" } },
    ];
    expect(replayStatus(steps[1]!)).toBe("unknown");
    expect(replayStatus(steps[3]!)).toBe("unknown");
    expect(replayCounts(steps)).toEqual([["exact", 1], ["approximate", 1], ["unknown", 2]]);
  });
});
