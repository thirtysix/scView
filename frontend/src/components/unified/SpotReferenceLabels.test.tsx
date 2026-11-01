import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import type { SpotInfo, SpotReferenceCheck, SpotSummary } from "@/api/spotDeconvolution";
import type { DatasetInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getSpotDeconvolution: vi.fn(),
  checkSpotReference: vi.fn(),
  checkSpotLabels: vi.fn(),
  getSpotShare: vi.fn(),
  runSpotDeconvolution: vi.fn(),
  getSpotCheck: vi.fn(),
  runSpotCheck: vi.fn(),
}));
vi.mock("@/api/spotDeconvolution", async (orig) => ({
  ...(await orig<typeof import("@/api/spotDeconvolution")>()),
  ...api,
}));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => vi.fn(async () => ({})) }));

import { SpotCompositionSubtab } from "./SpotCompositionSubtab";
import { SpotReferenceLabels } from "./SpotReferenceLabels";
import {
  NO_EDITS,
  type LabelEdits,
  editsBody,
  hasEdits,
  labelsUsed,
  likelySeparator,
  planLabels,
  planProblem,
  prefixGroups,
  pruneEdits,
  sharedName,
  withGroups,
  withTypeOut,
} from "@/lib/spotLabels";
import { useDatasetStore } from "@/stores/datasetStore";

// The real reference's 59 labels (cell2location mouse brain snRNA, refined annotation), with their cells.
const BRAIN: Record<string, number> = {
  Oligo_2: 10819, Ext_L56: 1422, Inh_4: 1389, Ext_L23: 1244, Micro: 1218, Ext_Thal_1: 1197, OPC_1: 1015, Ext_Pir: 954,
  Ext_L25: 922, Ext_Hpc_DG2: 915, Ext_L5_1: 893, Unk_1: 864, Inh_2: 839, Inh_3: 818, Inh_Pvalb: 804, Ext_Amy_2: 797,
  Inh_Meis2_3: 789, Ext_Thal_2: 671, Ext_Hpc_CA1: 671, Inh_Sst: 640, Oligo_1: 613, Inh_Meis2_2: 602, Inh_1: 573, Ext_L6: 568,
  Ext_Hpc_DG1: 554, Inh_Vip: 545, Ext_Amy_1: 519, Astro_AMY: 498, Inh_6: 467, Astro_HYPO: 451, Astro_CTX: 443, Ext_Unk_3: 416,
  Astro_THAL_med: 405, Inh_Meis2_1: 363, Ext_L6B: 356, Astro_THAL_lat: 332, Inh_Lamp5: 331, OPC_2: 300, Astro_AMY_CTX: 279,
  Ext_ClauPyr: 272, LowQ_2: 265, Astro_HPC: 261, Ext_Hpc_CA3: 232, Nb_1: 225, Astro_WM: 220, Ext_Unk_2: 205, Nb_2: 203,
  Inh_Meis2_4: 193, Ext_L5_2: 166, Ext_L5_3: 127, Inh_5: 118, LowQ_1: 110, Ext_Unk_1: 90, Unk_2: 85, Astro_STR: 81, Ext_Med: 70,
  Ext_Hpc_CA2: 48, Astro_THAL_hab: 43, Endo: 22,
};
const LABELS = Object.entries(BRAIN).map(([type, n_cells]) => ({ type, n_cells }));
const SUGGESTED = ["Ext_Unk_1", "Ext_Unk_2", "Ext_Unk_3", "LowQ_1", "LowQ_2", "Unk_1", "Unk_2"];

describe("spot label helpers", () => {
  it("counts types as the backend does: merge before the minimum, excluding wins", () => {
    const labels = [{ type: "A", n_cells: 100 }, { type: "B", n_cells: 20 }, { type: "C", n_cells: 10 }, { type: "D", n_cells: 40 }];
    const none = planLabels(labels, NO_EDITS, 25);
    expect(none.types.map((t) => t.type)).toEqual(["A", "D"]);
    expect(none.small.map((t) => t.type)).toEqual(["B", "C"]);
    // Two small labels merged pass the minimum together; an existing name is joined.
    const merged = planLabels(labels, { exclude: [], merge: { B: "BC", C: "BC", D: "A" } }, 25);
    expect(merged.types).toEqual([{ type: "A", n_cells: 140 }, { type: "BC", n_cells: 30 }]);
    expect(merged.nMerged).toBe(3);
    // Left out and merged: out. Merged into a left-out name: out too.
    const out = planLabels(labels, { exclude: ["D"], merge: { D: "A", B: "D" } }, 25);
    expect(out.types).toEqual([{ type: "A", n_cells: 100 }]);
    expect(out.nOut).toBe(2);
  });

  it("gives one reason per problem", () => {
    const labels = [{ type: "A", n_cells: 100 }, { type: "B", n_cells: 100 }, { type: "C", n_cells: 100 }];
    const reason = (e: LabelEdits, max = 2) => planProblem(planLabels(labels, e, 25), e, max, 25);
    expect(reason(NO_EDITS)).toBe("3 types, at most 2: merge or leave out at least 1 more.");
    expect(reason(NO_EDITS, 50)).toBeNull();
    expect(reason({ exclude: ["A", "B"], merge: {} }, 50)).toBe("1 type with 25+ cells: a run needs at least 2.");
    expect(reason({ exclude: [], merge: { A: "B", B: "C" } }, 50)).toBe(
      "A merges into B, which itself merges into C: merge both into the final name.",
    );
    // B is left out, so no chain: A goes with it and only C stays.
    expect(reason({ exclude: ["B"], merge: { A: "B", B: "C" } }, 50)).toBe("1 type with 25+ cells: a run needs at least 2.");
  });

  it("sends only what does something", () => {
    expect(hasEdits(NO_EDITS)).toBe(false);
    expect(hasEdits({ exclude: [], merge: { A: "  ", B: "B" } })).toBe(false);
    expect(hasEdits({ exclude: ["A"], merge: {} })).toBe(true);
    expect(editsBody({ exclude: ["A"], merge: { A: "X", B: " Y ", C: "", D: "D" } })).toEqual({
      exclude_labels: ["A"],
      merge_labels: { B: "Y" },
    });
  });

  it("drops choices the column no longer has", () => {
    const labels = [{ type: "A", n_cells: 1 }];
    const same = { exclude: ["A"], merge: { A: "X" } };
    expect(pruneEdits(same, labels)).toBe(same);
    expect(pruneEdits({ exclude: ["Gone", "X"], merge: { A: "X", Old: "Y" } }, labels)).toEqual({ exclude: ["X"], merge: { A: "X" } });
  });

  it("merges by shared name: on the real brain labels a trailing number gets under 50", () => {
    expect(likelySeparator(LABELS)).toBe("_");
    expect(sharedName("Inh_Meis2_3", "_", "number")).toBe("Inh_Meis2");
    expect(sharedName("Inh_Pvalb", "_", "number")).toBe("Inh_Pvalb");
    expect(sharedName("Inh_Pvalb", "_", "last")).toBe("Inh");
    expect(sharedName("Astro_THAL_med", "_", "first")).toBe("Astro");
    expect(sharedName("Micro", "_", "first")).toBe("Micro");
    expect(sharedName("_x", "_", "first")).toBe("_x");
    const before = planLabels(LABELS, NO_EDITS, 25);
    expect(before.types).toHaveLength(58);
    const counts = (cut: "number" | "last" | "first") => {
      const merge = withGroups({}, prefixGroups(LABELS, [], "_", cut));
      return planLabels(LABELS, { exclude: [], merge }, 25).types.length;
    };
    expect(counts("number")).toBe(39);
    expect(counts("last")).toBe(17);
    expect(counts("first")).toBe(9);
    const inh = prefixGroups(LABELS, [], "_", "number").find((g) => g.name === "Inh");
    expect(inh?.labels).toHaveLength(6);
    // What is left out is not offered for merging.
    const without = prefixGroups(LABELS, SUGGESTED, "_", "number");
    expect(without.find((g) => g.name === "Unk")).toBeUndefined();
    expect(withGroups({ Micro: "x" }, [{ name: "Micro", labels: ["Micro", "Micro_2"], n_cells: 2 }])).toEqual({ Micro_2: "Micro" });
  });

  it("says what a saved run used, nothing for an older one", () => {
    expect(labelsUsed({})).toBeNull();
    expect(labelsUsed({ excluded_labels: [], merged_labels: {} })).toBeNull();
    const used = labelsUsed({ excluded_labels: ["Unk_1"], n_excluded_cells: 864, merged_labels: { A_1: "A", A_2: "A", B_1: "B" } });
    expect(used?.text).toBe("left out Unk_1 (864 cells); 3 labels merged into 2");
    expect(used?.detail).toEqual(["A = A_1 + A_2", "B = B_1"]);
  });

  it("leaves a run's type out by the labels it stands for", () => {
    const labels = [{ type: "Unk_1", n_cells: 5 }, { type: "Unk_2", n_cells: 5 }, { type: "A", n_cells: 5 }, { type: "B", n_cells: 5 }];
    expect(withTypeOut({ merged_labels: { Unk_1: "Unk", Unk_2: "Unk" }, excluded_labels: ["B"] }, "Unk", labels)).toEqual({
      exclude: ["B", "Unk_1", "Unk_2"],
      merge: {},
    });
    expect(withTypeOut({}, "A", labels)).toEqual({ exclude: ["A"], merge: {} });
    // A label others joined: it and they all go.
    expect(withTypeOut({ merged_labels: { B: "A" } }, "A", labels)).toEqual({ exclude: ["B", "A"], merge: {} });
  });
});

function Harness({ labels = LABELS, suggested = SUGGESTED, start = NO_EDITS }: { labels?: typeof LABELS; suggested?: string[]; start?: LabelEdits }) {
  const [edits, setEdits] = useState(start);
  return (
    <>
      <SpotReferenceLabels labels={labels} suggested={suggested} edits={edits} onChange={setEdits} maxTypes={50} minCells={25} />
      <output data-testid="edits">{JSON.stringify(edits)}</output>
    </>
  );
}
const editsNow = () => JSON.parse(screen.getByTestId("edits").textContent ?? "{}") as LabelEdits;

describe("SpotReferenceLabels", () => {
  it("opens itself over the limit, marks the catch-alls without ticking them and scrolls inside", () => {
    render(<Harness />);
    expect(screen.getByTestId("spot-labels-count")).toHaveTextContent("58 types, at most 50");
    expect(screen.getByTestId("spot-labels-table").className).toMatch(/max-h-56 overflow-y-auto/);
    expect(screen.getAllByTestId("spot-label-suggested")).toHaveLength(7);
    const firstRows = within(screen.getByTestId("spot-labels-table")).getAllByRole("row").slice(1, 8);
    expect(firstRows.every((r) => within(r).queryByTestId("spot-label-suggested"))).toBe(true);
    expect(screen.getByLabelText("Leave out Unk_1")).not.toBeChecked();
    expect(editsNow()).toEqual(NO_EDITS);
    fireEvent.click(screen.getByLabelText("Leave out Unk_1"));
    expect(editsNow().exclude).toEqual(["Unk_1"]);
    expect(screen.getByTestId("spot-labels-count")).toHaveTextContent("57 types");
    expect(screen.getByLabelText("Merge Unk_1 into")).toBeDisabled();
  });

  it("stays closed under the limit until asked, with a word about the catch-alls", () => {
    const two = [{ type: "Unknown", n_cells: 100 }, { type: "B", n_cells: 100 }];
    render(<Harness labels={two} suggested={["Unknown"]} />);
    expect(screen.queryByTestId("spot-labels-table")).toBeNull();
    expect(screen.getByTestId("spot-labels")).toHaveTextContent("1 look like catch-alls");
    fireEvent.click(screen.getByRole("button", { name: "Reference labels" }));
    expect(screen.getByTestId("spot-labels-table")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Leave out Unknown"));
    expect(screen.getByTestId("spot-labels-count")).toHaveTextContent("1 type, at most 50");
  });

  it("merges a typed name and clears", () => {
    render(<Harness />);
    fireEvent.change(screen.getByLabelText("Merge Inh_1 into"), { target: { value: "Inh_2" } });
    expect(editsNow().merge).toEqual({ Inh_1: "Inh_2" });
    expect(screen.getByTestId("spot-labels-count")).toHaveTextContent("57 types");
    fireEvent.change(screen.getByLabelText("Merge Inh_1 into"), { target: { value: "" } });
    expect(editsNow().merge).toEqual({});
    fireEvent.click(screen.getByRole("button", { name: "Leave out the 7 marked" }));
    expect(editsNow().exclude).toEqual(SUGGESTED);
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(editsNow()).toEqual(NO_EDITS);
  });

  it("previews a coarsening before applying it, group by group", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "Preview" }));
    const preview = within(screen.getByTestId("spot-labels-preview"));
    expect(screen.getByTestId("spot-labels-preview-count")).toHaveTextContent("58 types would become 39.");
    expect(editsNow()).toEqual(NO_EDITS); // nothing yet
    fireEvent.click(preview.getByLabelText("Merge into Inh"));
    expect(screen.getByTestId("spot-labels-preview-count")).toHaveTextContent("58 types would become 44.");
    fireEvent.click(preview.getByRole("button", { name: /^Apply \d+ merges$/ }));
    expect(screen.queryByTestId("spot-labels-preview")).toBeNull();
    const merge = editsNow().merge;
    expect(merge.Ext_L5_2).toBe("Ext_L5");
    expect(merge.Inh_1).toBeUndefined();
    expect(screen.getByTestId("spot-labels-count")).toHaveTextContent("44 types");
  });

  it("says when no labels share a name", () => {
    render(<Harness labels={[{ type: "A", n_cells: 100 }, { type: "B", n_cells: 100 }]} suggested={[]} />);
    fireEvent.click(screen.getByRole("button", { name: "Reference labels" }));
    fireEvent.click(screen.getByRole("button", { name: "Preview" }));
    expect(screen.getByTestId("spot-labels-preview")).toHaveTextContent("No two labels share a name this way.");
  });
});

// --- In the subtab ---

const SPATIAL = {
  technology: "visium" as const, unit: "spot" as const, library_column: "library_id",
  libraries: [{ id: "lib1", n_obs: 4, images: {}, spot_diameter_fullres: 89, bounds: [0, 0, 1, 1] as [number, number, number, number] }],
};

function ds(id: string, spatial: DatasetInfo["spatial"] = null): DatasetInfo {
  return {
    id, name: id === "d1" ? "Brain" : "Atlas", filename: "", n_cells: 4, n_genes: 100, status: "ready",
    available_embeddings: [], created_at: "", spatial, obs_columns: [],
  };
}

const INFO: SpotInfo = {
  methods: ["rctd", "nnls"], method: "rctd", method_names: { rctd: "RCTD-style", nnls: "NNLS" }, min_type_cells: 25,
  max_types: 50, min_spot_umi: 100, min_umi_range: [1, 10000], min_umi_default: 100,
  counts: { below: [0, 0, ...Array(999).fill(1)], limit: 1000, n: 4, median: 900 }, min_shared_genes: 500,
  available: true, reason: null, saved: null, n_spots: 4, running: false, note: "",
};
const OVER: SpotReferenceCheck = {
  ok: false, reason: "'annotation_1' has 58 types with 25+ cells (at most 50): merge or leave out labels under Reference labels.",
  n_cells: 40572, n_genes: 31053, label_columns: [{ column: "annotation_1", n_labels: 59 }], celltype_column: "annotation_1",
  types: LABELS.map((l) => ({ ...l, small: l.n_cells < 25 })), n_small: 1, species: "mouse", dataset_species: "mouse",
  suggested_exclude: SUGGESTED, shared_genes_by_name: 19000, min_shared_genes: 500,
  estimate: { peak_bytes: 2 * 1024 ** 3, seconds: 900, seconds_range: [450, 1800], note: "about 15 min" },
};
const AFTER: SpotReferenceCheck = {
  ...OVER, ok: true, reason: null, excluded_labels: [], merged_labels: {},
  estimate: { peak_bytes: 2 * 1024 ** 3, seconds: 400, seconds_range: [200, 800], note: "about 7 min" },
};
const SMALL_CHECK: SpotReferenceCheck = {
  ...AFTER, types: [{ type: "Unk", n_cells: 200, small: false }, { type: "Oligo", n_cells: 900, small: false }, { type: "Astro", n_cells: 800, small: false }],
  n_small: 0, suggested_exclude: ["Unk"], label_columns: [{ column: "cell_type", n_labels: 3 }], celltype_column: "cell_type",
};
const RESULT: SpotSummary = {
  result_id: "r1", method: "rctd", method_name: "RCTD-style Poisson-lognormal fit", types: ["Astro", "Oligo", "Unk"],
  type_stats: [
    { type: "Astro", n_reference_cells: 800, cell_size: 5000, rna_mean: 0.3, rna_max: 0.9, cell_mean: 0.25, cell_max: 0.88, n_top: 1 },
    { type: "Oligo", n_reference_cells: 900, cell_size: 3000, rna_mean: 0.4, rna_max: 1, cell_mean: 0.45, cell_max: 1, n_top: 2 },
    { type: "Unk", n_reference_cells: 200, cell_size: 3000, rna_mean: 0.3, rna_max: 1, cell_mean: 0.3, cell_max: 1, n_top: 1 },
  ],
  small_types: [], reference: { id: "ref", name: "Atlas", celltype_column: "cell_type", n_reference_cells: 1900 },
  celltype_column: "cell_type", sources: { reference: "X", spots: "X", renamed: null }, s_e: 0.4,
  libraries: [{ id: "lib1", n_spots: 4, n_fitted: 4, s_g: 0.62, bulk: [0.3, 0.4, 0.3] }],
  n_shared_genes: 1900, n_genes: 1200, n_platform_genes: 3000, n_spots: 4, n_fitted: 4, n_skipped: 0, n_unconverged: 0,
  n_clipped: 0, min_umi: 100, cluster_column: null, clusters: { rna: {}, cells: {} },
  columns: { rna: "spot_deconv", cells: "spot_deconv_cells", top: "spot_deconv_top", top_share: "spot_deconv_top_share" },
  seconds: 12, notes: ["Unk holds 30% of the RNA here but is 5% of the reference's cells."], note: "",
  excluded_labels: [], merged_labels: {}, n_excluded_cells: 0, over_share: ["Unk"],
};

beforeEach(() => {
  vi.clearAllMocks();
  const spots = ds("d1", SPATIAL);
  useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: spots, availableDatasets: [spots, ds("ref")] });
  api.getSpotDeconvolution.mockResolvedValue(INFO);
  api.checkSpotReference.mockResolvedValue(OVER);
  api.checkSpotLabels.mockResolvedValue(AFTER);
  api.getSpotCheck.mockResolvedValue(null);
});

const fitButton = () => screen.getByRole("button", { name: /Fit spots/ });

describe("Reference labels in the subtab", () => {
  it("turns the 59-type refusal into a run after merging, and sends the labels", async () => {
    api.runSpotDeconvolution.mockResolvedValue({ ...RESULT, over_share: [] });
    render(<SpotCompositionSubtab />);
    fireEvent.change(await screen.findByLabelText("Reference dataset"), { target: { value: "ref" } });
    await screen.findByTestId("spot-labels");
    expect(screen.getByTestId("spot-run-blocked")).toHaveTextContent("58 types, at most 50: merge or leave out at least 8 more.");
    await waitFor(() => expect(fitButton()).toBeDisabled());
    expect(fitButton()).toHaveAttribute("title", expect.stringContaining("58 types"));
    expect(screen.getByTestId("spot-reference-check")).toHaveTextContent("about 15 min");
    expect(api.checkSpotLabels).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Preview" }));
    fireEvent.click(screen.getByRole("button", { name: /^Apply \d+ merges$/ }));
    expect(screen.queryByTestId("spot-run-blocked")).toBeNull();
    // Not until the server has seen these choices.
    expect(fitButton()).toBeDisabled();
    await waitFor(() => expect(api.checkSpotLabels).toHaveBeenCalled());
    await waitFor(() => expect(fitButton()).toBeEnabled());
    expect(screen.getByTestId("spot-reference-check")).toHaveTextContent("about 7 min");
    expect(screen.getByTestId("spot-reference-check")).toHaveTextContent("39 types with 25+ cells");
    fireEvent.click(fitButton());
    await waitFor(() => expect(api.runSpotDeconvolution).toHaveBeenCalled());
    const body = api.runSpotDeconvolution.mock.calls[0]?.[1];
    expect(body.merge_labels.Inh_1).toBe("Inh");
    expect(body.exclude_labels).toEqual([]);
  });

  it("with nothing ticked the request is as it always was", async () => {
    api.checkSpotReference.mockResolvedValue(SMALL_CHECK);
    api.runSpotDeconvolution.mockResolvedValue(RESULT);
    render(<SpotCompositionSubtab />);
    fireEvent.change(await screen.findByLabelText("Reference dataset"), { target: { value: "ref" } });
    await screen.findByTestId("spot-labels");
    await waitFor(() => expect(fitButton()).toBeEnabled());
    fireEvent.click(fitButton());
    await waitFor(() => expect(api.runSpotDeconvolution).toHaveBeenCalled());
    expect(api.runSpotDeconvolution.mock.calls[0]?.[1]).toEqual({
      reference_id: "ref", celltype_column: "cell_type", method: "rctd", min_umi: 100,
    });
    expect(api.checkSpotLabels).not.toHaveBeenCalled();
  });

  it("over-share: one click leaves the label out and runs again", async () => {
    api.getSpotDeconvolution.mockResolvedValue({ ...INFO, saved: RESULT });
    api.checkSpotReference.mockResolvedValue(SMALL_CHECK);
    api.runSpotDeconvolution.mockResolvedValue({
      ...RESULT, types: ["Astro", "Oligo"], over_share: [], excluded_labels: ["Unk"], n_excluded_cells: 200,
    });
    render(<SpotCompositionSubtab />);
    const button = await screen.findByRole("button", { name: "Leave out Unk and run again" });
    await waitFor(() => expect(button).toBeEnabled());
    expect(screen.queryByTestId("spot-labels-used")).toBeNull();
    fireEvent.click(button);
    await waitFor(() => expect(api.runSpotDeconvolution).toHaveBeenCalled());
    expect(api.runSpotDeconvolution.mock.calls[0]?.[1].exclude_labels).toEqual(["Unk"]);
    expect(await screen.findByTestId("spot-labels-used")).toHaveTextContent("left out Unk (200 cells)");
    expect(screen.queryByTestId("spot-over-share")).toBeNull();
  });

  it("a saved run's labels come back, and the self-check takes them", async () => {
    const saved = { ...RESULT, types: ["Astro", "Oligo"], over_share: [], excluded_labels: ["Unk"], n_excluded_cells: 200 };
    api.getSpotDeconvolution.mockResolvedValue({ ...INFO, saved });
    api.checkSpotReference.mockResolvedValue(SMALL_CHECK);
    api.checkSpotLabels.mockResolvedValue({ ...SMALL_CHECK, excluded_labels: ["Unk"] });
    api.runSpotCheck.mockResolvedValue({
      id: "c", types: ["Astro", "Oligo"], runs: [], n_spots: 10, cells_per_spot: [2, 10], depths: [5000],
      reference: { id: "ref", name: "Atlas", celltype_column: "cell_type" }, seconds: 1, note: "Synthetic.",
      excluded_labels: ["Unk"], merged_labels: {},
    });
    render(<SpotCompositionSubtab />);
    expect(await screen.findByTestId("spot-labels-used")).toHaveTextContent("left out Unk");
    const out = await screen.findByLabelText("Leave out Unk");
    expect(out).toBeChecked();
    const run = await screen.findByRole("button", { name: /Run self-check/ });
    await waitFor(() => expect(run).toBeEnabled());
    fireEvent.click(run);
    await waitFor(() => expect(api.runSpotCheck).toHaveBeenCalled());
    expect(api.runSpotCheck.mock.calls[0]?.[1]).toMatchObject({ exclude_labels: ["Unk"], merge_labels: {} });
    expect(await screen.findByTestId("spot-check")).toHaveTextContent("left out Unk");
  });

  it("an older saved run with no such fields opens", async () => {
    const old: SpotSummary = { ...RESULT };
    delete old.excluded_labels;
    delete old.merged_labels;
    delete old.n_excluded_cells;
    delete old.over_share;
    api.getSpotDeconvolution.mockResolvedValue({ ...INFO, saved: old });
    api.checkSpotReference.mockResolvedValue(SMALL_CHECK);
    render(<SpotCompositionSubtab />);
    expect(await screen.findByTestId("spot-composition-result")).toHaveTextContent("3 types from");
    expect(screen.queryByTestId("spot-labels-used")).toBeNull();
    expect(screen.queryByTestId("spot-over-share")).toBeNull();
  });

  it("shows the server's sentence for choices it refuses and keeps Run off", async () => {
    api.checkSpotReference.mockResolvedValue(SMALL_CHECK);
    api.checkSpotLabels.mockRejectedValue(new Error("API error 422: 'cell_type' has no label 'Nope': pick from its labels."));
    render(<SpotCompositionSubtab />);
    fireEvent.change(await screen.findByLabelText("Reference dataset"), { target: { value: "ref" } });
    await screen.findByTestId("spot-labels");
    fireEvent.click(screen.getByRole("button", { name: "Reference labels" }));
    fireEvent.click(screen.getByLabelText("Leave out Astro"));
    expect(await screen.findByText(/has no label 'Nope'/)).toBeInTheDocument();
    expect(fitButton()).toBeDisabled();
  });
});
