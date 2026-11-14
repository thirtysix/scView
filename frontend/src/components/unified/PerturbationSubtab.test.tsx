import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { AssignmentPreview, MixscapeResult, PerturbationSummary } from "@/api/perturbation";
import type { DatasetInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getPerturbation: vi.fn(),
  previewAssignment: vi.fn(),
  saveAssignment: vi.fn(),
  runMixscape: vi.fn(),
}));
vi.mock("@/api/perturbation", async (orig) => ({ ...(await orig<typeof import("@/api/perturbation")>()), ...api }));
const refresh = vi.hoisted(() => vi.fn());
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; name?: string }[] }) => (
    <div data-testid="plot">{data.map((t) => `${t.type}:${t.name ?? ""}`).join(";")}</div>
  ),
}));

import { PerturbationSubtab } from "./PerturbationSubtab";
import { hoverTip, unhover } from "@/test/hoverTip";
import { useDatasetStore } from "@/stores/datasetStore";
import { useDESettingsStore } from "@/stores/deSettingsStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const BASE: PerturbationSummary = {
  n_cells: 400,
  guides: ["Ag1", "Bg1", "NTg1"],
  has_counts: true,
  has_cellranger: false,
  columns: [],
  obs_columns: ["rep"],
  suggested: { source: "counts", column: null },
  defaults: { min_umi: 5, min_ratio: 2 },
  sources: [],
  call: null,
  mixscape: null,
  mixscape_saved: null,
  mixscape_defaults: {},
  running: false,
};

const PREVIEW: AssignmentPreview = {
  source: "counts", source_label: "guide counts", column: null, n_cells: 400, guides: ["Ag1", "Bg1", "NTg1"],
  mapping: { Ag1: "A", Bg1: "B", NTg1: "NT" }, auto_mapping: { Ag1: "A", Bg1: "B", NTg1: "NT" }, control: "NT",
  targets: ["A", "B"], counts: { assigned: 390, control: 150, multiple: 6, none: 4 },
  per_target: [
    { target: "NT", n_cells: 150, control: true, guides: ["NTg1"] },
    { target: "A", n_cells: 140, control: false, guides: ["Ag1"] },
    { target: "B", n_cells: 100, control: false, guides: ["Bg1"] },
  ],
};

const RESULT: MixscapeResult = {
  params: {}, labels: "perturbation", control: "NT", split_by: null, n_cells: 390, n_control: 150, n_targets: 2,
  n_genes: 60, gene_source: "60 variable genes", expression_source: "log-normalised X",
  targets: [
    { target: "A", n_cells: 140, n_ko: 98, n_np: 42, ko_fraction: 0.7, n_de_genes: 12, rounds: 2,
      top_de_genes: [{ gene: "G1", logfc: -3.1, pval_adj: 1e-20 }],
      posterior_hist: { edges: [0, 0.5, 1], ko: [0, 98], np: [42, 0] } },
    { target: "B", n_cells: 100, n_ko: 0, n_np: 100, ko_fraction: 0, n_de_genes: 0, rounds: 0,
      top_de_genes: [], note: "fewer than 5 DE genes: all NP", posterior_hist: { edges: [0, 0.5, 1], ko: [0, 0], np: [100, 0] } },
  ],
  warnings: [], lda: { x: [0, 1, 2], y: [0, 1, 0], label: ["NT", "A", "A"], n_cells: 248, n_shown: 3, dims: 1 },
  seconds: 1, created: "", n_cells_total: 400,
  summary: { n_target_cells: 240, n_ko: 98, ko_fraction: 98 / 240, n_targets_with_ko: 1 },
};

const CALL = {
  source: "counts" as const, source_label: "guide counts", column: null, params: { min_umi: 5, min_ratio: 2 },
  control: "NT", mapping: {}, columns: { guide_call: "guide_call", perturbation: "perturbation", n_guides: "n_guides" },
  when: "", n_cells_at_call: 400, counts: PREVIEW.counts, n_targets: 2, per_target: [],
};

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
  refresh.mockReset().mockResolvedValue({});
  api.previewAssignment.mockResolvedValue(PREVIEW);
  useSettingsStore.setState({ colorBy: "" });
  useDESettingsStore.getState().reset();
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: {
      id: "d1", name: "d1", filename: "", n_cells: 400, n_genes: 60, status: "ready", available_embeddings: [], created_at: "",
      obs_columns: [{ name: "rep", dtype: "category", n_unique: 2, values: ["r1", "r2"] }],
      perturbation: { n_guides: 3, cellranger: false, assigned: false, columns: [] },
    } as DatasetInfo,
  });
});

describe("PerturbationSubtab", () => {
  it("keeps each label on its control, and explains it on hover", async () => {
    api.getPerturbation.mockResolvedValue(BASE);
    render(<PerturbationSubtab />);
    await screen.findByText(/390 cells with one target/);
    expect(screen.getByLabelText("Min UMIs")).toHaveValue(5);
    expect(screen.getByLabelText("Top / second ≥")).toHaveValue(2);
    expect(await hoverTip(screen.getByText("Top / second ≥"))).toMatch(/0 turns this off, so every guide counts/);
    unhover(screen.getByText("Top / second ≥"));
    expect(await hoverTip(screen.getByText("Perturbation (CRISPR screen)"))).toMatch(/In a pooled screen each cell carries a guide RNA/);
  });

  it("previews the assignment from the guide counts and saves it", async () => {
    api.getPerturbation.mockResolvedValueOnce(BASE).mockResolvedValue({ ...BASE, call: CALL });
    api.saveAssignment.mockResolvedValue({ ...BASE, call: CALL });
    render(<PerturbationSubtab />);
    expect(await screen.findByText(/390 cells with one target/)).toBeInTheDocument();
    expect(screen.getByLabelText("Guide source")).toHaveValue("counts");
    expect(screen.getByLabelText("Control label")).toHaveValue("NT");
    fireEvent.click(screen.getByRole("button", { name: "Save assignment" }));
    await waitFor(() => expect(api.saveAssignment).toHaveBeenCalled());
    expect(api.saveAssignment.mock.calls[0]?.[1]).toMatchObject({ source: "counts", min_umi: 5, min_ratio: 2, control: "NT", mapping: {} });
    await waitFor(() => expect(useSettingsStore.getState().colorBy).toBe("perturbation"));
    expect(refresh).toHaveBeenCalledWith("d1", "perturbation");
  });

  it("shows a saved run's Within choice, so the control matches the result", async () => {
    const saved = { ...RESULT, split_by: "rep", lda: undefined };
    api.getPerturbation.mockResolvedValue({ ...BASE, call: CALL, mixscape_saved: saved });
    render(<PerturbationSubtab />);
    await waitFor(() => expect(screen.getByLabelText("Split by")).toHaveValue("rep"));
    expect(screen.getByLabelText("DE genes from")).toHaveValue("variable"); // a run before 0.78
  });

  it("runs Mixscape, shows the per-target table, the posterior and hands a target to DE", async () => {
    api.getPerturbation.mockResolvedValueOnce({ ...BASE, call: CALL }).mockResolvedValue({ ...BASE, call: CALL, mixscape: RESULT });
    api.runMixscape.mockResolvedValue(RESULT);
    render(<PerturbationSubtab />);
    fireEvent.change(await screen.findByLabelText("Split by"), { target: { value: "rep" } });
    expect(screen.getByLabelText("DE genes from")).toHaveValue("all");
    fireEvent.click(screen.getByRole("button", { name: "Run Mixscape" }));
    await waitFor(() => expect(api.runMixscape).toHaveBeenCalled());
    expect(api.runMixscape.mock.calls[0]?.[1]).toEqual({ split_by: "rep", de_genes: "all" });
    const row = await screen.findByTestId("mixscape-row-A");
    expect(within(row).getByText("70%")).toBeInTheDocument();
    expect(screen.getByTestId("mixscape-row-B")).toHaveTextContent("fewer than 5 DE genes");
    expect(useSettingsStore.getState().colorBy).toBe("mixscape_class");
    fireEvent.click(row);
    expect(screen.getByTestId("mixscape-detail")).toHaveTextContent("G1 (-3.10)");
    expect(screen.getAllByTestId("plot")[0]).toHaveTextContent("bar:NP;bar:KO");
    expect(screen.getByTestId("mixscape-lda")).toHaveTextContent("scattergl:NT (control);scattergl:A KO");
    fireEvent.click(within(row).getByRole("button", { name: "DE" }));
    expect(useDESettingsStore.getState()).toMatchObject({ mode: "groups", splitBy: "mixscape_class", groupA: "A KO", groupB: "NT" });
    expect(useUnifiedViewStore.getState().activeSubtab).toBe("de");
  });

  it("shows a target's KO share per guide and per replicate", async () => {
    const withTables: MixscapeResult = {
      ...RESULT,
      guide_column: "guide_ID",
      replicate_column: "rep",
      per_guide: [
        { target: "A", guide: "Ag1", n_cells: 70, n_ko: 63, ko_fraction: 0.9 },
        { target: "A", guide: "Ag2", n_cells: 70, n_ko: 35, ko_fraction: 0.5 },
        { target: "B", guide: "Bg1", n_cells: 100, n_ko: 0, ko_fraction: 0 },
      ],
      per_replicate: [
        { target: "A", replicate: "r1", n_cells: 70, n_ko: 49, ko_fraction: 0.7 },
        { target: "A", replicate: "r2", n_cells: 70, n_ko: 49, ko_fraction: 0.7 },
      ],
    };
    api.getPerturbation.mockResolvedValue({ ...BASE, call: CALL, mixscape: withTables });
    render(<PerturbationSubtab />);
    fireEvent.click(await screen.findByTestId("mixscape-row-A"));
    const guides = screen.getByTestId("mixscape-per-guide");
    expect(guides).toHaveTextContent("Per guide (guide_ID)");
    expect(guides).toHaveTextContent(/Ag1\s*70\s*90%/);
    expect(guides).toHaveTextContent(/Ag2\s*70\s*50%/);
    expect(guides).not.toHaveTextContent("Bg1");
    expect(screen.getByTestId("mixscape-per-replicate")).toHaveTextContent(/Per rep.*r1\s*70\s*70%/);
    fireEvent.click(screen.getByTestId("mixscape-row-B"));
    expect(screen.queryByTestId("mixscape-per-replicate")).toBeNull(); // no rows for B
  });

  it("shows the protein check: flags, a protein chooser, the plot and the table", async () => {
    const row = (target: string, protein: string, diff: number, ko: number, sig: boolean, flag: boolean) => ({
      target, protein, ko_fraction: ko, unit: "replicate" as const, significant: sig, flag,
      all: { n: 100, diff, p: 1e-10, padj: 1e-9 }, ko: ko ? { n: 70, diff: diff * 1.3, p: 1e-12, padj: 1e-11 } : null,
      np: { n: 30, diff: diff / 4, p: 0.2, padj: 0.3 },
      replicates: [{ replicate: "r1", n: 33, diff }, { replicate: "r2", n: 33, diff }, { replicate: "r3", n: 34, diff }],
      n_replicates: 3, same_sign: 3, p_replicate: 0.03,
    });
    const withProtein: MixscapeResult = {
      ...RESULT,
      protein: {
        layer: "protein_clr", proteins: ["PDL1", "CD86"], n_proteins: 2, control: "NT", n_control: 150,
        replicate_column: "rep", min_cells: 5, ko_share_none: 0.1, alpha: 0.05, n_pairs: 4,
        rows: [row("A", "PDL1", -0.4, 0.7, true, false), row("B", "PDL1", -0.2, 0, true, true),
          row("A", "CD86", 0.01, 0.7, false, false), row("B", "CD86", 0.02, 0, false, false)],
        flags: [{ target: "B", protein: "PDL1", diff: -0.2, unit: "replicate", ko_fraction: 0,
          text: "B: PDL1 -0.20 (CLR) in all 100 guide cells vs the controls" }],
        notes: ["CLR per cell over 2 proteins: read with care."], default_protein: "PDL1",
      },
    };
    api.getPerturbation.mockResolvedValue({ ...BASE, call: CALL, mixscape: withProtein });
    render(<PerturbationSubtab />);
    const box = await screen.findByTestId("mixscape-protein");
    expect(within(box).getByTestId("mixscape-protein-flags")).toHaveTextContent("B: PDL1 -0.20 (CLR)");
    expect(within(box).getByTestId("plot")).toHaveTextContent("All cells: no RNA KO class");
    expect(within(box).getByTestId("plot")).toHaveTextContent("Each replicate");
    const table = within(box).getByTestId("mixscape-protein-table");
    expect(table.querySelectorAll("tbody tr")[0]).toHaveTextContent(/^A\s*70%\s*-0\.40/); // most negative first
    expect(within(box).getByTestId("mixscape-protein-row-B")).toHaveTextContent(/protein only.*-0\.20.*3\/3/);
    expect(box).toHaveTextContent("CLR per cell over 2 proteins");
    fireEvent.click(within(box).getByRole("button", { name: "CD86" }));
    expect(within(box).getByTestId("mixscape-protein-row-A")).toHaveTextContent("+0.01");
    fireEvent.click(within(box).getByTestId("mixscape-protein-row-B")); // opens the target's detail
    expect(screen.getByTestId("mixscape-detail")).toBeInTheDocument();
  });

  it("has no protein section without a protein check", async () => {
    api.getPerturbation.mockResolvedValue({ ...BASE, call: CALL, mixscape: RESULT });
    render(<PerturbationSubtab />);
    await screen.findByTestId("mixscape-row-A");
    expect(screen.queryByTestId("mixscape-protein")).toBeNull();
  });

  it("says how to get guides when there are none", async () => {
    api.getPerturbation.mockResolvedValue({ ...BASE, has_counts: false, guides: [], suggested: { source: null, column: null } });
    render(<PerturbationSubtab />);
    expect(await screen.findByRole("note")).toHaveTextContent("protospacer_calls_per_cell.csv");
    expect(api.previewAssignment).not.toHaveBeenCalled();
  });
});
