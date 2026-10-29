import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { TfMotifResult, TfMotifRow, TfMotifStatus } from "@/api/tfMotifs";
import type { DatasetInfo } from "@/api/types";
import { formatP, sortTfRows, tfCsvRows, tfGene } from "@/lib/tfMotifs";
import { stepLabel } from "@/lib/stepLabel";

const { getTfMotifStatus, getLastTfMotifs, prepareTfMotifs, runTfMotifs, getActivityCells, getActivitySummary } =
  vi.hoisted(() => ({
    getTfMotifStatus: vi.fn(),
    getLastTfMotifs: vi.fn(),
    prepareTfMotifs: vi.fn(),
    runTfMotifs: vi.fn(),
    getActivityCells: vi.fn(),
    getActivitySummary: vi.fn(),
  }));
vi.mock("@/api/tfMotifs", async (orig) => ({
  ...(await orig<typeof import("@/api/tfMotifs")>()),
  getTfMotifStatus,
  getLastTfMotifs,
  prepareTfMotifs,
  runTfMotifs,
}));
vi.mock("@/api/activity", async (orig) => ({
  ...(await orig<typeof import("@/api/activity")>()),
  getActivityCells,
  getActivitySummary,
}));
vi.mock("@/components/plots/Plot", () => ({ default: () => <div data-testid="plot" /> }));

import { TfMotifCandidates } from "./TfMotifCandidates";
import { UnifiedActivitySubtab } from "./UnifiedActivitySubtab";
import { useDatasetStore } from "@/stores/datasetStore";

const ASM = {
  assembly: "hg38" as const, species: "human" as const, label: "human (hg38)", estimate_seconds: 120,
  building: false, progress: null,
  files: [{ name: "JASPAR", url: "u", sha256: "s", bytes: 20_000_000, downloaded: false }],
};

function status(prepared: boolean): TfMotifStatus {
  return {
    assemblies: [{ ...ASM, prepared }], auto_download: true, citation: "JASPAR 2024 (Rauluseviciute et al. 2024)",
    licence: "CC BY 4.0", caveats: "Promoters only. A motif in a promoter is not binding: each row is a candidate.",
    p_threshold: 1e-4, promoter_bp: 1000, dataset: { species: "human", assembly: "hg38", prepared },
  };
}

function row(over: Partial<TfMotifRow>): TfMotifRow {
  return {
    motif_id: "MA0001.1", motif: "JUNB", tfs: ["JUNB"], k: 33, n: 98, K: 1448, N: 11521, odds_ratio: 3.59,
    p: 4.7e-8, padj: 1.8e-5, gc: 0.4, gc_rich: false, overlap: ["FOS", "IER2"], n_overlap: 2,
    tf_expressed: { gene: "JUNB", pct: 0.71, logfc: 1.2, expressed: true }, activity_agrees: null, ...over,
  };
}

const RESULT: TfMotifResult = {
  result_id: "r1", column: "cell_type", assembly: "hg38", species: "human", n_markers_requested: 100, N: 11521,
  n_motifs: 879, n_motifs_tested: 759, n_unreachable: 120,
  groups: [
    {
      group: "CD14 Mono", n_markers: 100, n: 98, n_cells: 2100, n_tested: 759, n_significant: 2, n_reported: 3,
      rows: [
        row({}),
        row({ motif_id: "MA0002.1", motif: "FOS::JUND", tfs: ["FOS", "JUND"], k: 31, odds_ratio: 3.04, p: 2.2e-6, padj: 4.1e-4,
              tf_expressed: { gene: "FOS", pct: 0.18, logfc: null, expressed: true } }),
        row({ motif_id: "MA0003.1", motif: "SP1", tfs: ["SP1"], k: 40, odds_ratio: 1.2, p: 0.02, padj: 0.4, gc_rich: true,
              tf_expressed: null }),
      ],
    },
    { group: "B", n_markers: 50, n: 2, n_cells: 900, n_tested: 759, n_significant: 0, n_reported: 0, rows: [],
      note: "2 of its 50 marker genes have a promoter here; at least 3 are needed." },
  ],
  p_threshold: 1e-4, pseudocount: 0.8, min_overlap: 3, promoter_bp: 1000,
  activity: { available: false, hint: "Score TF activity once (Activity > TFs (CollecTRI)) and run again.", network: "CollecTRI" },
  matrix_source: "log-normalised X", citation: "JASPAR 2024 (Rauluseviciute et al. 2024)", licence: "CC BY 4.0",
  caveats: "Promoters only. A motif in a promoter is not binding: each row is a candidate.", elapsed_seconds: 0.7,
  created_at: "",
};

describe("tfMotifs helpers", () => {
  it("sorts, formats and writes CSV rows", () => {
    const rows = RESULT.groups[0]!.rows;
    expect(sortTfRows(rows, "padj").map((r) => r.motif)).toEqual(["JUNB", "FOS::JUND", "SP1"]);
    expect(sortTfRows(rows, "k").map((r) => r.motif)).toEqual(["SP1", "JUNB", "FOS::JUND"]);
    expect(sortTfRows(rows, "odds_ratio")[0]!.motif).toBe("JUNB");
    expect(formatP(1.8e-5)).toBe("1.8e-5");
    expect(formatP(0.4)).toBe("0.40");
    expect(tfGene(rows[1]!)).toBe("FOS");
    expect(tfGene(rows[2]!)).toBe("SP1");
    const csv = tfCsvRows(RESULT);
    expect(csv).toHaveLength(3);
    expect(csv[0]!.slice(0, 3)).toEqual(["CD14 Mono", "MA0001.1", "JUNB"]);
    expect(csv[2]![10]).toBe("yes"); // gc_rich
    expect(stepLabel("tf_motif_enrichment")).toBe("Upstream TF candidates (promoter motifs)");
  });
});

describe("TfMotifCandidates", () => {
  beforeEach(() => {
    for (const f of [getTfMotifStatus, getLastTfMotifs, prepareTfMotifs, runTfMotifs, getActivityCells, getActivitySummary]) f.mockReset();
    getLastTfMotifs.mockResolvedValue(null);
    getActivitySummary.mockResolvedValue(null);
    useDatasetStore.setState({
      currentDatasetId: "d1",
      currentDataset: {
        id: "d1", name: "d1", filename: "", n_cells: 3000, n_genes: 900, status: "ready", available_embeddings: [],
        created_at: "",
        obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "CD14 Mono"] }],
      } as DatasetInfo,
    });
  });

  it("prepares the motif table with its estimate and progress, then can run", async () => {
    getTfMotifStatus.mockResolvedValueOnce(status(false)).mockResolvedValueOnce(status(true));
    let finish: () => void = () => {};
    prepareTfMotifs.mockImplementation((_body, onProgress) => {
      onProgress({ stage: "scan", message: "Scanned 9,741 of 18,698 promoters", done: 9741, total: 18698 });
      return new Promise((resolve) => {
        finish = () => resolve({ assembly: "hg38", meta: {} });
      });
    });
    render(<TfMotifCandidates groupByColumn="cell_type" setGroupByColumn={() => {}} onScore={vi.fn()} />);
    await waitFor(() => screen.getByRole("button", { name: "Prepare motif table" }));
    expect(screen.getByText(/download of 20 MB and scan of about 2 min/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Prepare motif table" }));
    await waitFor(() => screen.getByTestId("tf-motif-progress"));
    expect(screen.getByText("Scanned 9,741 of 18,698 promoters")).toBeTruthy();
    expect(prepareTfMotifs).toHaveBeenCalledWith({ species: "human", assembly: "hg38" }, expect.any(Function));
    finish();
    await waitFor(() => screen.getByRole("button", { name: "Find candidate TFs" }));
  });

  it("runs, shows the table with caps, flags and caveats, and colours by expression", async () => {
    getTfMotifStatus.mockResolvedValue(status(true));
    runTfMotifs.mockResolvedValue(RESULT);
    const onGeneClick = vi.fn();
    render(<TfMotifCandidates groupByColumn="cell_type" setGroupByColumn={() => {}} onScore={vi.fn()} onGeneClick={onGeneClick} />);
    fireEvent.click(await screen.findByRole("button", { name: "Find candidate TFs" }));
    expect(runTfMotifs).toHaveBeenCalledWith("d1", { column: "cell_type" });
    const table = await screen.findByTestId("tf-motif-table");
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(4); // header + 3
    expect(rows[1]!.textContent).toContain("JUNB");
    expect(rows[1]!.textContent).toContain("JUNB 71% (+1.2)");
    expect(rows[3]!.textContent).toContain("GC-rich");
    expect(rows[3]!.textContent).toContain("not measured");
    expect(screen.getByTestId("tf-motif-caption").textContent).toContain(
      "n = 98 of 100 markers with a promoter; 11,521 measured genes with a promoter; 2,100 cells.",
    );
    expect(screen.getByTestId("tf-motif-caveats").textContent).toContain("candidate");
    expect(screen.getByTestId("tf-activity-hint").textContent).toContain("CollecTRI");
    expect(screen.getByText(/Rauluseviciute/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "k / n" }));
    expect(within(table).getAllByRole("row")[1]!.textContent).toContain("SP1");

    fireEvent.click(within(table).getAllByRole("row")[2]!); // JUNB
    await waitFor(() => expect(onGeneClick).toHaveBeenCalledWith("JUNB"));
    expect(getActivityCells).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText(/Group \(cell_type\)/), { target: { value: "B" } });
    expect(screen.getByText(/at least 3 are needed/)).toBeTruthy();
    expect(screen.getByText(/No motif has 3 or more/)).toBeTruthy();
  });

  it("colours by CollecTRI activity when the run has it", async () => {
    getTfMotifStatus.mockResolvedValue(status(true));
    getLastTfMotifs.mockResolvedValue({
      ...RESULT, activity: { available: true, hint: null, network: "CollecTRI" },
      groups: [{ ...RESULT.groups[0]!, rows: [row({ activity_agrees: true })] }],
    });
    getActivityCells.mockResolvedValue({ network: "collectri", source: "JUNB", n_targets: 40, scores: [1, 2] });
    const onScore = vi.fn();
    const onGeneClick = vi.fn();
    render(<TfMotifCandidates groupByColumn="" setGroupByColumn={() => {}} onScore={onScore} onGeneClick={onGeneClick} />);
    const table = await screen.findByTestId("tf-motif-table");
    expect(within(table).getAllByRole("row")[1]!.textContent).toContain("yes");
    fireEvent.click(within(table).getAllByRole("row")[1]!);
    await waitFor(() => expect(onScore).toHaveBeenCalled());
    expect(getActivityCells).toHaveBeenCalledWith("d1", "collectri", "JUNB");
    expect(onScore.mock.calls[0]![1]).toBe("JUNB (TF activity)");
    expect(onGeneClick).not.toHaveBeenCalled();
  });

  it("says when the species is unknown, and shows a server refusal", async () => {
    getTfMotifStatus.mockResolvedValue({ ...status(true), dataset: { species: "unknown", assembly: null, prepared: false } });
    render(<TfMotifCandidates groupByColumn="cell_type" setGroupByColumn={() => {}} onScore={vi.fn()} />);
    await waitFor(() => screen.getByText(/species of this dataset's genes is unknown/));
    expect(screen.queryByRole("button", { name: "Find candidate TFs" })).toBeNull();
  });

  it("shows a run refusal", async () => {
    getTfMotifStatus.mockResolvedValue(status(true));
    runTfMotifs.mockRejectedValue(new Error('API error 422: {"detail": "No marker genes are stored for \'cell_type\'"}'));
    render(<TfMotifCandidates groupByColumn="cell_type" setGroupByColumn={() => {}} onScore={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Find candidate TFs" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/No marker genes are stored/));
  });

  it("opens from the activity subtab", async () => {
    getTfMotifStatus.mockResolvedValue(status(false));
    render(<UnifiedActivitySubtab onScore={vi.fn()} groupByColumn="" setGroupByColumn={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Upstream TF candidates" }));
    expect(await screen.findByRole("button", { name: "Prepare motif table" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "About upstream TF candidates" })).toBeTruthy();
  });
});
