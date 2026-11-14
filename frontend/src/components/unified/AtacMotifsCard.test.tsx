import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ChromVarGroups, MotifEnrichment, MotifState } from "@/api/atacMotifs";

const api = vi.hoisted(() => ({
  getMotifState: vi.fn(),
  scanPeaks: vi.fn(),
  cancelScan: vi.fn(),
  runEnrichment: vi.fn(),
  getLatestEnrichment: vi.fn(),
  getEnrichmentPage: vi.fn(),
  getMotifLogo: vi.fn(),
  getChromVarGroups: vi.fn(),
  getChromVarMotif: vi.fn(),
}));
vi.mock("@/api/atacMotifs", async (orig) => ({ ...(await orig<typeof import("@/api/atacMotifs")>()), ...api }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data, onClick }: { data: { type: string; y: string[] }[]; onClick: (e: unknown) => void }) => (
    <button data-testid="plot" onClick={() => onClick({ points: [{ y: data[0]!.y[0] }] })}>{data.map((t) => t.type).join(";")}</button>
  ),
}));

import { AtacMotifsCard } from "./AtacMotifsCard";
import { useAtacRunStore } from "@/stores/atacRunStore";

const META = {
  genome: "GRCh38", n_peaks: 143887, n_motifs: 879, n_scanned: 143836, n_left_out: 51, n_unknown_bases: 5197, p_cutoff: 5e-5, pseudocount: 0.8,
  background: [0.26, 0.24, 0.24, 0.26], n_hits: 17060261, hit_rate: 0.1349, n_best_word_only: 198, seconds: 110,
};
const BASE: MotifState = {
  available: true, genome: "GRCh38", key: "k", running: null,
  scan: { ready: false, running: false, progress: null, n_peaks: 143887, n_bases: 121861766, estimate_seconds: 110, estimate_bytes: 1.1e9, meta: null, p_cutoff: 5e-5 },
  files: {
    genome: { genome: "GRCh38", source: "UCSC hg38", url: "u", size: 835393456, downloaded: false }, genome_auto: true,
    jaspar: { name: "JASPAR", url: "u", size: 283814, downloaded: false }, jaspar_auto: true,
  },
  chromvar: null,
  chromvar_estimate: { n_cells: 11208, estimate_seconds: 540, estimate_bytes: 9.5e8, disk_bytes: 1.5e8, iterations: 50 },
};
const SCANNED: MotifState = { ...BASE, scan: { ...BASE.scan!, ready: true, meta: META } };
const ENRICH: MotifEnrichment = {
  id: "e1", da_id: "d1", column: "cell_type", groups: ["B", "T"],
  summary: { B: { n_query: 500, n_left_out: 2, n_background: 40000, n_significant: 12, capped: true, note: null },
             T: { n_query: 3, n_left_out: 0, n_background: 40000, n_significant: 0, capped: false, note: "T has only 3 query peaks." } },
  rows: {
    B: [{ motif_id: "MA0105", motif: "PAX5", tfs: "PAX5", observed: 100, n_query: 500, background: 3000, n_background: 40000, pct_observed: 20, pct_background: 7.5,
          fold_enrichment: 2.67, pval: 1e-30, pval_adj: 1e-28, gc: 0.5, gc_rich: false, tf_gene: "PAX5", tf_pct: 0.62, tf_log2fc: 3.1, tf_expressed: true },
        { motif_id: "MA9999", motif: "SP1", tfs: "SP1", observed: 90, n_query: 500, background: 3000, n_background: 40000, pct_observed: 18, pct_background: 7.5,
          fold_enrichment: null, pval: 1e-5, pval_adj: 1e-4, gc: 0.8, gc_rich: true, tf_gene: "", tf_pct: null, tf_log2fc: null, tf_expressed: false }],
    T: [],
  },
  params: { p_column: "pval", p_max: 0.005, min_pct: 0.2, da_test: "lr" }, method: "Signac's FindMotifs.", genome: "GRCh38",
  scan: { n_motifs: 879, n_peaks: 143887, n_left_out: 51, n_best_word_only: 198, p_cutoff: 5e-5 }, caveats: "", citation: "", created_at: "",
};
const CHROMVAR = {
  run_id: "r1", n_cells: 11208, n_peaks_used: 143836, n_peaks_no_counts: 0, n_peaks_left_out: 51, seed: 0, iterations: 50, w: 0.1, bs: 50, genome: "GRCh38",
  seconds: 500, stale: false, method: "chromVAR computeDeviations.", caveats: "", citation: "",
  ranking: [{ motif_id: "MA0080", motif: "SPI1", variability: 7.1, p: 0, padj: 0, n_hits: 20000, n_nan: 0, gc_rich: false }],
};
const GROUPS: ChromVarGroups = {
  column: "cell_type", groups: ["B", "T"], n_cells: 11208, n_rows: 50,
  heatmap: { motif_ids: ["MA0080"], motifs: ["SPI1"], variability: [7.1], mean_z: [[1.2, -1.2]] },
  rows: { B: [{ motif_id: "MA0105", motif: "PAX5", tfs: ["PAX5"], mean_z: 3.5, mean_diff: 4.1, pval: 1e-50, pval_adj: 1e-48, gc_rich: false }], T: [] },
};
const MOTIF = { motif_id: "MA0080", motif: "SPI1", n_cells: 3, n_nan: 0, values: [1, -1, null], violin: null, column: null };

function renderCard(groupby = "cell_type", onMotif = vi.fn()) {
  render(<AtacMotifsCard datasetId="d1" groupby={groupby} activeMotif={null} onMotif={onMotif} />);
  return onMotif;
}

describe("AtacMotifsCard", () => {
  beforeEach(() => {
    Object.values(api).forEach((f) => f.mockReset());
    api.getLatestEnrichment.mockResolvedValue(null);
    api.getChromVarGroups.mockResolvedValue(GROUPS);
    useAtacRunStore.setState({ da: null, running: false, datasetId: null, kind: null, error: null, version: 0 });
  });

  it("says what the first scan downloads and costs before it starts", async () => {
    api.getMotifState.mockResolvedValue(BASE);
    renderCard();
    const scan = await screen.findByTestId("atac-motif-scan");
    expect(scan.textContent).toContain("downloads the UCSC hg38 sequence (835 MB) and the JASPAR 2024 motifs");
    expect(scan.textContent).toContain("143,887 peaks");
    expect(screen.queryByTestId("atac-motif-enrichment")).toBeNull();
  });

  it("says when a switch stops the download", async () => {
    api.getMotifState.mockResolvedValue({ ...BASE, files: { ...BASE.files!, genome_auto: false } });
    renderCard();
    expect(await screen.findByText(/GENOME_AUTO_DOWNLOAD is off/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Scan peaks" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("scans with progress and Cancel, then opens the analyses", async () => {
    api.getMotifState.mockResolvedValueOnce(BASE).mockResolvedValue(SCANNED);
    let finish: (v: string) => void = () => undefined;
    api.scanPeaks.mockImplementation((_d: string, h: { onProgress: (p: { message: string }) => void }) => {
      h.onProgress({ message: "Scanned 10 of 20 peaks" });
      return new Promise<string>((resolve) => {
        finish = resolve;
      });
    });
    api.cancelScan.mockResolvedValue({ cancelled: true });
    renderCard();
    fireEvent.click(await screen.findByRole("button", { name: "Scan peaks" }));
    expect((await screen.findByTestId("atac-motif-scan-progress")).textContent).toContain("Scanned 10 of 20 peaks");
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(api.cancelScan).toHaveBeenCalledWith("d1"));
    finish("done");
    expect((await screen.findByTestId("atac-motif-scan-done")).textContent).toContain("51 peaks on contigs");
    expect(screen.getByTestId("atac-motif-enrichment")).toBeTruthy();
    expect(screen.getByTestId("atac-chromvar")).toBeTruthy();
  });

  it("joins a scan already running when it opens", async () => {
    api.getMotifState.mockResolvedValue({ ...BASE, scan: { ...BASE.scan!, running: true } });
    api.scanPeaks.mockReturnValue(new Promise(() => undefined));
    renderCard();
    await waitFor(() => expect(api.scanPeaks).toHaveBeenCalledTimes(1));
    expect(await screen.findByTestId("atac-motif-scan-progress")).toBeTruthy();
  });

  it("needs a differential result, then tests motifs with the vignette's thresholds", async () => {
    api.getMotifState.mockResolvedValue(SCANNED);
    api.runEnrichment.mockResolvedValue(ENRICH);
    renderCard();
    const button = await screen.findByRole("button", { name: "Test motifs" });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/Find differential peaks first/)).toBeTruthy();
    useAtacRunStore.setState({ da: { id: "d9", column: "cell_type", test: "lr" } as never });
    await waitFor(() => expect((screen.getByRole("button", { name: "Test motifs" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Test motifs" }));
    await waitFor(() => expect(api.runEnrichment).toHaveBeenCalledWith("d1", { da_id: "d9", p_column: "pval", p_max: 0.005, min_pct: 0.05 }));
    const result = await screen.findByTestId("atac-motif-enrichment-result");
    expect(result.textContent).toContain("PAX5");
    expect(result.textContent).toContain("62%");
    expect(result.textContent).toContain("GC-rich");
    expect(screen.getByTestId("atac-motif-enrichment-caption").textContent).toContain("kept this group's 5,000 lowest p");
    fireEvent.change(screen.getByRole("combobox", { name: /Group/ }), { target: { value: "T" } });
    expect(await screen.findByText("T has only 3 query peaks.")).toBeTruthy();
  });

  it("opens a motif's logo on its row", async () => {
    api.getMotifState.mockResolvedValue(SCANNED);
    api.getLatestEnrichment.mockResolvedValue(ENRICH);
    api.getMotifLogo.mockResolvedValue({ motif_id: "MA0105", motif: "PAX5", length: 8, consensus: "GCATGACC", svg: "<svg data-testid='svg-logo'></svg>" });
    renderCard();
    fireEvent.click(await screen.findByText("MA0105"));
    const logo = await screen.findByTestId("atac-motif-logo");
    await waitFor(() => expect(logo.querySelector("img")?.getAttribute("src")).toContain("data:image/svg+xml"));
    expect(logo.textContent).toContain("GCATGACC");
    expect(logo.textContent).toContain("log2FC 3.10");
  });

  it("runs chromVAR with an estimate, and a motif goes on the plot from each table and the heatmap", async () => {
    api.getMotifState.mockResolvedValue({ ...SCANNED, chromvar: CHROMVAR });
    api.getChromVarMotif.mockResolvedValue(MOTIF);
    const onMotif = renderCard();
    expect((await screen.findByTestId("atac-chromvar-estimate")).textContent).toContain("11,208 cells x 879 motifs");
    expect(await screen.findByTestId("atac-chromvar-heatmap")).toBeTruthy();
    expect(api.getChromVarGroups).toHaveBeenCalledWith("d1", "cell_type");
    fireEvent.click(screen.getByTestId("plot"));
    await waitFor(() => expect(onMotif).toHaveBeenCalledWith(MOTIF));
    expect(api.getChromVarMotif).toHaveBeenCalledWith("d1", "MA0080", "cell_type");
    fireEvent.click(await screen.findByText("MA0105"));
    await waitFor(() => expect(api.getChromVarMotif).toHaveBeenCalledWith("d1", "MA0105", "cell_type"));
    // Drawn and saved: the line says the column.
    api.getChromVarMotif.mockResolvedValueOnce({ ...MOTIF, saved_column: "chromvar_SPI1_MA0080", save_note: null });
    fireEvent.click(screen.getByTestId("plot"));
    expect((await screen.findByTestId("score-saved")).textContent).toBe(
      "Saved as column chromvar_SPI1_MA0080: Compare and Color offer it.",
    );
  });

  it("says a run on other cells is stale, and asks for a grouping before comparing", async () => {
    api.getMotifState.mockResolvedValue({ ...SCANNED, chromvar: { ...CHROMVAR, stale: true } });
    renderCard("");
    expect(await screen.findByText(/read as stale/)).toBeTruthy();
    expect(screen.getByText(/Choose Group by/)).toBeTruthy();
    expect(api.getChromVarGroups).not.toHaveBeenCalled();
  });

  it("explains itself on a genome it has no sequence for", async () => {
    api.getMotifState.mockResolvedValue({ available: false, reason: "This ATAC input is on a fly (Drosophila) genome (dm6), and scView has gene positions and genome sequence for GRCh38 (human) and mm10 or mm39 (mouse) only." });
    renderCard();
    expect(await screen.findByText(/genome \(dm6\)/)).toBeTruthy();
  });
});
