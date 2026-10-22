import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { SeqFlag, SeqMetricsResponse, SeqSample } from "@/api/seqqc";
import { useDatasetStore } from "@/stores/datasetStore";

const { getSeqMetrics, attachSeqMetrics } = vi.hoisted(() => ({
  getSeqMetrics: vi.fn(),
  attachSeqMetrics: vi.fn(),
}));
vi.mock("@/api/seqqc", () => ({ getSeqMetrics, attachSeqMetrics }));
// Plotly needs a real canvas; the traces and shapes it is handed are what matter here.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data, layout }: { data: { name: string; x: number[] }[]; layout: { shapes: unknown[] } }) => (
    <div data-testid="plot">
      {data.map((t) => `${t.name}:${t.x.length}`).join("|")} shapes={layout.shapes.length}
    </div>
  ),
}));

import { SequencingQc } from "./SequencingQc";

const FLAG: SeqFlag = {
  metric: "valid_barcodes",
  label: "Valid barcodes",
  level: "warn",
  title: "Low fraction valid barcodes",
  value: 0.7031,
  threshold: 0.75,
  warn: 0.75,
  error: 0.5,
  direction: "below",
  ideal: "Ideal > 75%",
  message: "Valid barcodes is 70.3%, below Cell Ranger's warn line of 75.0% (error below 50.0%). This may indicate a quality issue with the R1 read.",
  source: "Cell Ranger 10.1 web summary alert",
  url: "https://github.com/10XGenomics/cellranger",
};

// pbmc_1k_v3 as the backend answers it (values from its metrics_summary.csv and raw matrix).
const PBMC: SeqSample = {
  sample: "pbmc_1k_v3",
  column: null,
  cells_source: "ingest",
  attached_at: "2026-10-22T04:00:00+00:00",
  files: { metrics: "pbmc_1k_v3_metrics_summary.csv", raw: ["pbmc_1k_v3_raw_feature_bc_matrix.h5"] },
  metrics: {
    format: "count",
    values: {
      estimated_cells: 1222,
      mean_reads_per_cell: 54502,
      median_genes_per_cell: 1919,
      median_umi_per_cell: 6628,
      sequencing_saturation: 0.708,
      valid_barcodes: 0.974,
      conf_transcriptome: 0.537,
      q30_rna_read: 0.902,
      frac_reads_in_cells: 0.949,
    },
    scope: {},
    raw: [["Estimated Number of Cells", "1,222"], ["Mean Reads per Cell", "54,502"]],
    notes: [],
  },
  knee: {
    n_barcodes: 234600,
    total_umis: 10196940,
    n_cells: 1222,
    n_cells_matched: 1222,
    match: "barcode",
    knee: 5199.8,
    inflection: 376.0,
    n_above_knee: 873,
    n_above_inflection: 1264,
    cells_umi_min: 503,
    cells_umi_median: 6628,
    frac_umis_in_cells: 0.9057,
    n_cells_below_knee: 349,
    n_above_knee_not_cells: 0,
    feature_types: "gene expression only",
    method: "DropletUtils barcodeRanks (curve tracing)",
    url: "https://github.com/MarioniLab/DropletUtils",
    curve: { rank: [1, 2, 900, 1300, 234600], umi: [90000, 80000, 4000, 200, 1], kept: [1, 1, 1, 0, 0] },
  },
  notes: [],
  flags: [],
  summary: "No Cell Ranger alert fires: the 10 metrics with an alert are all on the good side of 10x's warning lines.",
};

const FLEX: SeqSample = {
  ...PBMC,
  sample: "OR00001_1",
  column: "sample",
  cells_source: "dataset",
  attached_at: "2026-10-22T05:00:00+00:00",
  metrics: { ...PBMC.metrics!, values: { estimated_cells: 2616, valid_barcodes: 0.7031 } },
  knee: null,
  flags: [FLAG],
  summary: "1 of 4 checked metrics cross a Cell Ranger alert: valid barcodes.",
};

const INFO = [
  { key: "estimated_cells" as const, label: "Estimated cells", kind: "int" as const, tip: "Barcodes Cell Ranger called as cells." },
  { key: "valid_barcodes" as const, label: "Valid barcodes", kind: "frac" as const, tip: "Share of reads whose cell barcode matches the whitelist." },
];

const resp = (samples: SeqSample[]): SeqMetricsResponse => ({ samples, n_samples: samples.length, metrics_info: INFO });

describe("SequencingQc", () => {
  beforeEach(() => {
    getSeqMetrics.mockReset();
    attachSeqMetrics.mockReset();
    useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  });

  it("tabulates the metrics, says why there are no flags and plots the barcode ranks", async () => {
    getSeqMetrics.mockResolvedValue(resp([PBMC]));
    render(<SequencingQc datasetId="d1" />);
    const table = await screen.findByTestId("seqqc-table");
    expect(getSeqMetrics).toHaveBeenCalledWith("d1");
    const rows = within(table).getAllByRole("row");
    expect(rows[1]!.textContent).toBe("pbmc_1k_v31,22254,5021,9196,62870.8%97.4%53.7%90.2%94.9%none");
    for (const th of within(table).getAllByRole("columnheader")) expect(th.getAttribute("title")).toBeTruthy();
    expect(screen.getByTestId("seqqc-summary").textContent).toContain("No Cell Ranger alert fires");
    expect(screen.queryByTestId("seqqc-flags")).toBeNull();
    // 3 cells + 2 other barcodes drawn, knee and inflection lines.
    expect(screen.getByTestId("plot").textContent).toBe("Other barcodes:2|Cells (filtered matrix):3 shapes=2");
    expect(screen.getByTestId("seqqc-curve-caption").textContent).toContain("n = 234,600 barcodes with UMIs.");
    expect(screen.getByTestId("seqqc-knee").textContent).toBe(
      "The filtered matrix kept 1,222 cells; 1,222 are in the raw matrix, 873 barcodes lie above the knee and 1,264 above the inflection. " +
        "The cells' UMI floor is 503 (median 6,628), and they hold 90.6% of the matrix's UMIs. " +
        "349 cells are below the knee: Cell Ranger's caller (EmptyDrops-style since 3.0) keeps low-RNA cells the knee alone would drop.",
    );
  });

  it("colours a flagged value, lists the flag with its threshold and source, and switches samples", async () => {
    getSeqMetrics.mockResolvedValue(resp([PBMC, FLEX]));
    render(<SequencingQc datasetId="d1" />);
    const table = await screen.findByTestId("seqqc-table");
    const flexRow = within(table).getAllByRole("row")[2]!;
    const cell = within(flexRow).getByText("70.3%");
    expect(cell.className).toContain("text-amber-700");
    expect(cell.getAttribute("title")).toBe(
      "Low fraction valid barcodes: 70.3% (warning below 75.0%, error below 50.0%; Ideal > 75%). Source: Cell Ranger 10.1 web summary alert.",
    );
    expect(flexRow.textContent).toContain("1 alert");
    fireEvent.click(flexRow);
    const badge = await screen.findByTestId("seqqc-flag-badge");
    expect(badge.textContent).toBe("Warning: Low fraction valid barcodes");
    expect(screen.getByText(/below Cell Ranger's warn line of 75.0%/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Cell Ranger 10.1 web summary alert" }).getAttribute("href")).toContain("cellranger");
    expect(screen.getByText(/No raw \(unfiltered\) matrix is attached for OR00001_1/)).toBeTruthy();
    expect(screen.queryByTestId("plot")).toBeNull();
  });

  it("offers the attach form when none is attached, and shows what comes back", async () => {
    getSeqMetrics.mockResolvedValue(null);
    attachSeqMetrics.mockResolvedValue(resp([PBMC]));
    render(<SequencingQc datasetId="d1" />);
    expect((await screen.findByTestId("seqqc-none")).textContent).toContain("No Cell Ranger metrics are attached");
    fireEvent.click(screen.getByRole("button", { name: /Attach Cell Ranger metrics/ }));
    const attach = screen.getByRole("button", { name: "Attach" }) as HTMLButtonElement;
    expect(attach.disabled).toBe(true);
    const file = new File(["Estimated Number of Cells\n"], "metrics_summary.csv", { type: "text/csv" });
    fireEvent.change(screen.getByTestId("seqqc-file-input"), { target: { files: [file] } });
    expect(screen.getByTestId("seqqc-file-names").textContent).toBe("metrics_summary.csv");
    fireEvent.click(attach);
    await waitFor(() => expect(attachSeqMetrics).toHaveBeenCalledWith("d1", [file], {}));
    expect(await screen.findByTestId("seqqc-table")).toBeTruthy();
    expect(screen.queryByTestId("seqqc-attach")).toBeNull();
  });

  it("attaches to one sample of a merged dataset and shows the server's sentence on failure", async () => {
    useDatasetStore.setState({
      currentDatasetId: "d1",
      currentDataset: {
        id: "d1", name: "merged", filename: "m.h5ad", n_cells: 10, n_genes: 3, status: "ready",
        available_embeddings: [], created_at: "",
        obs_columns: [
          { name: "sample", dtype: "category", n_unique: 2, values: ["s1", "s2"] },
          { name: "n_genes", dtype: "int64", n_unique: 900 },
        ],
      },
    });
    getSeqMetrics.mockResolvedValue(null);
    attachSeqMetrics.mockRejectedValueOnce(new Error("No cells have 'sample' = 's2'."));
    render(<SequencingQc datasetId="d1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Attach Cell Ranger metrics/ }));
    const file = new File(["x"], "s2_metrics_summary.csv");
    fireEvent.change(screen.getByTestId("seqqc-file-input"), { target: { files: [file] } });
    const select = screen.getByLabelText("For") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual(["the whole dataset", "one sample (2)"]);
    fireEvent.change(select, { target: { value: "sample" } });
    const attach = screen.getByRole("button", { name: "Attach" }) as HTMLButtonElement;
    expect(attach.disabled).toBe(true); // a sample must be chosen
    fireEvent.change(screen.getByLabelText("Sample"), { target: { value: "s2" } });
    fireEvent.click(attach);
    expect((await screen.findByRole("alert")).textContent).toBe("No cells have 'sample' = 's2'.");
    expect(attachSeqMetrics).toHaveBeenCalledWith("d1", [file], { column: "sample", sample: "s2" });
  });

  it("shows a load error instead of the attach prompt", async () => {
    getSeqMetrics.mockRejectedValue(new Error("Request failed (500)."));
    render(<SequencingQc datasetId="d1" />);
    expect((await screen.findByRole("alert")).textContent).toBe("Request failed (500).");
    expect(screen.queryByTestId("seqqc-none")).toBeNull();
  });
});
