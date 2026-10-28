import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { DeconvCheck, DeconvDefaults, DeconvResult } from "@/api/deconvolution";
import type { DatasetInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getDeconvolution: vi.fn(),
  getDeconvolutionDefaults: vi.fn(),
  runDeconvolution: vi.fn(),
  getDeconvolutionCheck: vi.fn(),
  runDeconvolutionCheck: vi.fn(),
}));
vi.mock("@/api/deconvolution", async (orig) => ({
  ...(await orig<typeof import("@/api/deconvolution")>()),
  ...api,
}));
// Plotly needs a real canvas; the traces it is handed are what matter here.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data, layout }: { data: { name: string; y: (number | null)[] }[]; layout: { barmode?: string } }) => (
    <div data-testid={layout.barmode === "stack" ? "bars" : "scatter"}>
      {data.map((t) => `${t.name}:${t.y.map((v) => (v == null ? "-" : Math.round(v))).join(",")}`).join("|")}
    </div>
  ),
}));

import { DeconvolutionSubtab } from "./DeconvolutionSubtab";
import { checkScatterTraces, stackedBarTraces, topType } from "@/lib/deconvolution";
import { useDatasetStore } from "@/stores/datasetStore";

function setDataset() {
  const ds: DatasetInfo = {
    id: "d1", name: "d1", filename: "", n_cells: 1000, n_genes: 2000, status: "ready", available_embeddings: [],
    created_at: "", spatial: null,
    obs_columns: [
      { name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "T"] },
      { name: "donor", dtype: "category", n_unique: 4, values: ["D1", "D2", "D3", "D4"] },
      { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    ],
  };
  useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: ds });
}

const DEFAULTS: DeconvDefaults = {
  celltype_column: "cell_type", donor_column: "donor", donor_reason: "donor", n_donors: 4, min_donors_weighted: 2,
  min_donors_check: 3, check_seconds_per_donor: 3.06, max_bulk_mb: 100, max_samples: 1000,
};

const sample = (name: string, b: number, r2 = 0.95, converged = true) => ({
  sample: name, cell_fractions: { B: b, T: 1 - b }, rna_fractions: { B: b * 1.5 / (b * 1.5 + (1 - b)), T: 0 },
  r2, residual: 0.2, genes: 1800, iterations: 6, converged,
});
const RESULT: DeconvResult = {
  result_id: "r1", method: "weighted", method_name: "MuSiC-style weighted NNLS", method_requested: "auto",
  method_reason: "Auto: weighted, since 4 donors give each gene's variation between donors.",
  celltype_column: "cell_type", donor_column: "donor", donor_reason: "donor", types: ["B", "T"], cell_size: [1500, 1000],
  n_donors: [4, 4], n_reference_cells: 1000, n_reference_donors: 4, counts_source: ".raw", filename: "bulk.tsv",
  n_samples: 2, samples: [sample("s1", 0.4), sample("s2", 0.7, 0.81, false)], n_genes_file: 20000,
  n_genes_matched: 1900, gene_share: 0.95, n_genes_used: 1800, marker_source: null, id_kind: "symbol",
  input_kind: "tpm", warnings: ["These look like TPM: the estimates are relative."], notes: [], elapsed_seconds: 0.3,
  created_at: 0,
};
const CHECK: DeconvCheck = {
  result_id: "c1", celltype_column: "cell_type", donor_column: "donor", methods: ["weighted", "nnls"],
  method_names: { weighted: "MuSiC-style weighted NNLS", nnls: "NNLS on marker genes" }, n_mixtures: 50, seed: 0,
  donors: ["D1", "D2", "D3", "D4"], types: ["B", "T"],
  overall: [{ method: "weighted", r: 0.97, rmse: 0.031, bias: 0, n: 400 }, { method: "nnls", r: 0.88, rmse: 0.07, bias: 0, n: 400 }],
  per_type: [
    { method: "weighted", cell_type: "B", r: 0.96, rmse: 0.03, bias: 0.01, n: 200 },
    { method: "weighted", cell_type: "T", r: 0.95, rmse: 0.032, bias: -0.01, n: 200 },
    { method: "nnls", cell_type: "B", r: 0.8, rmse: 0.09, bias: 0.05, n: 200 },
    { method: "nnls", cell_type: "T", r: 0.79, rmse: 0.08, bias: -0.05, n: 200 },
  ],
  points: {
    weighted: { type: [0, 1, 0], truth: [0.2, 0.8, 0.5], estimate: [0.25, 0.75, 0.5], pseudobulk: [true, true, false] },
    nnls: { type: [0, 1], truth: [0.2, 0.8], estimate: [0.3, 0.7], pseudobulk: [true, true] },
  },
  points_capped: false, n_points: { weighted: 3, nnls: 2 }, notes: ["Mixtures are drawn from this dataset's own cells."],
  elapsed_seconds: 12, created_at: 0,
};

beforeEach(() => {
  for (const f of Object.values(api)) f.mockReset();
  api.getDeconvolutionDefaults.mockResolvedValue(DEFAULTS);
  api.getDeconvolution.mockResolvedValue(null);
  api.getDeconvolutionCheck.mockResolvedValue(null);
});

describe("DeconvolutionSubtab", () => {
  it("uploads the file with the chosen columns and method and shows the fractions", async () => {
    setDataset();
    api.runDeconvolution.mockResolvedValue(RESULT);
    render(<DeconvolutionSubtab />);
    await waitFor(() => expect(screen.getByLabelText("Cell types")).toHaveValue("cell_type"));
    const run = screen.getByRole("button", { name: "Deconvolve" });
    expect(run).toBeDisabled(); // no file yet, no earlier run
    fireEvent.change(screen.getByLabelText("Donors"), { target: { value: "donor" } });
    fireEvent.change(screen.getByLabelText("Method"), { target: { value: "nnls" } });
    const file = new File(["gene\ts1\n"], "bulk.tsv", { type: "text/tab-separated-values" });
    fireEvent.change(screen.getByLabelText("Bulk expression file"), { target: { files: [file] } });
    fireEvent.click(run);
    await screen.findByTestId("deconv-result");
    expect(api.runDeconvolution).toHaveBeenCalledWith("d1", file, {
      celltype_column: "cell_type", donor_column: "donor", method: "nnls",
    });
    expect(screen.getByTestId("bars")).toHaveTextContent("B:40,70|T:60,30");
    expect(screen.getByText(/n = 2 bulk samples; 2 cell types/)).toBeInTheDocument();
    expect(screen.getByText(/These look like TPM/)).toBeInTheDocument();
    expect(screen.getByText(/Auto: weighted/)).toBeInTheDocument();
    const rows = screen.getByRole("table", { name: "Estimated fractions" });
    const cells = within(rows).getAllByRole("row")[2]!.textContent;
    expect(cells).toContain("s270.0%30.0%0.81 *");
    expect(screen.getByRole("link", { name: /CSV/ })).toHaveAttribute("href", expect.stringContaining("/deconvolution/r1.csv"));
    // The button now offers a re-run on the kept file.
    expect(screen.getByRole("button", { name: "Run again on bulk.tsv" })).toBeEnabled();
  });

  it("restores the last run and switches to RNA fractions", async () => {
    setDataset();
    api.getDeconvolution.mockResolvedValue(RESULT);
    render(<DeconvolutionSubtab />);
    await screen.findByTestId("deconv-result");
    expect(screen.getByRole("button", { name: "Run again on bulk.tsv" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "RNA fractions" }));
    expect(screen.getByRole("button", { name: "RNA fractions" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("bars")).toHaveTextContent("B:50,78|T:0,0");
  });

  it("shows the server's reason when a file is refused", async () => {
    setDataset();
    api.runDeconvolution.mockRejectedValue(new Error('API error 422: {"detail":"The largest value is 14.2: this looks log-scaled."}'));
    render(<DeconvolutionSubtab />);
    await waitFor(() => expect(screen.getByLabelText("Cell types")).toHaveValue("cell_type"));
    fireEvent.change(screen.getByLabelText("Bulk expression file"), { target: { files: [new File(["x"], "b.tsv")] } });
    fireEvent.click(screen.getByRole("button", { name: "Deconvolve" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The largest value is 14.2: this looks log-scaled.");
  });

  it("runs the self-check with progress and shows per-type scores for each method", async () => {
    setDataset();
    let finish: (c: DeconvCheck) => void = () => undefined;
    api.runDeconvolutionCheck.mockImplementation(
      (_id: string, _body: unknown, h: { onProgress?: (p: { done: number; total: number; donor: string }) => void }) =>
        new Promise<DeconvCheck>((resolve) => {
          h.onProgress?.({ done: 2, total: 4, donor: "D2" });
          finish = resolve;
        }),
    );
    render(<DeconvolutionSubtab />);
    await waitFor(() => expect(screen.getByText(/4 donors: about 15 s/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Run self-check" }));
    expect(await screen.findByTestId("check-progress")).toHaveTextContent("2 of 4 donors held out");
    expect(api.runDeconvolutionCheck).toHaveBeenCalledWith(
      "d1", { celltype_column: "cell_type", donor_column: null, method: "auto" }, expect.anything());
    finish(CHECK);
    const overall = await screen.findByRole("table", { name: "Self-check overall" });
    expect(within(overall).getAllByRole("row")[1]).toHaveTextContent("MuSiC-style weighted NNLS0.970.031");
    const perType = screen.getByRole("table", { name: "Self-check per cell type" });
    expect(within(perType).getAllByRole("row")[1]).toHaveTextContent("B0.960.0300.010");
    fireEvent.change(screen.getByLabelText("Per type for"), { target: { value: "nnls" } });
    expect(within(perType).getAllByRole("row")[1]).toHaveTextContent("B0.800.0900.050");
    expect(screen.getByTestId("scatter")).toHaveTextContent("B:30|T:70");
  });

  it("says the self-check needs 3 donors when there are none", async () => {
    setDataset();
    api.getDeconvolutionDefaults.mockResolvedValue({ ...DEFAULTS, donor_column: null, n_donors: 0 });
    render(<DeconvolutionSubtab />);
    await waitFor(() => expect(screen.getByText(/Needs 3\+ donors \(pick a donor column\)/)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Run self-check" })).toBeDisabled();
  });

  it("links the reference download with the chosen columns and states the caveats", async () => {
    setDataset();
    render(<DeconvolutionSubtab />);
    await waitFor(() => expect(screen.getByLabelText("Cell types")).toHaveValue("cell_type"));
    fireEvent.change(screen.getByLabelText("Donors"), { target: { value: "" } });
    const link = screen.getByRole("link", { name: /Download reference/ });
    expect(link.getAttribute("href")).toMatch(/\/datasets\/d1\/deconvolution\/reference\.zip\?celltype_column=cell_type&donor_column=$/);
    expect(screen.getByTestId("deconv-caveats")).toHaveTextContent("Cell types absent from this dataset cannot be found");
  });
});

describe("deconvolution helpers", () => {
  it("builds one stacked trace per type, in percent", () => {
    const t = stackedBarTraces(RESULT, "cell", () => "red") as { name: string; y: number[]; x: string[] }[];
    expect(t.map((x) => x.name)).toEqual(["B", "T"]);
    expect(t[0]!.x).toEqual(["s1", "s2"]);
    expect(t[0]!.y[0]).toBeCloseTo(40);
  });

  it("scatters truth against estimate per type and adds the identity line", () => {
    const t = checkScatterTraces(CHECK, "weighted", () => "red") as { name: string; x: number[] }[];
    expect(t.map((x) => x.name)).toEqual(["B", "T", "estimate = truth"]);
    expect(t[0]!.x).toEqual([20, 50]);
    expect(checkScatterTraces(CHECK, "other", () => "red")).toEqual([]);
  });

  it("finds each sample's largest fraction", () => {
    expect(topType(RESULT.samples[1]!, "cell")).toEqual({ type: "B", value: 0.7 });
  });
});
