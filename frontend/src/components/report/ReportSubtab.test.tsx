import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useReportStore } from "@/stores/reportStore";
import type { DatasetInfo } from "@/api/types";
import { ReportSubtab } from "./ReportSubtab";
import { ClusterReportButton, GeneReportButton } from "./ReportButtons";

const DATASET = {
  id: "ds1",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 13836,
  n_genes: 14053,
  status: "ready",
  available_embeddings: [],
  active_clustering: "leiden",
  obs_columns: [
    { name: "cluster", dtype: "category", n_unique: 3, values: ["B", "CD14 Mono", "NK"] },
    { name: "leiden", dtype: "category", n_unique: 2, values: ["0", "1"] },
    { name: "n_genes", dtype: "int64", n_unique: 900 },
  ],
} as unknown as DatasetInfo;

const meta = {
  dataset: { name: "Kang", n_cells: 13836, n_genes: 14053 },
  design: { condition: "stim", donor: "Donor" },
  scview_version: "0.28.0",
  generated_at: "2026-10-21T12:00:00+00:00",
  elapsed_seconds: 0.6,
};

const clusterReport = (column: string, value: string) => ({
  ...meta,
  kind: "cluster",
  title: `Kang - cluster ${value} report`,
  column,
  value,
  n_cells: 4225,
  sections: [
    {
      id: "identity", title: "Identity", status: "ok", note: null,
      caption: `${value}: 4,225 of 13,836 cells (30.5%) in '${column}'.`,
      method: "Cells counted from the column.",
      data: { n_cells: 4225, fraction: 0.305, n_groups: 3, condition: "stim", donor: "Donor", conditions: ["CTRL", "STIM"],
        by_condition: [{ value: "CTRL", n_cells: 2194, fraction: 0.52 }, { value: "STIM", n_cells: 2031, fraction: 0.48 }],
        by_donor: [{ donor: "D1", n_cells: 500, fraction: 0.12, by_condition: { CTRL: 281, STIM: 219 } }], by_donor_total: 1 },
    },
    {
      id: "markers", title: "Marker genes", status: "ok", note: null,
      caption: "Top 2 of 613 marker genes.", method: "Stored wilcoxon test.",
      data: {
        genes: [
          { gene: "CCL2", rank: 29, logfc: 6.2, padj: 0, pct_in: 0.74, pct_out: 0.11 },
          { gene: "S100A8", rank: 35, logfc: 6.0, padj: 1e-80, pct_in: 0.69, pct_out: 0.04 },
        ],
        n_passing: 613, n_shown: 2,
        database: { status: "not_computed", note: "Not downloaded yet: run the marker database check." },
      },
    },
    {
      id: "enrichment", title: "Pathway enrichment", status: "not_computed",
      caption: "The HALLMARK gene sets are not available.", method: "",
      note: "MSigDB is not on disk.", data: {},
    },
    {
      id: "composition", title: "Composition", status: "not_applicable",
      caption: "Cannot be split here.", method: "", note: "No condition or donor column was found.", data: {},
    },
  ],
});

const geneReport = (gene: string, groupby: string) => ({
  ...meta,
  kind: "gene",
  title: `Kang - gene ${gene} report`,
  gene,
  groupby,
  sections: [
    {
      id: "expression", title: "Expression by group", status: "ok", note: null,
      caption: `${gene} in the 3 groups of '${groupby}'.`, method: "Mean per group.",
      data: { layer_label: "Log-normalized (raw)", groups: [
        { group: "B", n_cells: 100, mean: 0.5, mean_expressing: 1.2, fraction: 0.4 },
        { group: "NK", n_cells: 50, mean: 1.5, mean_expressing: 2.0, fraction: 0.75 },
      ] },
    },
    {
      id: "per_donor", title: "Per donor", status: "not_applicable", caption: `${gene} per donor cannot be shown.`,
      method: "", note: "No donor or sample column was found.", data: {},
    },
  ],
});

function json(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status }));
}

const fetchMock = vi.fn();

function route(url: string) {
  const u = new URL(url, "http://x");
  if (u.pathname === "/api/v1/datasets/ds1/report/cluster")
    return json(clusterReport(u.searchParams.get("column")!, u.searchParams.get("value")!));
  if (u.pathname === "/api/v1/datasets/ds1/report/gene") {
    const gene = u.searchParams.get("gene")!;
    if (gene === "cd14")
      return json({ detail: { message: "Gene 'cd14' not found in this dataset. Did you mean CD14?", suggestion: "CD14" } }, 404);
    return json(geneReport(gene, u.searchParams.get("groupby") ?? "leiden"));
  }
  return json({ detail: "unexpected" }, 500);
}

const lastUrl = () => new URL(String(fetchMock.mock.calls.at(-1)![0]), "http://x");

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockImplementation((url: string) => route(url));
  vi.stubGlobal("fetch", fetchMock);
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "ds1" });
  useSelectionStore.setState({ highlightedGroup: null });
  useSettingsStore.setState({ colorBy: "cluster" });
  useUnifiedViewStore.setState({ scatterOverlay: null, activeSubtab: "markers", groupByColumn: "leiden" });
  useReportStore.setState({ mode: "cluster", datasetId: null, cluster: null, gene: null, groupby: null });
});

afterEach(() => vi.unstubAllGlobals());

describe("ReportSubtab", () => {
  it("reports on the highlighted group by default and renders every section with its status", async () => {
    useSelectionStore.setState({ highlightedGroup: { column: "cluster", value: "CD14 Mono" } });
    render(<ReportSubtab groupByColumn="leiden" />);
    expect(await screen.findByText("Kang - cluster CD14 Mono report")).toBeInTheDocument();
    expect(lastUrl().pathname).toBe("/api/v1/datasets/ds1/report/cluster");
    expect(lastUrl().searchParams.get("value")).toBe("CD14 Mono");
    expect(screen.getByText("CD14 Mono: 4,225 of 13,836 cells (30.5%) in 'cluster'.")).toBeInTheDocument();
    // not computed / not applicable: badge + the note naming where to compute it
    const enr = screen.getByTestId("report-section-enrichment");
    expect(within(enr).getByText("Not computed")).toBeInTheDocument();
    expect(within(enr).getByText("MSigDB is not on disk.")).toBeInTheDocument();
    expect(within(screen.getByTestId("report-section-composition")).getByText("Not applicable")).toBeInTheDocument();
    const db = screen.getByTestId("report-part-Marker databases");
    expect(within(db).getByText(/run the marker database check/)).toBeInTheDocument();
    // tooltips on the columns, p = 0 shown as below double precision
    expect(screen.getByText("log2FC")).toHaveAttribute("title", "Log2 fold change vs the rest of the column");
    expect(screen.getByText("< 1e-300")).toBeInTheDocument();
    expect(screen.getByText("Top 2 of 613 by log fold change.")).toBeInTheDocument();
  });

  it("falls back to the colouring's first group, and the download link carries the target", async () => {
    render(<ReportSubtab groupByColumn="leiden" />);
    await screen.findByText("Kang - cluster B report");
    const link = screen.getByRole("link", { name: /Download HTML/ });
    expect(link).toHaveAttribute("href", "/api/v1/datasets/ds1/report/cluster.html?column=cluster&value=B");
    expect(link).toHaveAttribute("download");
    fireEvent.change(screen.getByLabelText("Report group"), { target: { value: "NK" } });
    await screen.findByText("Kang - cluster NK report");
    expect(screen.getByRole("link", { name: /Download HTML/ })).toHaveAttribute(
      "href", "/api/v1/datasets/ds1/report/cluster.html?column=cluster&value=NK");
  });

  it("switches to the gene mode on the gene on the plot, grouped by the view's group-by", async () => {
    useUnifiedViewStore.setState({ scatterOverlay: { type: "expression", gene: "ISG15" } });
    render(<ReportSubtab groupByColumn="leiden" />);
    await screen.findByText(/cluster B report/);
    fireEvent.click(screen.getByRole("button", { name: "Gene" }));
    await screen.findByText("Kang - gene ISG15 report");
    expect(lastUrl().searchParams.get("gene")).toBe("ISG15");
    expect(lastUrl().searchParams.get("groupby")).toBe("leiden");
    expect(screen.getByRole("link", { name: /Download HTML/ })).toHaveAttribute(
      "href", "/api/v1/datasets/ds1/report/gene.html?gene=ISG15&groupby=leiden");
    expect(within(screen.getByTestId("report-section-per_donor")).getByText("No donor or sample column was found.")).toBeInTheDocument();
  });

  it("asks for a gene when none is on the plot", async () => {
    useReportStore.setState({ mode: "gene" });
    render(<ReportSubtab groupByColumn="leiden" />);
    expect(screen.getByText(/Name a gene/)).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Report gene"), { target: { value: "GNLY" } });
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await screen.findByText("Kang - gene GNLY report");
  });

  it("a marker gene opens its report, and a group in it opens that cluster's report", async () => {
    useSelectionStore.setState({ highlightedGroup: { column: "cluster", value: "CD14 Mono" } });
    render(<ReportSubtab groupByColumn="leiden" />);
    await screen.findByText(/CD14 Mono report/);
    fireEvent.click(screen.getByRole("button", { name: "CCL2" }));
    await screen.findByText("Kang - gene CCL2 report");
    expect(lastUrl().searchParams.get("groupby")).toBe("cluster"); // the cluster report's column
    fireEvent.click(screen.getByRole("button", { name: "NK" }));
    await screen.findByText("Kang - cluster NK report");
    expect(lastUrl().searchParams.get("column")).toBe("cluster");
    expect(useReportStore.getState().mode).toBe("cluster");
  });

  it("an unknown gene offers the close match", async () => {
    useReportStore.setState({ mode: "gene", datasetId: "ds1", gene: "cd14" });
    render(<ReportSubtab groupByColumn="leiden" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Gene 'cd14' not found in this dataset. Did you mean CD14?");
    fireEvent.click(screen.getByRole("button", { name: "Show CD14" }));
    await screen.findByText("Kang - gene CD14 report");
  });

  it("drops another dataset's targets", async () => {
    useReportStore.setState({ mode: "cluster", datasetId: "other", cluster: { column: "cluster", value: "NK" } });
    render(<ReportSubtab groupByColumn="leiden" />);
    await screen.findByText("Kang - cluster B report");
  });
});

describe("report entry points", () => {
  it("the gene button opens the gene report in the Report tab", () => {
    render(<GeneReportButton gene="LYZ" groupby="cluster" />);
    fireEvent.click(screen.getByRole("button", { name: "Gene report for LYZ" }));
    expect(useUnifiedViewStore.getState().activeSubtab).toBe("report");
    expect(useReportStore.getState()).toMatchObject({ mode: "gene", gene: "LYZ", groupby: "cluster", datasetId: "ds1" });
  });

  it("the cluster button opens that cluster's report", () => {
    render(<ClusterReportButton column="cluster" value="NK" />);
    fireEvent.click(screen.getByRole("button", { name: /Cluster report/ }));
    expect(useUnifiedViewStore.getState().activeSubtab).toBe("report");
    expect(useReportStore.getState()).toMatchObject({ mode: "cluster", cluster: { column: "cluster", value: "NK" } });
  });
});
