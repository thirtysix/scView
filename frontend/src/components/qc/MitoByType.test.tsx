import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { MitoByType as Report, TypeMito } from "@/api/mitoByType";
import type { DatasetInfo } from "@/api/types";

const { getMitoByType } = vi.hoisted(() => ({ getMitoByType: vi.fn() }));
vi.mock("@/api/mitoByType", () => ({ getMitoByType }));

import { MitoByType } from "./MitoByType";
import { useDatasetStore } from "@/stores/datasetStore";

function type(group: string, extra: Partial<TypeMito> = {}): TypeMito {
  return {
    group, n_cells: 300, median: 3.0, mad: 0.7, quantiles: { q05: 1.8, q25: 2.5, q75: 3.6, q95: 4.4 },
    bound: 5.1, removed_global: 0, removed_global_frac: 0, removed_bound: 6, removed_bound_frac: 0.02,
    flagged: false, damaged: false, reasons: [], note: "", ...extra,
  };
}

const LNCAP = type("LNCaP", {
  n_cells: 400, median: 12.1, mad: 1.0, bound: 15.1, removed_global: 260, removed_global_frac: 0.65,
  removed_bound: 16, removed_bound_frac: 0.04, flagged: true,
  reasons: ["The 10.0% ceiling sits below its median (12.1%): it removes 65% of its 400 cells; its own bound, 15.1%, would remove 4%."],
});

function report(extra: Partial<Report> = {}): Report {
  return {
    column: "cell_type", reason: "cell type", measurable: true, n_cells: 720, n_labelled: 700,
    ceiling: 10, ceiling_source: "recorded", filter_mode: "fixed", already_filtered: true,
    mad_k: 3, loss_flag: 0.3, min_cells: 30, types: [LNCAP, type("CAF")], n_flagged: 1,
    removed_global_total: 260, removed_bound_total: 22,
    summary: "A 10.0% ceiling removes 65% of LNCaP (median 12.1%); its own bound (15.1%) would remove 4%.",
    note: "", rule: "Each type's own bound is its median % mito + 3 MADs.",
    citations: "Osorio & Cai 2021, Bioinformatics, doi:10.1093/bioinformatics/btaa751.", ...extra,
  };
}

beforeEach(() => {
  getMitoByType.mockReset();
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: {
      id: "d1",
      obs_columns: [
        { name: "cell_type", dtype: "category", n_unique: 2 },
        { name: "leiden", dtype: "category", n_unique: 8 },
        { name: "n_genes", dtype: "float64", n_unique: 500 },
      ],
    } as unknown as DatasetInfo,
  });
});

describe("MitoByType", () => {
  it("opens on a flagged type: summary, highlighted row, both removal counts, caption with n", async () => {
    getMitoByType.mockResolvedValue(report());
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByText("1 of 2 types cut hard by the 10.0% ceiling")).toBeInTheDocument();
    expect(screen.getByText(/A 10.0% ceiling removes 65% of LNCaP/)).toBeInTheDocument();
    const row = screen.getByTestId("mito-row-LNCaP");
    expect(row.className).toMatch(/bg-amber/);
    expect(row.textContent).toContain("260 (65%)");
    expect(row.textContent).toContain("16 (4%)");
    expect(row.textContent).toContain("15.1%");
    expect(screen.getByTestId("mito-row-CAF").className).not.toMatch(/bg-amber/);
    expect(screen.getByText(/sits below its median/)).toBeInTheDocument();
    expect(screen.getByText(/n = 700 of 720 cells; 2 types of 'cell_type'/)).toBeInTheDocument();
    expect(screen.getByText(/the dataset's last filtering/)).toBeInTheDocument();
    expect(screen.getByText("Own bound").getAttribute("title")).toMatch(/median \+ 3 MADs/);
    expect(screen.getByText("Median mito").getAttribute("title")).toMatch(/mitochondrial/);
  });

  it("says when % mito can't be measured (Kang: no MT- genes)", async () => {
    getMitoByType.mockResolvedValue(report({
      measurable: false, types: [], n_flagged: 0, ceiling: null, ceiling_source: null, summary: "",
      note: "No mitochondrial (MT-) genes in this dataset, so % mito can't be measured and no per-type mito bound can be set.",
    }));
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByText("% mito not measurable")).toBeInTheDocument();
    expect(screen.queryByLabelText("Global % mito ceiling")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Mito by cell type/ }));
    expect(screen.getByText(/can't be measured/)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("re-asks with a typed ceiling and a chosen column", async () => {
    getMitoByType.mockResolvedValue(report());
    render(<MitoByType datasetId="d1" />);
    await screen.findByText("1 of 2 types cut hard by the 10.0% ceiling");
    expect(getMitoByType).toHaveBeenLastCalledWith("d1", undefined, undefined);
    const input = screen.getByLabelText("Global % mito ceiling");
    fireEvent.change(input, { target: { value: "20" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(getMitoByType).toHaveBeenLastCalledWith("d1", undefined, 20));
    fireEvent.change(screen.getByLabelText("Label column for mito by cell type"), { target: { value: "leiden" } });
    await waitFor(() => expect(getMitoByType).toHaveBeenLastCalledWith("d1", "leiden", 20));
    // Only categorical columns are offered.
    expect(screen.queryByRole("option", { name: "n_genes" })).not.toBeInTheDocument();
  });

  it("marks a damaged type in red and stays collapsed when nothing needs a look", async () => {
    getMitoByType.mockResolvedValue(report({
      types: [type("CAF"), type("8", { median: 96.5, bound: null, removed_bound: null, removed_bound_frac: null,
        removed_global: 56, removed_global_frac: 0.98, damaged: true,
        note: "Median 96.5% mito: most of its reads are mitochondrial, likely damaged or dying cells." })],
      n_flagged: 0, summary: "A 20.0% ceiling removes at most 0% of any type. 8 (median 96.5%) looks damaged.",
    }));
    render(<MitoByType datasetId="d1" />);
    const row = await screen.findByTestId("mito-row-8");
    expect(row.className).toMatch(/bg-red/);
    expect(row.textContent).toContain("—");
    expect(screen.getByText(/likely damaged or dying cells/)).toBeInTheDocument();

    getMitoByType.mockResolvedValue(report({ types: [type("CAF")], n_flagged: 0, summary: "no type is flagged" }));
    render(<MitoByType datasetId="d2" />);
    expect(await screen.findByText("The 10.0% ceiling suits every type (1 types)")).toBeInTheDocument();
  });

  it("shows an error without breaking the panel", async () => {
    getMitoByType.mockRejectedValue(new Error("API error 500: boom"));
    render(<MitoByType datasetId="d1" />);
    expect(await screen.findByText(/Mito by cell type unavailable: API error 500: boom/)).toBeInTheDocument();
  });
});
