import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { result } from "@/test/pseudobulkQcFixture";

const { getPseudobulkQc } = vi.hoisted(() => ({ getPseudobulkQc: vi.fn() }));
vi.mock("@/api/sampleQc", () => ({ getPseudobulkQc }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { name?: string }[] }) => (
    <div data-testid="plot">{data.map((t) => t.name ?? "").join(",")}</div>
  ),
}));

import { PseudobulkQc } from "./PseudobulkQc";

beforeEach(() => {
  getPseudobulkQc.mockReset().mockResolvedValue(result());
});

describe("PseudobulkQc", () => {
  it("asks for nothing until opened", () => {
    render(<PseudobulkQc datasetId="ds" />);
    expect(screen.getByText("Sample-level (pseudobulk) QC")).toBeInTheDocument();
    expect(getPseudobulkQc).not.toHaveBeenCalled();
    expect(screen.queryByTestId("plot")).toBeNull();
  });

  it("shows the PCA, drivers, association table and outliers, with n in the caption", async () => {
    render(<PseudobulkQc datasetId="ds" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await screen.findByTestId("pseudobulk-caption");
    expect(getPseudobulkQc).toHaveBeenCalledWith("ds", { splitBy: undefined, celltypeColumn: null, celltype: null });
    const caption = screen.getByTestId("pseudobulk-caption");
    expect(caption).toHaveTextContent("4 profiles (Donor x stim) from 980 of 1,000 cells");
    expect(caption).toHaveTextContent("1 profile under 10 cells left out");
    expect(screen.getByTestId("plot")).toHaveTextContent("CTRL (2),STIM (2)");
    const drivers = screen.getByRole("list", { name: "PC drivers" });
    expect(drivers).toHaveTextContent("PC1 (75%): stim (R² 0.98, p 3e-13)");
    expect(drivers).toHaveTextContent("+ IFIT1, CXCL10, ISG15, MX1");
    const table = screen.getByRole("table", { name: "PC-covariate association" });
    expect(within(table).getByText("(= orig_ident)")).toBeInTheDocument();
    expect(within(table).getByText("0.98*")).toBeInTheDocument();
    expect(within(table).getAllByText("–").length).toBeGreaterThan(0); // untested cell
    expect(screen.getByRole("list", { name: "Outlier profiles" })).toHaveTextContent("D2 | CTRL (90 cells)");
    expect(screen.getByText(/fewer than 10 cells were left out/)).toBeInTheDocument();
  });

  it("re-asks for one cell type, and per sample when the split is unticked", async () => {
    render(<PseudobulkQc datasetId="ds" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    await screen.findByTestId("pseudobulk-caption");
    fireEvent.change(screen.getByLabelText("Cell type"), { target: { value: "B" } });
    await waitFor(() =>
      expect(getPseudobulkQc).toHaveBeenLastCalledWith("ds", { splitBy: undefined, celltypeColumn: "cluster", celltype: "B" }),
    );
    getPseudobulkQc.mockResolvedValue(result({ split_by: null }));
    fireEvent.click(screen.getByLabelText(/One profile per Donor x stim/));
    await waitFor(() =>
      expect(getPseudobulkQc).toHaveBeenLastCalledWith("ds", { splitBy: "", celltypeColumn: "cluster", celltype: "B" }),
    );
    // The box stays, so the split can be ticked back.
    expect(screen.getByLabelText(/One profile per Donor x stim/)).not.toBeChecked();
  });

  it("says why there is no PCA", async () => {
    getPseudobulkQc.mockResolvedValue(
      result({ sample_column: null, profiles: [], covariates: [], association: [], outliers: [], notes: [],
        message: "No sample, donor or batch column was found, so there are no samples to compare." }),
    );
    render(<PseudobulkQc datasetId="ds" />);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    expect(await screen.findByText(/No sample, donor or batch column/)).toBeInTheDocument();
    expect(screen.queryByTestId("plot")).toBeNull();
    expect(screen.queryByLabelText("Cell type")).toBeNull();
  });
});
