import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import { renderHook } from "@testing-library/react";
import type { FilterPreview } from "@/api/qc";
import type { LibraryConcordance } from "@/api/sampleQc";
import { SampleQcTable } from "./SampleQcTable";
import { hoverTip } from "@/test/hoverTip";

const getLibraryConcordance = vi.fn();
vi.mock("@/api/sampleQc", () => ({ getLibraryConcordance: (...a: unknown[]) => getLibraryConcordance(...a) }));
const { useLibraryConcordance } = await import("./LibraryConcordance");

const MESSAGE =
  "NCD07 · inguinal fat pad: libraries Mm_ING_10-1 (868 cells) and Mm_ING_10-2 (1,399) disagree in cell types: distance 0.49";

function preview(): FilterPreview {
  const g = (name: string) => ({
    name, n_cells: 1000, n_removed: 0, fraction: 0, by_criterion: {},
    median_genes: 800, median_counts: 2000, median_pct_mt: 0.1, flags: [], reasons: [],
  });
  return {
    n_cells: 3000, n_removed: 0, n_kept: 3000, fraction_removed: 0,
    thresholds: { min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false, mode: "mad" },
    has_mt: true, doublet_calls: false, by_criterion: {},
    samples: { column: "donor_id", reason: "donor", compared: true, typical: {}, groups: [g("NCD06"), g("NCD07"), g("HFD01")] },
    clusters: null, warnings: [],
  };
}

function libs(over: Partial<LibraryConcordance> = {}): LibraryConcordance {
  return {
    status: "checked", reason: null, sample_column: "donor_id", library_column: "biosample_id",
    library_source: "sample_like", celltype_column: "author_cell_type_subcluster", split_by: ["tissue"],
    n_checked: 2, n_flagged: 1, flagged: ["NCD07 · inguinal fat pad"],
    reference: { basis: "matched", matched_on: ["diet_type", "sex", "tissue"], n_pairs: 29, cut: 0.35, depth_cut_log2: 1.5 },
    note: "Each pair is judged against 29 distances between different samples alike in diet_type, sex, tissue.",
    samples: [
      { name: "HFD01 · epididymal fat pad", sample: "HFD01", split: { tissue: "epididymal fat pad" }, status: "checked",
        libraries: [], flagged: false, flags: [], pair: null, message: null },
      { name: "NCD07 · inguinal fat pad", sample: "NCD07", split: { tissue: "inguinal fat pad" }, status: "checked",
        libraries: [], flagged: true, flags: ["composition"], pair: null, message: MESSAGE },
    ],
    ...over,
  };
}

describe("library concordance in the QC per sample table", () => {
  it("puts a caution on the flagged sample, counts it, and gives the numbers under the table", async () => {
    render(<SampleQcTable preview={preview()} mad={null} libraries={libs()} />);
    expect(screen.getByText("1 of 3 samples stand out")).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "QC per sample" });
    const row = within(table).getByText("NCD07").closest("tr")!;
    const chip = within(row).getByTestId("libraries-differ");
    expect(chip).toHaveTextContent("Libraries differ");
    expect(within(table).getAllByTestId("libraries-differ")).toHaveLength(1);
    const tip = await hoverTip(screen.getByText("Libraries differ"));
    expect(tip).toContain("another sample's cells");
    const caption = screen.getByTestId("library-caption");
    expect(caption).toHaveTextContent("2 samples have two or more 'biosample_id' libraries (compared within one tissue)");
    expect(caption).toHaveTextContent("1 with libraries that disagree");
    expect(caption).toHaveTextContent(MESSAGE);
  });

  it("says nothing for other samples, or when nothing was checked", () => {
    const { rerender } = render(<SampleQcTable preview={preview()} mad={null} libraries={libs({ sample_column: "biosample_id" })} />);
    expect(screen.queryByTestId("libraries-differ")).toBeNull();
    expect(screen.queryByTestId("library-caption")).toBeNull();
    expect(screen.getByText("No sample stands out (3 samples)")).toBeInTheDocument();
    rerender(<SampleQcTable preview={preview()} mad={null} libraries={libs({ status: "none", samples: [], n_checked: 0, n_flagged: 0, flagged: [] })} />);
    expect(screen.queryByTestId("library-caption")).toBeNull();
  });
});

describe("useLibraryConcordance", () => {
  beforeEach(() => {
    getLibraryConcordance.mockReset();
  });

  it("asks once per sample column and drops a failure", async () => {
    getLibraryConcordance.mockResolvedValue(libs());
    const { result, rerender } = renderHook(({ col }) => useLibraryConcordance("ds", col), {
      initialProps: { col: "donor_id" as string | null },
    });
    await waitFor(() => expect(result.current?.n_flagged).toBe(1));
    expect(getLibraryConcordance).toHaveBeenCalledWith("ds", "donor_id");
    getLibraryConcordance.mockRejectedValue(new Error("boom"));
    rerender({ col: null });
    await waitFor(() => expect(result.current).toBeNull());
    expect(getLibraryConcordance).toHaveBeenCalledTimes(2);
  });
});
