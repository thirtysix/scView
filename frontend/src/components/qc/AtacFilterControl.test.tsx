import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { AtacQcSuggestion, FilterPreview, FilterThresholds } from "@/api/qc";
import { filterThresholds } from "@/components/panels/DataAssessmentPanel";
import { AtacFilterControl, atacPatch } from "./AtacFilterControl";

const ATAC: AtacQcSuggestion = {
  columns: ["atac_nucleosome_signal", "atac_total_counts", "atac_tss_enrichment"],
  suggested: { max_counts: 25000, min_atac_counts: 1800, max_atac_counts: 100000, max_nucleosome_signal: 2,
    min_tss_enrichment: 1 },
  n_removed_by: { max_counts: 4, min_atac_counts: 504, max_atac_counts: 13, max_nucleosome_signal: 31,
    min_tss_enrichment: 1 },
  n_removed: 546,
  source: "Signac's 10x multiome vignette (PBMC 10k): ...",
  missing: [],
  missing_note: null,
  on: [],
};

function preview(atac: AtacQcSuggestion | null, byCriterion: FilterPreview["by_criterion"] = {}): FilterPreview {
  return { n_cells: 11898, by_criterion: byCriterion, atac_qc: atac } as unknown as FilterPreview;
}

const T = filterThresholds(undefined);

describe("AtacFilterControl", () => {
  it("is not shown for RNA-only data", () => {
    const { container } = render(<AtacFilterControl preview={preview(null)} thresholds={T} onChange={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("offers Signac's cuts with what they remove, and fills them in on a click", () => {
    const onChange = vi.fn();
    render(<AtacFilterControl preview={preview(ATAC)} thresholds={T} onChange={onChange} />);
    expect(screen.getByTestId("atac-suggestion")).toHaveTextContent("Together they would remove 546 of 11,898 cells.");
    fireEvent.click(screen.getByRole("button", { name: "Use Signac's cuts" }));
    expect(onChange).toHaveBeenCalledWith({ max_counts: 25000, qc_min_atac_counts: 1800, qc_max_atac_counts: 100000,
      qc_max_nucleosome_signal: 2, qc_min_tss_enrichment: 1 });
    fireEvent.change(screen.getByLabelText("Min TSS enrichment"), { target: { value: "1.5" } });
    expect(onChange).toHaveBeenLastCalledWith({ qc_min_tss_enrichment: 1.5 });
  });

  it("shows only the cuts the data's columns allow, with the cells each removes once on", () => {
    const partial = { ...ATAC, suggested: { max_counts: 25000, min_atac_counts: 1800, max_atac_counts: 100000 },
      missing: ["max_nucleosome_signal", "min_tss_enrichment"] as AtacQcSuggestion["missing"],
      missing_note: "TSS enrichment and nucleosome signal come from the fragment QC." };
    const on: FilterThresholds = { ...T, qc_min_atac_counts: 1800 };
    render(<AtacFilterControl preview={preview(partial, { min_atac_counts: { label: "Min ATAC counts", n_removed: 504 } })}
      thresholds={on} onChange={() => {}} />);
    expect(screen.queryByLabelText("Min TSS enrichment")).toBeNull();
    expect(screen.getByText("removes 504 on its own")).toBeInTheDocument();
    expect(screen.getByText(/come from the fragment QC/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Turn them off" })).toBeInTheDocument();
    expect(atacPatch(partial, false)).toEqual({ max_counts: 0, qc_min_atac_counts: 0, qc_max_atac_counts: 0 });
  });

  it("keeps the cuts in the step's thresholds, off by default", () => {
    expect(T.max_counts).toBe(0);
    expect(T.qc_min_tss_enrichment).toBe(0);
    expect(filterThresholds({ qc_max_nucleosome_signal: 2, max_counts: 25000 }).qc_max_nucleosome_signal).toBe(2);
  });
});
