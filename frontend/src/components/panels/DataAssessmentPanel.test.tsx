import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { filterThresholds, OpenInUnifiedView, SkippedSteps, SpeciesNote, SuggestedRateNote } from "./DataAssessmentPanel";
import { useViewStore } from "@/stores/viewStore";

describe("SuggestedRateNote", () => {
  const base = { reason: "donor", n_cells: 6000, basis: "Rule of thumb" };

  it("says the default rate comes from the cells per sample, as a rule of thumb", () => {
    render(
      <SuggestedRateNote
        suggestion={{ ...base, rate: 0.0153, column: "Donor", samples: [{ name: "a", n_cells: 3000, rate: 0.024 }] }}
      />,
    );
    expect(screen.getByText(/Default 1.5%, from the cells per Donor sample \(1\): a rule of thumb/)).toBeTruthy();
  });

  it("says so when all cells are taken as one channel, and is empty before it loads", () => {
    render(<SuggestedRateNote suggestion={{ ...base, rate: 0.048, column: null, reason: null, samples: [] }} />);
    expect(screen.getByText(/all cells as one channel/)).toBeTruthy();
    const { container } = render(<SuggestedRateNote suggestion={null} />);
    expect(container.textContent).toBe("");
  });
});

describe("SkippedSteps", () => {
  it("lists a skipped step by its label, with the reason", () => {
    render(<SkippedSteps notes={{ dissociation_stress: "Only 5 of the 22 genes are in this dataset." }} />);
    expect(screen.getByText(/Dissociation Stress: Only 5 of the 22 genes/)).toBeTruthy();
  });

  it("renders nothing when no step was skipped", () => {
    const { container } = render(<SkippedSteps notes={{}} />);
    expect(container.textContent).toBe("");
  });
});

describe("SpeciesNote", () => {
  it("names the species and the evidence for it", () => {
    render(
      <SpeciesNote
        call={{ species: "mouse", method: "symbol_case", evidence: "Gene symbols are title case." }}
      />,
    );
    expect(screen.getByText("mouse")).toBeTruthy();
    expect(screen.getByText(/Gene symbols are title case/)).toBeTruthy();
  });

  it("renders nothing for an older backend without species", () => {
    const { container } = render(<SpeciesNote call={undefined} />);
    expect(container.textContent).toBe("");
  });
});

describe("OpenInUnifiedView", () => {
  beforeEach(() => useViewStore.setState({ activePanel: "assessment" }));

  it("opens Unified View only when clicked", () => {
    render(<OpenInUnifiedView />);
    expect(useViewStore.getState().activePanel).toBe("assessment");
    fireEvent.click(screen.getByRole("button", { name: /open in unified view/i }));
    expect(useViewStore.getState().activePanel).toBe("unified");
  });
});

describe("filterThresholds", () => {
  it("takes the filtering step's params as set, else their defaults", () => {
    expect(filterThresholds(undefined)).toEqual({
      min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false,
    });
    expect(filterThresholds({ min_genes: 500, drop_doublets: true, min_cells: 5 })).toEqual({
      min_genes: 500, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: true,
    });
  });
});
