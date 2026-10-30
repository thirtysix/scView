import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  filterThresholds,
  OpenInUnifiedView,
  SkippedSteps,
  ChannelNote,
  SpeciesNote,
  StepCaution,
  SuggestedRateNote,
  withSuggestedMode,
} from "./DataAssessmentPanel";
import { useViewStore } from "@/stores/viewStore";

describe("StepCaution", () => {
  it("shows a step's caution in full", () => {
    const text =
      "13,162 of 13,836 cells (95%) have both S and G2M scores within ±0.1 of 0: phase calls for these cells are not reliable.";
    render(<StepCaution caution={text} />);
    expect(screen.getByText(text)).toBeTruthy();
  });

  it("renders nothing without one (or from an older backend)", () => {
    const { container } = render(
      <>
        <StepCaution caution={null} />
        <StepCaution />
      </>,
    );
    expect(container.textContent).toBe("");
  });
});

describe("SuggestedRateNote", () => {
  const base = { n_cells: 6000, basis: "Rule of thumb" };

  it("says each channel gets its own rate from its cells, as a rule of thumb", () => {
    render(
      <SuggestedRateNote
        suggestion={{
          ...base, rate: 0.0153, per_channel: true, column: "gem_well", reason: "channel_column", n_channels: 2,
          channels: [
            { name: "1", n_cells: 3000, rate: 0.024, scored: true },
            { name: "2", n_cells: 600, rate: 0.0048, scored: true },
          ],
        }}
      />,
    );
    expect(
      screen.getByText(/Default: each channel's own, 0.5-2.4%, in 2 capture channels \(1, 2\), from gem_well: a rule of thumb/),
    ).toBeTruthy();
    expect(screen.getByText(/A rate typed here applies to every channel/)).toBeTruthy();
  });

  it("says so when all cells are taken as one channel, and is empty before it loads", () => {
    render(<SuggestedRateNote suggestion={{ ...base, rate: 0.048, column: null, reason: "none", channels: [] }} />);
    expect(screen.getByText(/Default 4.8%, from all cells as one channel \(no capture channel found\)/)).toBeTruthy();
    const { container } = render(<SuggestedRateNote suggestion={null} />);
    expect(container.textContent).toBe("");
  });
});

describe("ChannelNote", () => {
  it("names the channels read from the barcodes", () => {
    render(
      <ChannelNote
        suggestion={{
          rate: 0.055, per_channel: true, column: "capture_channel", reason: "barcode_prefix", derived: true,
          n_cells: 13836, n_channels: 2, n_unscored: 0, basis: "",
          channels: [
            { name: "CTRL", n_cells: 6573, rate: 0.0526, scored: true },
            { name: "STIM", n_cells: 7263, rate: 0.0581, scored: true },
          ],
        }}
      />,
    );
    expect(screen.getByTestId("doublet-channel-note")).toHaveTextContent(
      "Scrublet will run in 2 capture channels (CTRL, STIM), from the prefix of the cell barcodes.",
    );
  });

  it("warns when donors stand in for the channels, and counts the cells left unscored", () => {
    render(
      <ChannelNote
        suggestion={{
          rate: 0.01, per_channel: true, column: "Donor", reason: "donor", note: "Donors pooled in a lane: a guess.",
          n_cells: 1200, n_channels: 3, n_unscored: 90, min_channel_cells: 100, basis: "",
          skipped: ["'well': 1,200 groups with a median of 1 cells, under the 100 Scrublet needs"],
          channels: [
            { name: "d1", n_cells: 600, rate: 0.0048, scored: true },
            { name: "d2", n_cells: 510, rate: 0.004, scored: true },
            { name: "d3", n_cells: 90, rate: 0.004, scored: false },
          ],
        }}
      />,
    );
    const note = screen.getByTestId("doublet-channel-note");
    expect(note).toHaveTextContent("3 capture channels (d1, d2, d3), from the donor / sample column");
    expect(note).toHaveTextContent("Donors pooled in a lane: a guess.");
    expect(note).toHaveTextContent("90 cells are in channels under 100 cells (or have no channel): not scored");
    expect(note).toHaveTextContent("Passed over 'well': 1,200 groups");
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
    const mad = {
      qc_threshold_mode: "mad", qc_mad_k: 5, qc_mad_k_mt: 3, qc_mad_upper: false, qc_sample_column: "",
      qc_mito_mode: "global", qc_mito_type_column: "", qc_mito_type_floor: false, qc_doublet_source: "auto", drop_hash_negatives: false,
    };
    expect(filterThresholds(undefined)).toEqual({
      min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false, ...mad,
    });
    expect(filterThresholds({ min_genes: 500, drop_doublets: true, min_cells: 5 })).toEqual({
      min_genes: 500, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: true, ...mad,
    });
    expect(
      filterThresholds({ qc_threshold_mode: "fixed", qc_mad_k: 4, qc_sample_column: "lane" }),
    ).toMatchObject({ qc_threshold_mode: "fixed", qc_mad_k: 4, qc_sample_column: "lane" });
    expect(filterThresholds({ qc_mito_mode: "per_type", qc_mito_type_column: "cell_type" })).toMatchObject({
      qc_mito_mode: "per_type", qc_mito_type_column: "cell_type", qc_mito_type_floor: false,
    });
    expect(filterThresholds({ qc_mito_mode: "per_type", qc_mito_type_floor: true })).toMatchObject({
      qc_mito_mode: "per_type", qc_mito_type_floor: true,
    });
    expect(filterThresholds({ qc_doublet_source: "either", drop_hash_negatives: true })).toMatchObject({
      qc_doublet_source: "either", drop_hash_negatives: true,
    });
    expect(filterThresholds({ qc_doublet_source: "nonsense" }).qc_doublet_source).toBe("auto");
    // Visium HD bins: no min genes, min counts 1, % mito off until asked for; a choice stands.
    expect(filterThresholds(undefined, null, "bin")).toMatchObject({ min_genes: 0, min_counts: 1, max_pct_mt: 100 });
    expect(filterThresholds({ min_genes: 50 }, null, "bin").min_genes).toBe(50);
    expect(filterThresholds({ qc_grid_mito: true, max_pct_mt: 30 }, null, "bin")).toMatchObject({ max_pct_mt: 30, qc_grid_mito: true });
    expect(filterThresholds({ qc_grid_mito: true })).not.toHaveProperty("qc_grid_mito");
  });
});

describe("withSuggestedMode", () => {
  it("turns suggested fixed cut-offs into the fixed mode, unless a mode is named", () => {
    expect(withSuggestedMode("filtering", { min_genes: 200, max_pct_mt: 20 })).toEqual({
      min_genes: 200, max_pct_mt: 20, qc_threshold_mode: "fixed",
    });
    const mad = { qc_threshold_mode: "mad", min_genes: 200 };
    expect(withSuggestedMode("filtering", mad)).toBe(mad);
    const ceiling = { max_pct_mt: 15, min_cells: 3 };
    expect(withSuggestedMode("filtering", ceiling)).toBe(ceiling); // no cut-off MAD would ignore
    expect(withSuggestedMode("pca", { min_genes: 1 })).toEqual({ min_genes: 1 });
  });
});
