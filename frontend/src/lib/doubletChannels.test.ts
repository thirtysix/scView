import { describe, it, expect } from "vitest";
import type { DoubletRateSuggestion } from "@/api/qc";
import { channelsText, rateRange, statusText, suggestionText, unscoredText } from "./doubletChannels";

const wells: DoubletRateSuggestion = {
  rate: 0.006, per_channel: true, column: "capture_channel", reason: "barcode_suffix", derived: true,
  n_cells: 100000, n_channels: 133, n_unscored: 0, basis: "",
  channels: Array.from({ length: 133 }, (_, i) => ({ name: String(i + 1), n_cells: 640 + i * 2, rate: 0.00512 + i * 1.6e-5, scored: true })),
};

describe("doublet channel texts", () => {
  it("names many channels by number and where they came from", () => {
    expect(channelsText(wells)).toBe("133 capture channels, from the -N suffix of the cell barcodes");
    expect(channelsText({ ...wells, column: "gem_well", reason: "channel_column" })).toBe("133 capture channels, from gem_well");
    expect(channelsText({ ...wells, column: null, reason: "requested_all", channels: [] })).toBe(
      "all cells as one channel, as chosen",
    );
  });

  it("gives the range of the channels' rates", () => {
    expect(rateRange([0.00512, 0.0073])).toBe("0.5-0.7%");
    expect(rateRange([0.004, 0.004])).toBe("0.4%");
    expect(rateRange([])).toBe("—");
    expect(suggestionText(wells)).toMatch(/^Default: each channel's own, 0.5-0.7%, in 133 capture channels/);
  });

  it("explains unscored cells only when there are any", () => {
    expect(unscoredText(0, 0, 100)).toBeNull();
    expect(unscoredText(3, 0, 100)).toBe("3 cells not scored, so never called doublets; 3 with under 3 genes or no channel.");
  });

  it("marks a channel's status", () => {
    const row = { name: "1", n_cells: 50, n_scored: 0, n_called: 0, fraction: 0, expected_rate: 0.004, scrublet_threshold: null };
    expect(statusText({ ...row, status: "too_small" }, 100)).toBe("under 100 cells: not scored");
    expect(statusText({ ...row, status: "failed" }, 100)).toBe("Scrublet failed: not scored");
    expect(statusText({ ...row, n_scored: 50, status: "scored" }, 100)).toBe("no threshold found: no calls");
    expect(statusText({ ...row, n_scored: 50, status: "scored", scrublet_threshold: 0.2 }, 100)).toBeNull();
  });
});
