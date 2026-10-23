import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { DoubletSummary } from "@/api/qc";

const { getDoublets, applyDoubletThreshold, refresh } = vi.hoisted(() => ({
  getDoublets: vi.fn(),
  applyDoubletThreshold: vi.fn(),
  refresh: vi.fn(),
}));
vi.mock("@/api/qc", () => ({ getDoublets, applyDoubletThreshold }));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));
// The plot is a stand-in that can "drag" the threshold line to 0.42: it hands the
// histogram a fake plot div, as react-plotly does, and fires Plotly's event on it.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ onInitialized }: { onInitialized?: (fig: unknown, gd: unknown) => void }) => {
    const handlers: Record<string, (ev: Record<string, number>) => void> = {};
    const gd = {
      on: (name: string, fn: (ev: Record<string, number>) => void) => (handlers[name] = fn),
      removeListener: () => undefined,
    };
    onInitialized?.({}, gd);
    return (
      <button
        type="button"
        onClick={() => handlers.plotly_relayout?.({ "shapes[0].x0": 0.4213, "shapes[0].x1": 0.4213 })}
      >
        drag line
      </button>
    );
  },
}));

import { DoubletCard, thresholdStep } from "./DoubletCard";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

function summary(threshold: number, applied: DoubletSummary["applied"] = null): DoubletSummary {
  return {
    available: true,
    n_cells: 1000,
    n_scored: 1000,
    hist: { bin_edges: [0, 0.5, 1], counts: [950, 50] },
    median: 0.05,
    scrublet_threshold: 0.3,
    applied,
    scrublet_call_kept: applied !== null,
    n_called_now: 40,
    expected_rate_used: 0.008,
    threshold,
    n_called: threshold < 0.3 ? 80 : 40,
    fraction_called: threshold < 0.3 ? 0.08 : 0.04,
    suggestion: {
      rate: 0.008,
      column: null,
      reason: "none",
      n_cells: 1000,
      channels: [],
      basis: "Rule of thumb from 10x Genomics",
    },
    // A run of all cells as one channel (no channel found): no channel table.
    per_channel: false,
    channels: [],
  };
}

/** A run in each of `n` GEM wells of 100 cells, the first with 30 calls. */
function perChannel(n: number, extra: Partial<DoubletSummary> = {}): DoubletSummary {
  const channels = Array.from({ length: n }, (_, i) => ({
    name: String(i + 1), n_cells: 100, n_scored: 100, n_called: i === 0 ? 30 : 1,
    fraction: i === 0 ? 0.3 : 0.01, expected_rate: 0.004, scrublet_threshold: 0.25 + i / 1000, status: "scored",
  }));
  return {
    ...summary(0.3),
    scrublet_threshold: null,
    threshold: null,
    n_called: null,
    fraction_called: null,
    expected_rate_used: null,
    n_cells: n * 100,
    n_called_now: 30 + (n - 1),
    per_channel: true,
    channel_column: "gem_well",
    channel_reason: "channel_column",
    channels_column: "gem_well",
    channels_recorded: true,
    channels,
    ...extra,
  };
}

describe("thresholdStep", () => {
  it("is about a 200th of the histogram's range", () => {
    expect(thresholdStep(1)).toBe(0.005);
    expect(thresholdStep(0.084)).toBe(0.0005); // 133 GEM wells: every score under 0.1
    expect(thresholdStep(undefined)).toBe(0.005);
    expect(thresholdStep(3)).toBe(0.01);
  });
});

describe("DoubletCard", () => {
  beforeEach(() => {
    getDoublets.mockReset().mockImplementation((_id: string, t?: number | null) =>
      Promise.resolve(summary(t ?? 0.3)),
    );
    applyDoubletThreshold.mockReset();
    refresh.mockReset().mockResolvedValue({});
  });

  it("opens on Scrublet's threshold with one channel of every cell and the rule of thumb", async () => {
    render(<DoubletCard datasetId="d1" />);
    expect(await screen.findByTestId("doublet-calls")).toHaveTextContent("40 cells (4.0%)");
    expect(getDoublets).toHaveBeenCalledWith("d1", null);
    expect(screen.getByLabelText("Threshold")).toHaveValue(0.3);
    expect(screen.getByText(/Expected about 0.8% doublets, taking all cells as one channel/)).toBeInTheDocument();
    expect(screen.getByText(/ran with an expected rate of 0.8%/)).toBeInTheDocument();
    expect(screen.queryByRole("table", { name: "Doublet calls per capture channel" })).not.toBeInTheDocument();
    expect(screen.queryByTestId("doublet-per-channel")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply threshold" })).not.toBeDisabled();
  });

  it("says the calls are per channel, with each channel's rate and threshold, and no one threshold", async () => {
    getDoublets.mockResolvedValue(perChannel(3));
    render(<DoubletCard datasetId="d1" />);
    const line = await screen.findByTestId("doublet-per-channel");
    expect(line).toHaveTextContent("Scrublet ran in each of 3 capture channels alone (from gem_well)");
    expect(line).toHaveTextContent("each with its own expected rate (0.4%) and its own threshold: the default calls are per channel");
    expect(line).toHaveTextContent("Scrublet's calls: 32 cells (10.7%)");
    expect(screen.getByLabelText("Threshold")).toHaveValue(null); // none until one is given
    expect(screen.queryByRole("button", { name: /Scrublet's:/ })).not.toBeInTheDocument();
    expect(screen.queryByTestId("doublet-calls")).not.toBeInTheDocument();
    expect(screen.getByText(/Expected 0.4% doublets per channel, from each channel's cells/)).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "Doublet calls per capture channel" });
    expect(table).toHaveTextContent("gem_well");
    expect(screen.getByRole("cell", { name: "30.0%" })).toHaveClass("text-amber-700"); // over 2x expected
    expect(screen.getByRole("cell", { name: "0.25" })).toBeInTheDocument(); // channel 1's own threshold
    expect(screen.getByRole("button", { name: "Apply threshold" })).toBeDisabled();
  });

  it("caps a long channel table honestly, the most doublet-heavy first", async () => {
    getDoublets.mockResolvedValue(perChannel(133));
    render(<DoubletCard datasetId="d1" />);
    await screen.findByTestId("doublet-per-channel");
    const rows = () => screen.getAllByRole("row").length - 1; // minus the header row
    expect(rows()).toBe(8);
    expect(screen.getByTestId("cap-note")).toHaveTextContent("Top 8 of 133 by share called.");
    expect(screen.getAllByRole("row")[1]).toHaveTextContent("30.0%");
    fireEvent.click(screen.getByRole("button", { name: "Show all 133" }));
    expect(rows()).toBe(133);
    expect(screen.queryByTestId("cap-note")).not.toBeInTheDocument();
  });

  it("says which cells were left unscored and marks a channel too small for Scrublet", async () => {
    const s = perChannel(2, { n_unscored: 45, n_unscored_small_channels: 40, min_channel_cells: 100, n_cells: 240 });
    s.channels!.push({ name: "3", n_cells: 40, n_scored: 0, n_called: 0, fraction: 0, expected_rate: 0.004,
      scrublet_threshold: null, status: "too_small" });
    getDoublets.mockResolvedValue(s);
    render(<DoubletCard datasetId="d1" />);
    expect(await screen.findByTestId("doublet-unscored")).toHaveTextContent(
      "45 cells not scored, so never called doublets; 40 of them in channels under 100 cells, too few for " +
        "Scrublet (pick coarser channels, or all cells as one, in the doublet step to score them); 5 with " +
        "under 3 genes or no channel.",
    );
    expect(screen.getByRole("cell", { name: "under 100 cells: not scored" })).toBeInTheDocument();
  });

  it("warns when donors stood in for the channels", async () => {
    getDoublets.mockResolvedValue(perChannel(2, { channel_column: "Donor", channel_reason: "donor", channels_column: "Donor" }));
    render(<DoubletCard datasetId="d1" />);
    expect(await screen.findByTestId("doublet-guess")).toHaveTextContent(/donors pooled in one lane are not channels/);
  });

  it("asks for the counts at a typed or dragged threshold", async () => {
    render(<DoubletCard datasetId="d1" />);
    await screen.findByTestId("doublet-calls");
    fireEvent.change(screen.getByLabelText("Threshold"), { target: { value: "0.2" } });
    await waitFor(() => expect(getDoublets).toHaveBeenLastCalledWith("d1", 0.2));
    expect(await screen.findByText(/80 cells \(8.0%\)/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "drag line" }));
    expect(screen.getByLabelText("Threshold")).toHaveValue(0.42); // rounded to 0.005
    await waitFor(() => expect(getDoublets).toHaveBeenLastCalledWith("d1", 0.42));
  });

  it("applies the threshold, refreshes the calls column, and can put Scrublet's back", async () => {
    const onApplied = vi.fn();
    render(<DoubletCard datasetId="d1" onApplied={onApplied} />);
    await screen.findByTestId("doublet-calls");
    fireEvent.change(screen.getByLabelText("Threshold"), { target: { value: "0.2" } });
    applyDoubletThreshold.mockResolvedValueOnce(summary(0.2, { threshold: 0.2, n_called: 80 }));
    fireEvent.click(screen.getByRole("button", { name: "Apply threshold" }));
    await waitFor(() => expect(applyDoubletThreshold).toHaveBeenCalledWith("d1", 0.2));
    await waitFor(() => expect(onApplied).toHaveBeenCalled());
    expect(refresh).toHaveBeenCalledWith("d1", "predicted_doublet");
    expect(screen.getByText(/kept as predicted_doublet_scrublet/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply threshold" })).toBeDisabled(); // already applied

    applyDoubletThreshold.mockResolvedValueOnce(summary(0.3));
    fireEvent.click(screen.getByRole("button", { name: "Back to Scrublet's call" }));
    await waitFor(() => expect(applyDoubletThreshold).toHaveBeenLastCalledWith("d1", null));
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Back to Scrublet's call" })).not.toBeInTheDocument(),
    );
  });

  it("colours the embedding by the calls", async () => {
    render(<DoubletCard datasetId="d1" />);
    await screen.findByTestId("doublet-calls");
    fireEvent.click(screen.getByRole("button", { name: "Show calls on the embedding" }));
    expect(useSettingsStore.getState().colorBy).toBe("predicted_doublet");
    expect(useViewStore.getState().activePanel).toBe("unified");
  });

  it("says the doublet views use Scrublet when there is no hashing", async () => {
    getDoublets.mockResolvedValue({
      ...summary(0.3),
      source: { source: "scrublet", label: "Scrublet", columns: ["predicted_doublet"], n_cells: 1000, n_doublets: 40,
        fraction: 0.04, hashing: null, scrublet: { n_doublets: 40, fraction: 0.04 }, both: null, options: ["auto", "scrublet"] },
      agreement: null,
    });
    render(<DoubletCard datasetId="d1" />);
    expect(await screen.findByTestId("doublet-source")).toHaveTextContent("Doublet views use: 4.0% doublets by Scrublet.");
    expect(screen.queryByTestId("doublet-agreement")).not.toBeInTheDocument();
    expect(screen.getByText(/Cell\/Gene Filtering drops them when/)).toBeInTheDocument();
  });

  it("puts cell hashing first and shows how Scrublet agrees with it", async () => {
    getDoublets.mockResolvedValue({
      ...summary(0.3),
      source: { source: "hashing", label: "cell hashing", columns: ["hash_class"], n_cells: 1000, n_doublets: 120,
        fraction: 0.12, hashing: { n_doublets: 120, fraction: 0.12, n_negatives: 30 },
        scrublet: { n_doublets: 70, fraction: 0.07 }, both: { n: 40, fraction: 0.04 },
        options: ["auto", "hashing", "scrublet", "either"] },
      agreement: { n_cells: 1000, hashing: 120, scrublet: 70, both: 40, hashing_only: 80, scrublet_only: 30, neither: 850,
        negatives: 30, negatives_scrublet: 5, samples: [], unassigned: null, threshold: 0.3, column: null },
    });
    render(<DoubletCard datasetId="d1" />);
    expect(await screen.findByTestId("doublet-source")).toHaveTextContent(
      "Doublet views use: 12.0% doublets by cell hashing; Scrublet flags 7.0%, 4.0% by both.",
    );
    expect(screen.getByTestId("doublet-agreement-text")).toHaveTextContent(
      "Of 120 hashing doublets, Scrublet at 0.3 also calls 40 (33.3%).",
    );
    expect(screen.getByText(/Cell hashing is the primary call here/)).toBeInTheDocument();
  });

  it("renders nothing before doublet detection has run", async () => {
    getDoublets.mockResolvedValue({ available: false, n_cells: 10, suggestion: summary(0.3).suggestion });
    const { container } = render(<DoubletCard datasetId="d1" />);
    await waitFor(() => expect(getDoublets).toHaveBeenCalled());
    expect(container.textContent).toBe("");
  });
});
