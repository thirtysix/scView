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

import { DoubletCard } from "./DoubletCard";
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
      rate: 0.004,
      column: "Donor",
      reason: "donor",
      n_cells: 1000,
      samples: [
        { name: "d1", n_cells: 500, rate: 0.004 },
        { name: "d2", n_cells: 500, rate: 0.004 },
      ],
      basis: "Rule of thumb from 10x Genomics",
    },
    samples: [
      { name: "d1", n_cells: 500, n_called: 30, fraction: 0.06, expected_rate: 0.004 },
      { name: "d2", n_cells: 500, n_called: 10, fraction: 0.02, expected_rate: 0.004 },
    ],
  };
}

describe("DoubletCard", () => {
  beforeEach(() => {
    getDoublets.mockReset().mockImplementation((_id: string, t?: number | null) =>
      Promise.resolve(summary(t ?? 0.3)),
    );
    applyDoubletThreshold.mockReset();
    refresh.mockReset().mockResolvedValue({});
  });

  it("opens on Scrublet's threshold with the calls per sample and the rule of thumb", async () => {
    render(<DoubletCard datasetId="d1" />);
    expect(await screen.findByTestId("doublet-calls")).toHaveTextContent("40 cells (4.0%)");
    expect(getDoublets).toHaveBeenCalledWith("d1", null);
    expect(screen.getByLabelText("Threshold")).toHaveValue(0.3);
    expect(screen.getByText(/rule of thumb from 10x Genomics/)).toBeInTheDocument();
    expect(screen.getByText(/ran with an expected rate of 0.8%/)).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "d1" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "6.0%" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply threshold" })).not.toBeDisabled();
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

  it("renders nothing before doublet detection has run", async () => {
    getDoublets.mockResolvedValue({ available: false, n_cells: 10, suggestion: summary(0.3).suggestion });
    const { container } = render(<DoubletCard datasetId="d1" />);
    await waitFor(() => expect(getDoublets).toHaveBeenCalled());
    expect(container.textContent).toBe("");
  });
});
