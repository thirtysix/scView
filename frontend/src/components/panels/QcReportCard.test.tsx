import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { QcReport, QcReportHandlers } from "@/api/qc";

const { getQcReport, postQcReport, streamQcReport } = vi.hoisted(() => ({
  getQcReport: vi.fn(),
  postQcReport: vi.fn(),
  streamQcReport: vi.fn(),
}));
vi.mock("@/api/qc", () => ({ getQcReport, postQcReport, streamQcReport }));

import { QcReportCard } from "./QcReportCard";
import { useSelectionStore } from "@/stores/selectionStore";
import { useViewStore } from "@/stores/viewStore";

const FACTS = [{ kind: "qc", ref: "qc:mito", detail: "Median 2.1% mito." }];
const REPORT = "## Overview\n910 cells [qc:size].\n\n## Concerns\n- Cluster 5 looks damaged [result:qc:leiden:5].";

function empty(extra: Partial<QcReport> = {}): QcReport {
  return { report: "", facts: FACTS, facts_hash: "h1", grounded: false, cached: false, llm_available: true, ...extra };
}

describe("QcReportCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useViewStore.setState({ activePanel: "assessment" });
    useSelectionStore.setState({ highlightedGroup: null });
  });

  it("shows a saved report without writing a new one, and its cluster chips jump", async () => {
    getQcReport.mockResolvedValue(
      empty({ report: REPORT, grounded: true, cached: true, model: "deepseek-ai/DeepSeek-V4.1-Flash", generated_at: "2026-10-16T10:00:00+00:00" }),
    );
    render(<QcReportCard datasetId="d1" />);
    expect(await screen.findByText("Overview")).toBeInTheDocument();
    expect(screen.getByText(/saved report from/)).toBeInTheDocument();
    expect(screen.getByText(/DeepSeek-V4.1-Flash/)).toBeInTheDocument();
    expect(streamQcReport).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "QC 5" }));
    expect(useSelectionStore.getState().highlightedGroup).toEqual({ column: "leiden", value: "5" });
    expect(useViewStore.getState().activePanel).toBe("unified");

    streamQcReport.mockResolvedValue(undefined);
    fireEvent.click(screen.getByRole("button", { name: /regenerate/i }));
    await waitFor(() => expect(streamQcReport).toHaveBeenCalledWith("d1", true, expect.anything()));
  });

  it("streams a new report when asked", async () => {
    getQcReport.mockResolvedValue(empty());
    streamQcReport.mockImplementation(async (_id: string, _regen: boolean, h: QcReportHandlers) => {
      h.onFacts?.({ facts: FACTS, facts_hash: "h1", llm_available: true });
      h.onDelta?.("## Overview\n");
      h.onDelta?.("Healthy data [qc:mito].");
      h.onDone?.({ grounded: true, cached: false, model: "m/flash", generated_at: null });
    });
    render(<QcReportCard datasetId="d1" />);
    fireEvent.click(await screen.findByRole("button", { name: /explain this data/i }));
    expect(await screen.findByText(/Healthy data/)).toBeInTheDocument();
    expect(streamQcReport).toHaveBeenCalledWith("d1", false, expect.anything());
    expect(screen.getByText(/AI-generated · flash/)).toBeInTheDocument();
  });

  it("says when there is no language model and falls back to the one-shot call", async () => {
    getQcReport.mockResolvedValue(empty({ llm_available: false }));
    streamQcReport.mockRejectedValue(new Error("no stream"));
    postQcReport.mockResolvedValue(empty({ llm_available: false, report: "## Overview\n- 910 cells [qc:size]" }));
    render(<QcReportCard datasetId="d1" />);
    expect(await screen.findByText(/No language model is configured/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /explain this data/i }));
    expect(await screen.findByText(/Deterministic facts summary/)).toBeInTheDocument();
    expect(postQcReport).toHaveBeenCalledWith("d1", false);
  });
});
