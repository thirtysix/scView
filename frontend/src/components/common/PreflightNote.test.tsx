import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({ getRunPreflight: vi.fn(), getIngestPreflight: vi.fn() }));
vi.mock("@/api/preflight", () => api);

import type { PreflightVerdict } from "@/api/preflight";
import { PreflightIngestNotes, PreflightNote, PreflightRunNote } from "./PreflightNote";

const GB = 1024 ** 3;

function verdict(level: PreflightVerdict["level"], extra: Partial<PreflightVerdict> = {}): PreflightVerdict {
  return {
    level,
    sentence: `About 3.1 GB more than now at its peak (doublet detection), roughly 1 to 4 min. scView has 5.2 GB free of its 8.0 GB limit.`,
    need_bytes: 3.1 * GB,
    peak_bytes: 4.0 * GB,
    peak_stage: "doublet detection",
    available_bytes: 5.2 * GB,
    limit_bytes: 8 * GB,
    used_bytes: 2.8 * GB,
    limited: true,
    source: "container limit",
    seconds: [60, 240],
    cached: { count: 0, bytes: 0, frees_enough: false },
    suggestions: [],
    error_band: 0.3,
    breakdown: [
      { label: "This dataset now", bytes: 0.9 * GB },
      { label: "At the peak (doublet detection)", bytes: 4.0 * GB },
    ],
    notes: [],
    ...extra,
  };
}

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.useRealTimers());

describe("PreflightNote", () => {
  it("fits: the sentence, quiet, no alert and no suggestions", () => {
    render(<PreflightNote verdict={verdict("fits", { suggestions: ["never shown when it fits"] })} />);
    expect(screen.getByText(/About 3.1 GB more than now/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText("never shown when it fits")).toBeNull();
    expect(document.querySelector("[data-level=fits]")?.className).toContain("text-slate-500");
  });

  it("tight: amber text, suggestions left to the tooltip", () => {
    render(<PreflightNote verdict={verdict("tight", { suggestions: ["Raise SCVIEW_MEMORY_GB"] })} />);
    expect(document.querySelector("[data-level=tight]")?.className).toContain("text-amber-700");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText("Raise SCVIEW_MEMORY_GB")).toBeNull();
  });

  it("won't fit: an amber alert with the suggestions in order", () => {
    const suggestions = [
      "Raise SCVIEW_MEMORY_GB in the .env file in the scView folder from 8.0 to at least 12 GB.",
      "Close the 2 other open datasets (about 1.9 GB): restart scView and open only this one.",
      "Skip doublet detection (about 1.2 GB more at its peak).",
    ];
    render(<PreflightNote verdict={verdict("wont_fit", { suggestions })} />);
    const alert = screen.getByRole("alert");
    expect(alert.className).toContain("bg-amber-50");
    const items = screen.getAllByRole("listitem").map((li) => li.textContent);
    expect(items).toEqual(suggestions);
  });

  it("unknown memory: the sentence only", () => {
    render(
      <PreflightNote
        verdict={verdict("unknown", {
          available_bytes: null,
          limit_bytes: null,
          sentence: "About 1.0 GB at its peak (UMAP), a few seconds. scView could not read how much memory it has.",
        })}
      />,
    );
    expect(screen.getByText(/could not read how much memory/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("the tooltip breaks the estimate down: parts, free memory, other datasets, error band, VM caveat", () => {
    render(
      <PreflightNote
        verdict={verdict("tight", {
          cached: { count: 2, bytes: 1.9 * GB, frees_enough: true },
          notes: ["Not measured, estimated from what the step allocates: batch correction."],
        })}
      />,
    );
    fireEvent.mouseEnter(screen.getByRole("button", { name: "How this estimate is made" }));
    const tip = screen.getByRole("tooltip");
    expect(tip).toHaveTextContent("This dataset now");
    expect(tip).toHaveTextContent("0.9 GB");
    expect(tip).toHaveTextContent("At the peak (doublet detection)");
    expect(tip).toHaveTextContent("Free now: 5.2 GB of 8.0 GB (the SCVIEW_MEMORY_GB limit)");
    expect(tip).toHaveTextContent("Other datasets held in memory: 2, about 1.9 GB; closing them would free enough.");
    expect(tip).toHaveTextContent("within 30% of the measured peak");
    expect(tip).toHaveTextContent("about 10 times slower");
    expect(tip).toHaveTextContent("Not measured, estimated from what the step allocates: batch correction.");
  });

  it("a label names the sample", () => {
    render(<PreflightNote verdict={verdict("fits")} label="sampleA" />);
    expect(screen.getByText("sampleA:")).toBeInTheDocument();
  });
});

describe("PreflightRunNote", () => {
  it("asks for the selected steps and the parameters, debounced, then shows the note", async () => {
    vi.useFakeTimers();
    api.getRunPreflight.mockResolvedValue({ dataset_id: "d1", steps: [], verdict: verdict("wont_fit", { suggestions: ["Skip doublet detection"] }) });
    render(<PreflightRunNote datasetId="d1" steps={["qc_metrics", "doublet_detection"]} params={{ n_top_genes: 3000 }} />);
    expect(api.getRunPreflight).not.toHaveBeenCalled();
    await act(async () => {
      vi.advanceTimersByTime(350);
    });
    vi.useRealTimers();
    expect(api.getRunPreflight).toHaveBeenCalledTimes(1);
    const [id, steps, params] = api.getRunPreflight.mock.calls[0]!;
    expect([id, steps, params]).toEqual(["d1", ["qc_metrics", "doublet_detection"], { n_top_genes: 3000 }]);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Skip doublet detection"));
  });

  it("no steps: no request and no note", async () => {
    render(<PreflightRunNote datasetId="d1" steps={[]} params={null} />);
    await new Promise((r) => setTimeout(r, 400));
    expect(api.getRunPreflight).not.toHaveBeenCalled();
    expect(screen.queryByText(/GB/)).toBeNull();
  });

  it("an error leaves no note (the estimate is advisory)", async () => {
    api.getRunPreflight.mockRejectedValue(new Error("API error 404"));
    const { container } = render(<PreflightRunNote datasetId="d1" steps={["pca"]} params={null} />);
    await waitFor(() => expect(api.getRunPreflight).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });
});

describe("PreflightIngestNotes", () => {
  it("one note per sample, labelled when there are several, plus the merge", async () => {
    api.getIngestPreflight.mockResolvedValue({
      session_id: "s1",
      units: [
        { label: "A", format: "tenx_mex", verdict: verdict("fits", { sentence: "A fits." }) },
        { label: "B", format: "tenx_mex", verdict: verdict("tight", { sentence: "B is tight." }) },
      ],
      merge: verdict("wont_fit", { sentence: "The merge won't fit.", suggestions: ["Subsample"] }),
    });
    render(<PreflightIngestNotes sessionId="s1" />);
    await waitFor(() => expect(screen.getByText("The merge won't fit.")).toBeInTheDocument());
    expect(screen.getByText("A:")).toBeInTheDocument();
    expect(screen.getByText("B:")).toBeInTheDocument();
    expect(screen.getByText("Merging them:")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Subsample");
  });

  it("a single sample is not labelled", async () => {
    api.getIngestPreflight.mockResolvedValue({
      session_id: "s1",
      units: [{ label: "pbmc", format: "tenx_h5", verdict: verdict("fits", { sentence: "1,222 cells x 33,538 genes: fits." }) }],
      merge: null,
    });
    render(<PreflightIngestNotes sessionId="s1" />);
    await waitFor(() => expect(screen.getByText(/1,222 cells/)).toBeInTheDocument());
    expect(screen.queryByText("pbmc:")).toBeNull();
  });
});
