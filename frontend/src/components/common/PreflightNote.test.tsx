import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({ getRunPreflight: vi.fn(), getIngestPreflight: vi.fn() }));
vi.mock("@/api/preflight", () => api);

import type { LeidenChoice, PreflightVerdict, Umap3dChoice } from "@/api/preflight";
import { FastLeidenOffer, PreflightIngestNotes, PreflightNote, PreflightRunNote, Umap3dNote } from "./PreflightNote";

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
    expect(screen.queryByText(/Raise SCVIEW_MEMORY_GB/)).toBeNull();
    fireEvent.mouseEnter(screen.getByRole("button", { name: "How this estimate is made" }));
    expect(screen.getByRole("tooltip")).toHaveTextContent("If it does not fit: Raise SCVIEW_MEMORY_GB");
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

function leiden(extra: Partial<LeidenChoice> = {}): LeidenChoice {
  return {
    n_cells: 270_000,
    full_seconds: 3500,
    fast_seconds: 70,
    full_text: "roughly 29 min to 1.9 h",
    fast_text: "roughly 35 s to 2 min",
    offer_seconds: 600,
    offer_fast: true,
    iterations: -1,
    ...extra,
  };
}

describe("FastLeidenOffer", () => {
  it("over ten minutes: both times and a one-click opt-in, with what it changes in a tooltip", () => {
    const onChange = vi.fn();
    render(<FastLeidenOffer choice={leiden()} onChange={onChange} />);
    const offer = screen.getByTestId("fast-leiden-offer");
    expect(offer).toHaveTextContent(
      "Clustering 270,000 cells with Leiden's full optimisation may take roughly 29 min to 1.9 h. Fast Leiden: roughly 35 s to 2 min.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Use fast Leiden" }));
    expect(onChange).toHaveBeenCalledWith(2);
    fireEvent.mouseEnter(screen.getByRole("button", { name: "What fast Leiden changes" }));
    const tip = screen.getByRole("tooltip");
    expect(tip).toHaveTextContent("same Leiden algorithm with the same seed, stopped after two passes");
    expect(tip).toHaveTextContent("ARI 0.93-0.97");
    expect(tip).toHaveTextContent("different seeds agree with each other (ARI 0.46-0.60)");
    expect(tip).toHaveTextContent("0.1-0.35 %");
    expect(tip).toHaveTextContent("History and the replay script (sc.tl.leiden(..., n_iterations=2))");
  });

  it("once chosen: says so and offers the way back", () => {
    const onChange = vi.fn();
    render(<FastLeidenOffer choice={leiden({ iterations: 2 })} onChange={onChange} />);
    expect(screen.getByTestId("fast-leiden-offer")).toHaveTextContent(
      "Clustering 270,000 cells with fast Leiden: roughly 35 s to 2 min (the full optimisation: roughly 29 min to 1.9 h).",
    );
    fireEvent.click(screen.getByRole("button", { name: "Use the full optimisation" }));
    expect(onChange).toHaveBeenCalledWith(-1);
  });

  it("a quick clustering: no offer", () => {
    const { container } = render(<FastLeidenOffer choice={leiden({ offer_fast: false })} onChange={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("PreflightRunNote and fast Leiden", () => {
  it("shows the offer from the run estimate and sets the clustering param", async () => {
    api.getRunPreflight.mockResolvedValue({ dataset_id: "d1", steps: [], verdict: verdict("fits"), leiden: leiden() });
    const onLeiden = vi.fn();
    render(
      <PreflightRunNote datasetId="d1" steps={["neighbors", "clustering"]} params={null} onLeidenIterations={onLeiden} />,
    );
    await waitFor(() => expect(screen.getByTestId("fast-leiden-offer")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Use fast Leiden" }));
    expect(onLeiden).toHaveBeenCalledWith(2);
  });

  it("an older backend without the field: the note alone", async () => {
    api.getRunPreflight.mockResolvedValue({ dataset_id: "d1", steps: [], verdict: verdict("fits") });
    render(<PreflightRunNote datasetId="d1" steps={["clustering"]} params={null} />);
    await waitFor(() => expect(screen.getByText(/About 3.1 GB more than now/)).toBeInTheDocument());
    expect(screen.queryByTestId("fast-leiden-offer")).toBeNull();
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

function umap3d(extra: Partial<Umap3dChoice> = {}): Umap3dChoice {
  return {
    n_cells: 1_160_000, setting: "auto", on: false, default_on: false,
    extra_seconds: 1470, extra_text: "roughly 12 to 49 min", offer: true, ...extra,
  };
}

describe("Umap3dNote", () => {
  it("large dataset: no 3D UMAP, its time, and one click to add it", () => {
    const onChange = vi.fn();
    render(<Umap3dNote choice={umap3d()} onChange={onChange} />);
    expect(screen.getByTestId("umap-3d-note")).toHaveTextContent(
      "Large dataset: no 3D UMAP in this run (the 3D view needs one). Adding it would take roughly 12 to 49 min more.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Add the 3D UMAP" }));
    expect(onChange).toHaveBeenCalledWith("on");
  });

  it("added: what it costs, and back to automatic", () => {
    const onChange = vi.fn();
    render(<Umap3dNote choice={umap3d({ setting: "on", on: true })} onChange={onChange} />);
    expect(screen.getByTestId("umap-3d-note")).toHaveTextContent(
      "The 3D UMAP of 1,160,000 cells adds roughly 12 to 49 min to the run.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Leave it out" }));
    expect(onChange).toHaveBeenCalledWith("auto");
  });

  it("in memory and quick: no note", () => {
    const { container } = render(<Umap3dNote choice={umap3d({ on: true, default_on: true, offer: false })} />);
    expect(container).toBeEmptyDOMElement();
  });
});
