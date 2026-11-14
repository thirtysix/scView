import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { HashingCall, HashingPlotData, HashingSummary } from "@/api/hashing";

const { getHashing, getHashingPlotData, runHashingDemux, saveHashingSamples, refresh } = vi.hoisted(() => ({
  getHashing: vi.fn(),
  getHashingPlotData: vi.fn(),
  runHashingDemux: vi.fn(),
  saveHashingSamples: vi.fn(),
  refresh: vi.fn(),
}));
vi.mock("@/api/hashing", () => ({ getHashing, getHashingPlotData, runHashingDemux, saveHashingSamples }));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));
// Plotly needs a real canvas; the traces it is handed are what matter here.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { name?: string; type: string }[] }) => (
    <div data-testid="plot">{data.map((t) => t.name ?? t.type).join("|")}</div>
  ),
}));

import { Hashing } from "./Hashing";
import { tagFitText } from "@/lib/hashing";

const BASE: HashingSummary = {
  n_cells: 1000,
  available: true,
  tags: [],
  tag_sources: {},
  proteins: ["CD3", "CD19", "Hashtag1", "Hashtag2"],
  suggested: ["Hashtag1", "Hashtag2"],
  moved_from_protein: [],
  methods: [
    { id: "htodemux", label: "HTODemux (Seurat)" },
    { id: "hashedDrops", label: "hashedDrops (DropletUtils)" },
  ],
  defaults: { method: "htodemux", quantile: 0.99 },
  limits: { min_tags: 2, max_tags: 50, min_cells: 50 },
  default_sample_column: "hash_sample",
  call: null,
  samples: null,
  scrublet: null,
  doublet_source: null,
};

const CALL: HashingCall = {
  method: "htodemux",
  label: "HTODemux (Seurat)",
  tags: ["Hashtag1", "Hashtag2"],
  params: {},
  thresholds: { positive_quantile: 0.99 },
  per_tag: [
    { tag: "Hashtag1", cutoff: 88, n_positive: 500 },
    { tag: "Hashtag2", cutoff: 121, n_positive: 450 },
  ],
  when: "2026-10-22T01:00:00Z",
  n_cells_at_call: 1000,
  counts: { Singlet: 750, Doublet: 150, Negative: 100 },
  per_call: { Hashtag1: 400, Hashtag2: 350, Doublet: 150, Negative: 100 },
  n_uncalled: 0,
};

const CALLED: HashingSummary = {
  ...BASE,
  tags: ["Hashtag1", "Hashtag2"],
  tag_sources: { Hashtag1: "Antibody Capture", Hashtag2: "Antibody Capture" },
  proteins: ["CD3", "CD19"],
  suggested: [],
  moved_from_protein: ["Hashtag1", "Hashtag2"],
  call: CALL,
  doublet_source: "hash_class",
};

const PLOT: HashingPlotData = {
  available: true,
  tags: ["Hashtag1", "Hashtag2"],
  n_cells: 1000,
  method: "htodemux",
  per_tag: ["Hashtag1", "Hashtag2"].map((tag) => ({
    tag,
    bin_edges: [0, 1, 2],
    counts: { tag: [0, 5], doublet: [1, 1], other: [9, 0] },
    cutoff: 88,
    cutoff_clr: 1.2,
    n_above: 5,
    median_count: 12,
  })),
  heatmap: { tags: ["Hashtag1", "Hashtag2"], calls: ["Hashtag1", "Hashtag2"], n: [400, 350], mean_clr: [[2, 0.1], [0.1, 2]] },
  lfc: null,
};

describe("Hashing", () => {
  beforeEach(() => {
    getHashing.mockReset().mockResolvedValue(BASE);
    getHashingPlotData.mockReset().mockResolvedValue(PLOT);
    runHashingDemux.mockReset().mockResolvedValue(CALLED);
    saveHashingSamples.mockReset();
    refresh.mockReset().mockResolvedValue({});
  });

  it("stays hidden when the dataset has no tags and no antibodies", async () => {
    getHashing.mockResolvedValue({ ...BASE, available: false, proteins: [], suggested: [] });
    const { container } = render(<Hashing datasetId="pbmc" />);
    await waitFor(() => expect(getHashing).toHaveBeenCalledWith("pbmc"));
    expect(container.firstChild).toBeNull();
  });

  it("says when no antibody looks like a hashtag", async () => {
    getHashing.mockResolvedValue({ ...BASE, suggested: [] });
    render(<Hashing datasetId="cite" />);
    expect(await screen.findByText(/No hashtags found/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    expect(screen.getByTestId("hashing-none")).toBeTruthy();
    expect(screen.getByRole("button", { name: /Demultiplex/ }).hasAttribute("disabled")).toBe(true);
  });

  it("suggests the hashtags, runs the chosen method and refreshes the doublet views", async () => {
    const onSaved = vi.fn();
    render(<Hashing datasetId="d1" onSaved={onSaved} />);
    expect(await screen.findByText(/2 antibody features look like hashtags/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    const box = screen.getByTestId("hashing-tags");
    expect((within(box).getByLabelText("Hashtag1") as HTMLInputElement).checked).toBe(true);
    expect((within(box).getByLabelText("CD3") as HTMLInputElement).checked).toBe(false);
    expect(screen.getByText(/will leave the protein modality/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Demultiplex" }));
    await waitFor(() =>
      expect(runHashingDemux).toHaveBeenCalledWith("d1", {
        tags: ["Hashtag1", "Hashtag2"],
        method: "htodemux",
        quantile: 0.99,
        split_by: null,
      }),
    );
    expect(await screen.findByTestId("hashing-summary")).toBeTruthy();
    expect(screen.getByTestId("hashing-summary").textContent).toContain("750 singlets (75.0%)");
    expect(refresh).toHaveBeenCalledWith("d1", "hash_call");
    expect(onSaved).toHaveBeenCalled();
    expect(screen.getByText(/use these hashing doublets/)).toBeTruthy();
    // The cutoffs and the per-tag plots with the saved call.
    expect(within(screen.getByTestId("hashing-table")).getByText("> 121")).toBeTruthy();
    await waitFor(() => expect(getHashingPlotData).toHaveBeenCalledWith("d1"));
    const plots = await screen.findAllByTestId("plot");
    expect(plots[0]?.textContent).toBe("other cells|doublets|this tag's singlets|other cells|doublets|this tag's singlets");
    expect(plots[1]?.textContent).toBe("heatmap");
  });

  it("sends hashedDrops without a quantile and shows the server's reason on failure", async () => {
    runHashingDemux.mockRejectedValue(new Error('API error 400: {"detail":"A group of 12 cells has no Hashtag2 counts at all"}'));
    render(<Hashing datasetId="d1" />);
    await screen.findByText(/look like hashtags/);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    fireEvent.change(screen.getByLabelText("Method"), { target: { value: "hashedDrops" } });
    expect(screen.queryByLabelText("Positive quantile")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Demultiplex" }));
    await waitFor(() =>
      expect(runHashingDemux).toHaveBeenCalledWith("d1", {
        tags: ["Hashtag1", "Hashtag2"],
        method: "hashedDrops",
        split_by: null,
      }),
    );
    expect((await screen.findByRole("alert")).textContent).toContain("has no Hashtag2 counts");
  });

  it("starts from the suggested lane column, and offers Save anyway when the checks refuse a call", async () => {
    getHashing.mockResolvedValue({
      ...BASE,
      split_columns: ["orig.ident", "replicate"],
      split_suggestion: { column: "orig.ident", n_groups: 8 },
    });
    runHashingDemux
      .mockRejectedValueOnce(
        new Error('API error 409: {"detail":"Not saved: rep1-tx has an ambient level of 0. To save it as it is, demultiplex again with \'Save anyway\'."}'),
      )
      .mockResolvedValueOnce({
        ...CALLED,
        call: { ...CALL, warnings: ["rep1-tx has an ambient level of 0"], saved_with_problems: true },
      });
    render(<Hashing datasetId="d1" />);
    await screen.findByText(/look like hashtags/);
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    expect(screen.getByLabelText("Within")).toHaveProperty("value", "orig.ident");
    expect(screen.getByRole("option", { name: "orig.ident (suggested)" })).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Within"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Demultiplex" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Not saved: rep1-tx has an ambient level of 0");
    expect(screen.getByRole("alert").textContent).not.toContain("Could not demultiplex");
    fireEvent.click(screen.getByRole("button", { name: "Save anyway" }));
    await waitFor(() =>
      expect(runHashingDemux).toHaveBeenLastCalledWith("d1", {
        tags: ["Hashtag1", "Hashtag2"],
        method: "htodemux",
        quantile: 0.99,
        split_by: null,
        allow_warnings: true,
      }),
    );
    const warn = await screen.findByTestId("hashing-warnings");
    expect(warn.textContent).toContain("Saved although the checks failed");
    expect(warn.textContent).toContain("rep1-tx has an ambient level of 0");
  });

  it("shows a split call's cutoffs as a range over its channels", () => {
    const split: HashingCall = {
      ...CALL,
      split_by: "lane",
      per_tag: [
        { tag: "Hashtag1", cutoff: 80, group: "L1" },
        { tag: "Hashtag1", cutoff: 95, group: "L2" },
        { tag: "Hashtag2", cutoff: 121, group: "L1" },
      ],
    };
    expect(tagFitText(split, "Hashtag1")).toEqual({ text: "> 80–95", title: "L1: 80; L2: 95" });
    expect(tagFitText(split, "Hashtag2")).toEqual({ text: "> 121", title: "L1: 121" });
    expect(tagFitText(CALL, "Hashtag1")).toEqual({ text: "> 88", title: "" });
    expect(tagFitText({ ...CALL, method: "hashedDrops", per_tag: [{ tag: "Hashtag1", ambient: 0 }] }, "Hashtag1").text).toBe("0.0");
    expect(tagFitText(CALL, "nope").text).toBe("—");
  });

  it("writes the sample column from the mapping table", async () => {
    getHashing.mockResolvedValue(CALLED);
    saveHashingSamples.mockResolvedValue({
      ...CALLED,
      samples: { column: "donor", mapping: { Hashtag1: "donor A", Hashtag2: "Hashtag2" }, counts: { "donor A": 400, Hashtag2: 350 }, n_unassigned: 250 },
    });
    render(<Hashing datasetId="d1" />);
    expect(await screen.findByTestId("hashing-badge")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    fireEvent.change(screen.getByLabelText("Sample name for Hashtag1"), { target: { value: "donor A" } });
    fireEvent.change(screen.getByLabelText("Sample column"), { target: { value: "bad/name" } });
    expect(screen.getByRole("button", { name: /Use as sample column/ }).hasAttribute("disabled")).toBe(true);
    fireEvent.change(screen.getByLabelText("Sample column"), { target: { value: "donor" } });
    fireEvent.click(screen.getByRole("button", { name: /Use as sample column/ }));
    await waitFor(() =>
      expect(saveHashingSamples).toHaveBeenCalledWith("d1", { Hashtag1: "donor A", Hashtag2: "Hashtag2" }, "donor"),
    );
    expect((await screen.findByTestId("hashing-saved")).textContent).toContain("Saved donor: 400 donor A, 350 Hashtag2; 250 doublets");
    expect(refresh).toHaveBeenCalledWith("d1", "donor");
  });
});
