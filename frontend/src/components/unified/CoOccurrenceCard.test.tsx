import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { CoOccurrenceInfo } from "@/api/spatialDomains";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSpatialDomainsRunStore } from "@/stores/spatialDomainsRunStore";
import { SPATIAL_HELP } from "@/lib/help/spatial";
import { hoverTip, unhover } from "@/test/hoverTip";
import { CoOccurrenceCard } from "./CoOccurrenceCard";

const getCoOccurrence = vi.fn();
const runCoOccurrence = vi.fn();
vi.mock("@/api/spatialDomains", async (orig) => ({
  ...(await orig<typeof import("@/api/spatialDomains")>()),
  getCoOccurrence: (...a: unknown[]) => getCoOccurrence(...a),
  runCoOccurrence: (...a: unknown[]) => runCoOccurrence(...a),
}));

interface Drawn {
  data: { type: string; name?: string; x?: unknown[]; y?: unknown[]; z?: unknown[][]; line?: { color?: string }; zmin?: number; zmax?: number }[];
  layout: { xaxis?: { title?: { text?: string }; type?: string }; yaxis?: { type?: string }; shapes?: { y0: number }[] };
}
const drawn: Drawn[] = [];
vi.mock("@/components/plots/Plot", () => ({
  default: (p: Drawn) => {
    drawn.push(p);
    return <div data-testid="plot" />;
  },
}));

const verdict = {
  level: "fits",
  sentence: "Co-occurrence: about 0.3 GB more at its peak, roughly 30 to 40 s. 6 GB free.",
  need_bytes: 1,
  peak_bytes: 1,
  peak_stage: "counting pairs",
  available_bytes: 1,
  limit_bytes: 1,
  used_bytes: 0,
  limited: false,
  source: "machine",
  seconds: [30, 40],
  cached: { count: 0, bytes: 0, frees_enough: false },
  suggestions: [],
  error_band: 0.2,
  breakdown: [],
  notes: [],
};

const plan = {
  n_obs: 30000,
  n_labelled: 30000,
  n_used: 30000,
  unit: "cell",
  column: "clusters",
  labels: ["a", "b"],
  interval: 50,
  thresholds: [10, 500, 1000],
  combine: "sum",
  n_splits: 1,
  libraries: ["s"],
  distance_unit: "um",
  um_per_unit: 1,
  distance_source: "",
  threads: 2,
  visited_pairs: 5,
  notes: [],
  estimate_seconds: 30,
  estimate_bytes: 1,
};

const saved = {
  available: true,
  columns: ["clusters"],
  column: "clusters",
  labels: ["a", "b"],
  distances: [500, 1000],
  distance_unit: "um",
  um_per_unit: 1,
  n_used: 30000,
  n_labelled: 30000,
  unit: "cell",
  occ: [
    [
      [1.5, 1.2],
      [0.5, 0.8],
    ],
    [
      [0.5, 0.8],
      [2, 1.1],
    ],
  ],
  count: [
    [
      [40, 90],
      [10, 30],
    ],
    [
      [10, 30],
      [8, 20],
    ],
  ],
  stale: false,
  stale_reason: null,
};

const info = (over: Partial<CoOccurrenceInfo> = {}): CoOccurrenceInfo =>
  ({
    result: { available: false, columns: [] },
    plan,
    verdict,
    reason: null,
    defaults: { interval: 50, max_obs: 500000, split_above: 22897 },
    running: false,
    ...over,
  }) as CoOccurrenceInfo;

const columns = [
  { name: "clusters", n_unique: 2 },
  { name: "other", n_unique: 3 },
];

beforeEach(() => {
  drawn.length = 0;
  getCoOccurrence.mockReset();
  runCoOccurrence.mockReset();
  useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: { id: "d1", obs_columns: [] } as never });
  useSettingsStore.setState({ colorBy: "other" });
  useSpatialDomainsRunStore.getState().reset();
});

describe("CoOccurrenceCard", () => {
  it("defaults to the current colour-by, says what a run needs, and starts with those settings", async () => {
    getCoOccurrence.mockResolvedValue(info());
    runCoOccurrence.mockResolvedValue(saved);
    render(<CoOccurrenceCard columns={columns} labelDefault="clusters" />);
    await screen.findByText(/about 0\.3 GB more/);
    expect(getCoOccurrence).toHaveBeenCalledWith("d1", { column: "other", interval: 50, combine: "sum", max_distance: undefined });
    fireEvent.click(screen.getByRole("button", { name: "Compute co-occurrence" }));
    await waitFor(() => expect(runCoOccurrence).toHaveBeenCalled());
    expect(runCoOccurrence.mock.calls[0]?.slice(0, 2)).toEqual([
      "d1",
      { cluster_key: "other", interval: 50, combine: "sum", max_distance: undefined },
    ]);
    await waitFor(() => expect(getCoOccurrence.mock.calls.length).toBeGreaterThan(1)); // reloads after a save
  });

  it("falls back when the colour-by is not a label column, and says why a run cannot start", async () => {
    useSettingsStore.setState({ colorBy: "n_counts" });
    getCoOccurrence.mockResolvedValue(info({ plan: null, verdict: null, reason: "Only 3 cells have a label." }));
    render(<CoOccurrenceCard columns={columns} labelDefault="clusters" />);
    await screen.findByText("Only 3 cells have a label.");
    expect(getCoOccurrence.mock.calls[0]?.[1].column).toBe("clusters");
    expect(screen.getByRole("button", { name: "Compute co-occurrence" })).toBeDisabled();
  });

  it("converts the reach to the positions' units and offers squidpy's splits only above its limit", async () => {
    getCoOccurrence.mockResolvedValue(info({ plan: { ...plan, um_per_unit: 0.5, n_labelled: 1000 } as CoOccurrenceInfo["plan"] }));
    const { unmount } = render(<CoOccurrenceCard columns={columns} labelDefault="clusters" />);
    await screen.findByText(/about 0\.3 GB more/);
    expect(screen.queryByText(/squidpy's default for this size/)).toBeNull();
    fireEvent.change(screen.getByRole("spinbutton", { name: /Up to/ }), { target: { value: "200" } });
    await waitFor(() => expect(getCoOccurrence.mock.lastCall?.[1].max_distance).toBe(400));
    unmount();
    getCoOccurrence.mockResolvedValue(info());
    render(<CoOccurrenceCard columns={columns} labelDefault="clusters" />);
    fireEvent.click(await screen.findByRole("checkbox"));
    await waitFor(() => expect(getCoOccurrence.mock.lastCall?.[1].combine).toBe("squidpy_splits"));
  });

  it("explains the card and its fields on hover of their labels, with no info icons", async () => {
    getCoOccurrence.mockResolvedValue(info({ result: saved as CoOccurrenceInfo["result"] }));
    render(<CoOccurrenceCard columns={columns} labelDefault="clusters" />);
    await screen.findByTestId("cooccurrence-results");
    const title = screen.getByText("Co-occurrence");
    expect(await hoverTip(title)).toMatch(/P\(b \| a\) \/ P\(b\)/);
    unhover(title);
    for (const [text, id] of [["Labels", "domains.cooccur-labels"], ["Distances", "domains.cooccur-distances"],
      ["Near which label", "domains.cooccur-near"]] as const) {
      const el = screen.getByText(text);
      expect(await hoverTip(el)).toBe(SPATIAL_HELP[id]);
      unhover(el);
    }
    expect(screen.getByRole("spinbutton", { name: "Distances" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^About/ })).toBeNull();
  });

  it("draws score against distance with a line at 1, and a heatmap with category axes", async () => {
    getCoOccurrence.mockResolvedValue(info({ result: saved as CoOccurrenceInfo["result"] }));
    render(<CoOccurrenceCard columns={columns} labelDefault="clusters" />);
    await screen.findByTestId("cooccurrence-results");
    const lines = drawn.filter((p) => p.data[0]?.type === "scatter").at(-1)!;
    expect(lines.data.map((t) => t.name)).toEqual(["a", "b"]);
    expect(lines.data[1]).toMatchObject({ x: [500, 1000], y: [0.5, 0.8] });
    expect(lines.data.every((t) => t.line?.color?.startsWith("rgb("))).toBe(true);
    expect(lines.layout.shapes?.[0]?.y0).toBe(1);
    expect(lines.layout.xaxis?.title?.text).toBe("Distance (µm)");
    const heat = drawn.filter((p) => p.data[0]?.type === "heatmap").at(-1)!;
    expect(heat.layout.xaxis?.type).toBe("category");
    expect(heat.layout.yaxis?.type).toBe("category");
    expect(heat.data[0]?.z).toEqual([
      [1.2, 0.8],
      [0.8, 1.1],
    ]); // the shortest distance with enough pairs
    expect(screen.getByTestId("cooccurrence-curves").textContent).toContain("not necessarily interacting");
    expect(screen.getByTestId("cooccurrence-curves").textContent).toContain("30,000 cells");
    expect(screen.getByTestId("cooccurrence-curves").textContent).toContain("Distances are in microns.");
    fireEvent.change(screen.getByLabelText("Conditioning label"), { target: { value: "b" } });
    expect(drawn.filter((p) => p.data[0]?.type === "scatter").at(-1)?.data[0]?.y).toEqual([0.5, 0.8]);
    fireEvent.change(screen.getByLabelText("Heatmap distance"), { target: { value: "0" } });
    expect(drawn.filter((p) => p.data[0]?.type === "heatmap").at(-1)?.data[0]?.z?.[0]).toEqual([1.5, 0.5]);
  });

  it("says when a saved run no longer fits", async () => {
    getCoOccurrence.mockResolvedValue(
      info({ result: { ...saved, stale: true, stale_reason: "The labels in 'clusters' changed since this ran: run it again." } as CoOccurrenceInfo["result"] }),
    );
    render(<CoOccurrenceCard columns={columns} labelDefault="clusters" />);
    await screen.findByText(/changed since this ran/);
  });
});
