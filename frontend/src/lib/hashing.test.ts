import { describe, it, expect } from "vitest";
import type { HashingCall, HashingSummary, HashingTagHist } from "@/api/hashing";
import {
  classSentence,
  CUTOFF_COLOR,
  DOUBLET_COLOR,
  doubletSourceText,
  heatmapTrace,
  initialMapping,
  initialSelection,
  lfcFigure,
  MAX_TAG_PLOTS,
  orderedSelection,
  scrubletText,
  tagHistFigure,
  validColumnName,
} from "./hashing";

const summary = (over: Partial<HashingSummary> = {}): HashingSummary => ({
  n_cells: 100,
  available: true,
  tags: ["CMO301", "CMO302"],
  tag_sources: { CMO301: "Multiplexing Capture", CMO302: "Multiplexing Capture" },
  proteins: ["CD3", "Hashtag1", "Hashtag2"],
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
  ...over,
});

const call = (over: Partial<HashingCall> = {}): HashingCall => ({
  method: "htodemux",
  label: "HTODemux (Seurat)",
  tags: ["Hashtag1", "Hashtag2"],
  params: {},
  thresholds: { positive_quantile: 0.99 },
  per_tag: [
    { tag: "Hashtag1", cutoff: 88 },
    { tag: "Hashtag2", cutoff: 121 },
  ],
  when: "2026-10-22T00:00:00Z",
  n_cells_at_call: 100,
  counts: { Singlet: 75, Doublet: 15, Negative: 10 },
  per_call: { Hashtag1: 40, Hashtag2: 35, Doublet: 15, Negative: 10 },
  n_uncalled: 0,
  ...over,
});

describe("hashing helpers", () => {
  it("starts with every tag count and the suggested hashtags, or the saved call's tags", () => {
    expect([...initialSelection(summary())].sort()).toEqual(["CMO301", "CMO302", "Hashtag1", "Hashtag2"]);
    const s = summary({ call: call({ tags: ["CMO301", "Hashtag2"] }) });
    expect([...initialSelection(s)].sort()).toEqual(["CMO301", "Hashtag2"]);
    // Tag counts first, then antibodies, in the listing's order.
    expect(orderedSelection(s, new Set(["Hashtag2", "CD3", "CMO302"]))).toEqual(["CMO302", "CD3", "Hashtag2"]);
  });

  it("says the class counts with their shares", () => {
    expect(classSentence({ Singlet: 15223, Doublet: 3157, Negative: 1858 })).toBe(
      `${(15223).toLocaleString()} singlets (75.2%), ${(3157).toLocaleString()} doublets (15.6%) and ${(1858).toLocaleString()} negatives (9.2%)`,
    );
    expect(classSentence({ Singlet: 0, Doublet: 0, Negative: 0 })).toContain("0 singlets (0%)");
  });

  it("checks column names as the backend does", () => {
    expect(validColumnName("hash_sample")).toBe(true);
    expect(validColumnName("donor id")).toBe(true);
    for (const bad of ["", "1donor", "a/b", "scview_x", "hash_call", "x".repeat(65)]) {
      expect(validColumnName(bad)).toBe(false);
    }
  });

  it("names the doublet source and compares with Scrublet", () => {
    expect(doubletSourceText("hash_class")).toMatch(/use these hashing doublets/);
    expect(doubletSourceText("predicted_doublet")).toMatch(/Scrublet's calls/);
    expect(doubletSourceText(null)).toBe("");
    expect(scrubletText(null)).toBeNull();
    const t = scrubletText({
      Singlet: { scrublet_doublet: 12, scrublet_singlet: 700 },
      Doublet: { scrublet_doublet: 30, scrublet_singlet: 70 },
      Negative: { scrublet_doublet: 1, scrublet_singlet: 50 },
    });
    expect(t).toContain("Of 100 hashing doublets, Scrublet also calls 30 (30.0%)");
    expect(t).toContain("12 of the hashing singlets");
  });

  it("fills the mapping from the saved one, else the tag names", () => {
    expect(initialMapping(call(), undefined)).toEqual({ Hashtag1: "Hashtag1", Hashtag2: "Hashtag2" });
    expect(initialMapping(call(), { Hashtag1: "donor A" })).toEqual({ Hashtag1: "donor A", Hashtag2: "Hashtag2" });
  });
});

const hist = (tag: string, cutoffClr: number | null): HashingTagHist => ({
  tag,
  bin_edges: [0, 1, 2, 3],
  counts: { tag: [0, 1, 5], doublet: [0, 2, 1], other: [9, 1, 0] },
  cutoff: cutoffClr == null ? null : 88,
  cutoff_clr: cutoffClr,
  n_above: 7,
  median_count: 20,
});

describe("hashing figures", () => {
  it("stacks each tag's histogram by call, one subplot per tag, with the cutoff line", () => {
    const fig = tagHistFigure([hist("A", 1.5), hist("B", null), hist("C", 2.2)], () => "rgb(1,2,3)");
    expect(fig.data).toHaveLength(9);
    const first = fig.data[0] as { x: number[]; width: number; marker: { color: string }; xaxis: string };
    expect(first.x).toEqual([0.5, 1.5, 2.5]);
    expect(first.width).toBe(1);
    expect(first.marker.color).toBe("rgb(1,2,3)"); // the tag's own colour
    expect((fig.data[1] as { marker: { color: string } }).marker.color).toBe(DOUBLET_COLOR);
    expect(fig.data.filter((t) => (t as { showlegend: boolean }).showlegend)).toHaveLength(3);
    expect(fig.data.map((t) => (t as { xaxis: string }).xaxis)).toEqual(["x", "x", "x", "x2", "x2", "x2", "x3", "x3", "x3"]);
    const shapes = fig.layout.shapes as { x0: number; xref: string; line: { color: string } }[];
    expect(shapes.map((s) => [s.xref, s.x0])).toEqual([["x", 1.5], ["x3", 2.2]]); // B has no cutoff
    expect(shapes[0]?.line.color).toBe(CUTOFF_COLOR);
    // Two columns: A and B on row 1, C on row 2.
    const ax = fig.layout as Record<string, { domain: [number, number] }>;
    const dom = (k: string): [number, number] => ax[k]?.domain ?? [NaN, NaN];
    expect(dom("yaxis")[1]).toBe(1);
    expect(dom("yaxis2")).toEqual(dom("yaxis"));
    expect(dom("yaxis3")[1]).toBeLessThan(dom("yaxis")[0]);
    expect(dom("xaxis3")[0]).toBe(0);
  });

  it("before a call, draws one histogram per tag; a large panel is cut", () => {
    const pre: HashingTagHist = { ...hist("A", null), counts: { all: [3, 2, 1] } };
    expect(tagHistFigure([pre], () => "red").data).toHaveLength(1);
    const many = Array.from({ length: MAX_TAG_PLOTS + 5 }, (_, i) => hist(`T${i}`, 1));
    expect(tagHistFigure(many, () => "red").data).toHaveLength(MAX_TAG_PLOTS * 3);
  });

  it("labels the heatmap columns with their n", () => {
    const t = heatmapTrace(
      { tags: ["A", "B"], calls: ["A", "B", "Doublet"], n: [40, 35, 1500], mean_clr: [[2, 0.1, 1.5], [0.1, 2, 1.4]] },
      [[0, "rgb(0,0,0)"], [1, "rgb(255,255,255)"]],
    );
    expect(t.x).toEqual(["A (40)", "B (35)", `Doublet (${(1500).toLocaleString()})`]);
    expect(t.y).toEqual(["A", "B"]);
    expect(t.z).toEqual([[2, 0.1, 1.5], [0.1, 2, 1.4]]);
  });

  it("draws hashedDrops' two fold changes with their thresholds", () => {
    const f = lfcFigure({
      lfc: { bin_edges: [0, 2, 4], counts: [10, 90], threshold: 2 },
      lfc2: { bin_edges: [-1, 1, 3], counts: [95, 5], threshold: 2.69 },
    });
    expect(f.data).toHaveLength(2);
    expect((f.layout.shapes as { x0: number; xref: string }[]).map((s) => [s.xref, s.x0])).toEqual([["x", 2], ["x2", 2.69]]);
    expect((f.layout.annotations as { text: string }[])[1]?.text).toBe("doublet above 2.69");
    expect(lfcFigure({ lfc: null, lfc2: null }).data).toHaveLength(0);
  });
});
