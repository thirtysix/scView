import { describe, expect, it } from "vitest";
import { SUFFIX_COLUMN, type Diversity, type VdjFile, type VdjSummary } from "@/api/vdj";
import {
  chainText,
  diversityBars,
  droppedText,
  fileSentence,
  matchSentences,
  metricValue,
  needsSample,
  overlapCells,
  overlapHover,
  usageHover,
  valueLabel,
} from "./vdj";

const SUMMARY: VdjSummary = {
  has_vdj: true,
  n_cells: 10548,
  how_to: "",
  sample_choices: [],
  match: {
    cells_profiled: 10548,
    cells_with_chain: 6189,
    cells_with_contigs: 6189,
    cells_without_contigs: 4359,
    cells_not_profiled: 0,
    barcodes_without_cells: 26,
    barcodes_without_cells_usable: 26,
    ambiguous_barcodes: 0,
  },
};

const FILE: VdjFile = {
  name: "t.csv", sha256: "x", sample_column: SUFFIX_COLUMN, sample: "*-1:s2", stored_as: null, attached_at: "",
  format: "10x", n_rows: 12, n_contigs: 11, n_kept_chains: 9, dropped: { duplicate: 1, non_productive: 2, no_cdr3: 0 },
  kept_by_locus: {}, n_barcodes: 6, n_barcodes_usable: 5, matched: 4, matched_exact: 0, matched_core: 4,
  unmatched: 2, unmatched_usable: 1, ambiguous: 0, unmatched_examples: [], n_candidate_cells: 4, cells_without_contigs: 0,
};

describe("vdj wording", () => {
  it("states the join both ways with n", () => {
    const s = matchSentences(SUMMARY);
    expect(s[0]).toBe("6,189 of 10,548 cells have a usable TCR or BCR chain.");
    expect(s[1]).toBe("4,359 profiled cells had no contig.");
    expect(s[2]).toContain("When the tables were attached, 26 contig barcodes matched no cell");
    expect(matchSentences({ ...SUMMARY, match: { ...SUMMARY.match!, cells_not_profiled: 4 } })[1]).toContain(
      "4 cells are in samples with no contig table attached",
    );
  });

  it("describes a file, its sample and what was left out", () => {
    expect(fileSentence(FILE)).toBe(
      "t.csv for barcode suffix = …-1:s2: 9 of 11 contigs kept; 4 of 6 barcodes matched cells (4 by barcode core in decorated cell names), 2 did not.",
    );
    expect(droppedText(FILE.dropped)).toBe("1 repeated, 2 non-productive");
    expect(droppedText({ duplicate: 0 })).toBeNull();
    expect(valueLabel("sample", "*x")).toBe("*x");
  });

  it("asks for a sample when the cells come from several libraries", () => {
    expect(needsSample(SUMMARY)).toBe(false);
    expect(needsSample({ ...SUMMARY, sample_choices: [{ name: SUFFIX_COLUMN, values: ["*-1", "*-2"] }] })).toBe(true);
  });

  it("writes chains as locus + CDR3", () => {
    expect(
      chainText([
        { arm: "VJ", locus: "TRA", cdr3: "CAVA", v: "TRAV1-2", j: "TRAJ33" },
        { arm: "VDJ", locus: "TRB", cdr3: "CASSA", v: null, j: null },
      ]),
    ).toBe("TRA CAVA / TRB CASSA");
  });
});

const DIV: Diversity = {
  group_by: "donor", rarefy: true, depth: 20, seed: 0, n_draws: 20, min_cells: 10, caption: "",
  groups: [
    { name: "A", n_cells: 50, n_with_clonotype: 40, richness: 31, shannon: 3.2, simpson: 0.9, clonality: 0.1, gini: 0.2, note: null,
      rarefied: { richness: 17.5, richness_sd: 1.2, shannon: 2.8, shannon_sd: 0.1 } },
    { name: "B", n_cells: 20, n_with_clonotype: 20, richness: 16, shannon: 2.6, simpson: 0.9, clonality: 0.1, gini: 0.1, note: null,
      rarefied: { richness: 16, richness_sd: 0, shannon: 2.6, shannon_sd: 0 } },
    { name: "C", n_cells: 9, n_with_clonotype: 5, richness: null, shannon: null, simpson: null, clonality: null, gini: null,
      note: "fewer than 10 cells with a clonotype" },
  ],
};

describe("diversity bars", () => {
  it("uses the rarefied mean and SD when rarefied, and lists groups left out", () => {
    const b = diversityBars(DIV, "richness");
    expect(b.x).toEqual(["A", "B"]);
    expect(b.y).toEqual([17.5, 16]);
    expect(b.sd).toEqual([1.2, 0]);
    expect(b.left_out).toEqual(["C"]);
    expect(b.hover[0]).toContain("40 cells with a clonotype of 50");
    expect(metricValue(DIV.groups[0]!, "richness", false)).toEqual({ value: 31, sd: null });
  });
});

describe("heatmap hovers", () => {
  it("names both groups, the measure and the shared clonotypes", () => {
    const h = overlapHover({
      group_by: "g", metric: "morisita_horn", groups: ["A", "B"], n_cells: [40, 20], n_clonotypes: [31, 16],
      matrix: [[1, 0.69], [0.69, 1]], shared: [[31, 1], [1, 16]], n_groups_total: 2, n_groups_shown: 2,
    });
    expect(h[0]![1]).toContain("Morisita-Horn 0.690");
    expect(h[0]![1]).toContain("1 shared clonotypes");
    expect(h[0]![0]).toBe("<b>A</b>: 31 clonotypes, 40 cells");
    const u = usageHover({
      group_by: "g", chain: "TRB", gene: "v", count: "clonotypes", chains_present: { TRB: 60 }, genes: ["TRBV6-5"],
      n_genes_total: 1, groups: ["A"], n: [31], counts: [[16]], share: [[16 / 31]], n_groups_total: 1, n_groups_shown: 1,
    });
    expect(u[0]![0]).toBe("<b>A</b>, TRBV6-5<br>51.6% of 31 clonotypes (16)");
  });
});

describe("overlap cells", () => {
  it("blanks the diagonal and scales to the largest overlap between groups", () => {
    const base = { group_by: "g", metric: "jaccard" as const, groups: ["A", "B", "C"], n_cells: [1, 1, 1], n_clonotypes: [1, 1, 1],
      shared: [], n_groups_total: 3, n_groups_shown: 3 };
    const c = overlapCells({ ...base, matrix: [[1, 0.02, 0], [0.02, 1, 0.05], [0, 0.05, 1]] });
    expect(c.z).toEqual([[null, 0.02, 0], [0.02, null, 0.05], [0, 0.05, null]]);
    expect(c.zmax).toBe(0.05);
    expect(overlapCells({ ...base, matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }).zmax).toBe(1);
  });
});
