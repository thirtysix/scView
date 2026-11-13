import { describe, it, expect } from "vitest";
import type { ObsColumnInfo } from "@/api/types";
import { isScoreColumn, savedGeneSetMatches, scoreDetail, scoreTitle, scoreTooltip } from "./savedScores";

describe("program and chromVAR scores", () => {
  const program = (earlier: boolean, method = "NMF usage share"): ObsColumnInfo => ({
    name: "program_k12_3",
    dtype: "float32",
    n_unique: 100,
    saved_score: {
      kind: "program", name: "P3 (k 12)", method, k: 12, program: 3, run_id: "r1",
      top_genes: ["GZMB", "PRF1", "NKG7", "GNLY"], earlier_run: earlier,
    },
  });
  const motif: ObsColumnInfo = {
    name: "chromvar_SPI1_MA0080.6",
    dtype: "float32",
    n_unique: 100,
    saved_score: { kind: "chromvar", name: "SPI1", method: "chromVAR deviation z", motif_id: "MA0080.6", run_id: "c1", earlier_run: false },
  };

  it("names a program by its number and run, and says what it holds with three top genes", () => {
    expect(isScoreColumn(program(false))).toBe(true);
    expect(scoreTitle(program(false))).toBe("P3 (k 12)");
    expect(scoreDetail(program(false))).toBe("Program usage: GZMB, PRF1, NKG7");
    expect(scoreDetail(program(false, "NMF usage"))).toBe("Program usage (unscaled): GZMB, PRF1, NKG7");
    expect(scoreTooltip(program(false))).toBe("P3 (k 12), Program usage: GZMB, PRF1, NKG7 (column program_k12_3)");
  });

  it("says a program is from an earlier NMF run, in its detail and its hover text", () => {
    expect(scoreDetail(program(true))).toBe("Program usage: GZMB, PRF1, NKG7, earlier run");
    expect(scoreTooltip(program(true))).toMatch(/From an earlier gene programs \(NMF\) run: the dataset's programs now come from a later run\.$/);
  });

  it("names a motif score by its motif and JASPAR id", () => {
    expect(scoreTitle(motif)).toBe("SPI1");
    expect(scoreDetail(motif)).toBe("chromVAR z (MA0080.6)");
    const earlier = { ...motif, saved_score: { ...motif.saved_score!, earlier_run: true } };
    expect(scoreTooltip(earlier)).toMatch(/From an earlier chromVAR run/);
  });
});

describe("savedGeneSetMatches", () => {
  const saved = {
    score_IFN: { kind: "geneset", name: "IFN", method: "score_genes", genes: ["ISG15", "MX1"] },
    activity_STAT1: { kind: "activity", name: "IFN", method: "ulm" },
  };

  it("matches the same set, method and genes in any order", () => {
    expect(savedGeneSetMatches(saved, "IFN", "score_genes", ["MX1", "ISG15"])).toBe(true);
  });

  it("does not match another method, other genes, or no saved entry", () => {
    expect(savedGeneSetMatches(saved, "IFN", "aucell", ["ISG15", "MX1"])).toBe(false);
    expect(savedGeneSetMatches(saved, "IFN", "score_genes", ["ISG15"])).toBe(false);
    expect(savedGeneSetMatches(saved, "IFN", "score_genes", ["ISG15", "MX1", "IFIT1"])).toBe(false);
    expect(savedGeneSetMatches(saved, "OTHER", "score_genes", ["ISG15"])).toBe(false);
    expect(savedGeneSetMatches({}, "IFN", "score_genes", ["ISG15"])).toBe(false);
  });

  it("reads a record without a method as score_genes (the default)", () => {
    const old = { score_IFN: { kind: "geneset", name: "IFN", genes: ["ISG15"] } };
    expect(savedGeneSetMatches(old, "IFN", "score_genes", ["ISG15"])).toBe(true);
  });
});
