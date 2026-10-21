import { describe, expect, it } from "vitest";
import type { DEOverlapSet, DEResultSummary } from "@/api/deOverlap";
import { columnTitle, defaultChecked, runOrder, setRuleText, setTitle } from "./deOverlap";

const r = (id: string, minute: number): DEResultSummary => ({
  result_id: id,
  label: id,
  method: "wilcoxon",
  created_at: `2026-10-21T10:${String(minute).padStart(2, "0")}:00+00:00`,
  n_tested: 100,
  has_pvalues: true,
  rule: "padj",
  n_sig_up: 1,
  n_sig_down: 1,
});

const set = (over: Partial<DEOverlapSet>): DEOverlapSet => ({
  index: 0,
  name: "CD14 Mono",
  kind: "de",
  result_id: "m",
  method: "wilcoxon",
  rule: "padj",
  n: 1802,
  n_up: 711,
  n_down: 1091,
  n_tested: 7273,
  n_not_tested: null,
  ...over,
});
const CUTS = { padj_max: 0.05, lfc_min: 0.25, logreg_top_n: 50 };

describe("deOverlap helpers", () => {
  it("order results as they were run and tick the newest four", () => {
    const six = [r("f", 6), r("a", 1), r("c", 3), r("b", 2), r("e", 5), r("d", 4)];
    expect(runOrder(six).map((x) => x.result_id)).toEqual(["a", "b", "c", "d", "e", "f"]);
    expect(defaultChecked(six)).toEqual(["c", "d", "e", "f"]);
    expect(defaultChecked(six.slice(0, 2))).toEqual(["a", "f"]);
  });

  it("say how each set was chosen", () => {
    expect(setRuleText(set({}), CUTS)).toBe("adjusted p ≤ 0.05 and |log2FC| ≥ 0.25");
    expect(setRuleText(set({ rule: "padj_bh_here" }), CUTS)).toContain("raw p adjusted here, BH");
    expect(setRuleText(set({ rule: "top_n" }), CUTS)).toContain("top 50 genes per direction by |coefficient|");
    expect(setRuleText(set({ kind: "extra", rule: null }), CUTS)).toBe("a gene list (no direction)");
    expect(setTitle(set({}), { ...CUTS } as never)).toBe(
      "CD14 Mono: 1,802 genes (711 up, 1,091 down) of 7,273 tested; adjusted p ≤ 0.05 and |log2FC| ≥ 0.25",
    );
    expect(setTitle(set({ kind: "extra", name: "ISGs", n: 3, n_not_tested: 1 }), { ...CUTS } as never)).toBe(
      "ISGs: 3 genes, 1 tested in no result; a gene list (no direction)",
    );
  });

  it("name an intersection with its counts and directions", () => {
    const sets = [set({}), set({ index: 1, name: "B" }), set({ index: 2, name: "Mine", kind: "extra" })];
    expect(columnTitle({ sets: [0, 1], degree: 2, n: 136, n_up: 100, n_down: 36, genes: [] }, sets)).toBe(
      "136 genes in CD14 Mono and B only (100 up, 36 down)",
    );
    expect(columnTitle({ sets: [2], degree: 1, n: 1, n_up: 0, n_down: 0, genes: [] }, sets)).toBe("1 gene in only Mine");
  });
});
