import { describe, it, expect } from "vitest";
import type { DEComposition } from "@/api/de";
import {
  balancedText,
  compositionCautionForCopilot,
  compositionRule,
  flaggedGroups,
  formatPoints,
  formatShare,
  shareLine,
} from "./deComposition";

const group = (group: string, share_a: number, share_b: number, flagged = false) => ({
  group, n_a: Math.round(share_a * 1000), n_b: Math.round(share_b * 1000), share_a, share_b,
  diff_points: Math.round((share_a - share_b) * 1000) / 10, flagged, one_sided: false,
});

const shifted: DEComposition = {
  column: "cell_type", label_a: "STIM", label_b: "CTRL", auto: true, n_a: 1000, n_b: 1000,
  groups: [group("Monocytes", 0.42, 0.18, true), group("T", 0.38, 0.62, true), group("B", 0.2, 0.2)],
  n_groups: 3, warn: true, message: "Monocytes is 42% of STIM cells but 18% of CTRL cells (+24 points): genes specific to Monocytes will look like STIM vs CTRL differences.",
  note: null, threshold_points: 10, min_cells: 10,
};

const balanced: DEComposition = {
  ...shifted, warn: false, message: null,
  groups: [group("CD14 Mono", 0.28, 0.334), group("B", 0.078, 0.066)],
};

describe("composition formatting", () => {
  it("writes shares and points plainly", () => {
    expect(formatShare(0.42)).toBe("42%");
    expect(formatShare(0.004)).toBe("0.4%");
    expect(formatShare(0)).toBe("0%");
    expect(formatPoints(24)).toBe("+24 points");
    expect(formatPoints(-5.4)).toBe("−5 points");
    expect(formatPoints(0.4)).toBe("+0.4 points");
    expect(shareLine(shifted.groups[0]!, "STIM", "CTRL")).toBe("42% of STIM vs 18% of CTRL (+24 points)");
  });

  it("lists the flagged groups and states the rule with its numbers", () => {
    expect(flaggedGroups(shifted).map((g) => g.group)).toEqual(["Monocytes", "T"]);
    expect(compositionRule(shifted)).toContain("more than 10 percentage points");
    expect(compositionRule(shifted)).toContain("at least 10 cells");
  });

  it("says what the largest difference was when nothing is flagged", () => {
    expect(balancedText(balanced)).toBe(
      "No cell_type group differs by more than 10 points between the sides; the largest is CD14 Mono, 28% of STIM vs 33% of CTRL (−5 points).",
    );
    expect(balancedText({ ...balanced, nested: true })).toBeNull();
    expect(balancedText({ ...balanced, column: null, groups: [] })).toBeNull();
  });

  it("gives the co-pilot the message only for a warning", () => {
    expect(compositionCautionForCopilot(shifted)).toBe(shifted.message);
    expect(compositionCautionForCopilot(balanced)).toBeNull();
    expect(compositionCautionForCopilot(undefined)).toBeNull();
  });
});
