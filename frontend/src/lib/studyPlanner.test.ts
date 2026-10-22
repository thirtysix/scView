import { describe, expect, it } from "vitest";
import type { CompositionPlan, CompositionPoint, DEPlan, DEPoint } from "@/api/planner";
import {
  compositionCurves,
  compositionSentence,
  deSentence,
  donorsNeeded,
  foldText,
  groupName,
  lostGenesNote,
  parseNumbers,
  shareText,
} from "./studyPlanner";

const DONORS = [3, 4, 6, 8, 12, 16];

/** Power by donors for one curve, as the grid order gives it. */
function pts(group: string, fc: number, cells: number, power: (number | null)[], pessimistic = false): CompositionPoint[] {
  return DONORS.map((donors, i) => ({
    group, fold_change: fc, donors, cells, power: power[i] ?? null, se: 0.03, power_raw: null, pessimistic,
  }));
}

function compPlan(points: CompositionPoint[], over: Partial<CompositionPlan> = {}): CompositionPlan {
  return {
    kind: "composition", group_key: "cluster", condition_key: "stim", sample_key: "Donor",
    sample_key_detected: true, conditions: ["CTRL", "STIM"], reference: "CTRL", source: "pilot",
    assumed_sd: null, assumed_sd_default: 0.35, assumed_sd_source: "Kang", pilot: {
      design: "paired", n_samples: { CTRL: 8, STIM: 8 }, n_donors: 8, df: 7,
      cells_per_sample: { median: 814, min: 279, max: 1622 }, dropped_samples: [],
    },
    design: "unpaired", can_pair: true, alpha: 0.05, target_power: 0.8, reps: 100,
    donors: DONORS, cells: [500, 2000, 5000], fold_changes: [2], middle_cells: 2000,
    types: [{
      group: "CD14 Mono", share: 0.35, share_tested: 0.3, n_cells: 4225, sd_within: 0.02, sd_donor: 0.48,
      sd_total: 0.49, sd_used: 0.49, sd_low: 0, sd_high: 0.16, sd_low_used: 0.3, sd_upper_used: 0.63, targeted: true,
    }],
    points, needed: [], assumptions: [], notes: [], elapsed_s: 30, ...over,
  };
}

describe("donorsNeeded", () => {
  it("interpolates between grid values and rounds up", () => {
    // 0.7 at 4 donors, 0.85 at 6: 80% at 4 + 2 * (0.1 / 0.15) = 5.33 -> 6
    const p = [{ donors: 3, power: 0.4 }, { donors: 4, power: 0.7 }, { donors: 6, power: 0.85 }, { donors: 8, power: 0.95 }];
    expect(donorsNeeded(p, 0.8)).toBe(6);
    expect(donorsNeeded(p, 0.4)).toBe(3);
    expect(donorsNeeded(p, 0.99)).toBeNull();
    expect(donorsNeeded([{ donors: 3, power: null }], 0.8)).toBeNull();
  });
});

describe("compositionSentence", () => {
  it("answers in donors per group and says when cells stop helping", () => {
    const plan = compPlan([
      ...pts("CD14 Mono", 2, 500, [0.3, 0.5, 0.7, 0.85, 0.95, 1]),
      ...pts("CD14 Mono", 2, 2000, [0.4, 0.6, 0.86, 0.95, 1, 1]),
      ...pts("CD14 Mono", 2, 5000, [0.42, 0.62, 0.87, 0.96, 1, 1]),
      ...pts("CD14 Mono", 2, 2000, [0.1, 0.2, 0.4, 0.6, 0.85, 0.95], true),
    ]);
    // 500 cells: 0.7 at 6, 0.85 at 8 -> 6 + 2 * (0.1 / 0.15) = 7.33 -> 8;
    // 2,000: 0.6 at 4, 0.86 at 6 -> 4 + 2 * (0.2 / 0.26) = 5.54 -> 6; 5,000: 6 too.
    expect(compositionSentence(plan, "CD14 Mono", 2, 0.8)).toBe(
      "To detect a 2-fold increase in CD14 Mono's share (35.0% of the cells in CTRL) with 80% power you need " +
        "about 6 donors per group at 2,000 cells per donor; adding cells beyond 2,000 per donor barely helps. " +
        "If the donors vary as much as this pilot allows (the upper end of its 95% interval), about 12 (at 2,000 cells).",
    );
  });

  it("says when cells barely matter, when the grid falls short and when a change is impossible", () => {
    const same = compPlan([
      ...pts("CD14 Mono", 2, 500, [0.9, 1, 1, 1, 1, 1]),
      ...pts("CD14 Mono", 2, 2000, [0.95, 1, 1, 1, 1, 1]),
      ...pts("CD14 Mono", 2, 5000, [0.97, 1, 1, 1, 1, 1]),
    ]);
    expect(compositionSentence(same, "CD14 Mono", 2, 0.8)).toBe(
      "Even 3 donors per group at 500 cells per donor give 80% power to detect a 2-fold increase in CD14 Mono's " +
        "share (35.0% of the cells in CTRL); the cells per donor (500 to 5,000) barely change this.",
    );
    const short = compPlan([
      ...pts("CD14 Mono", 2, 500, [0.05, 0.1, 0.15, 0.2, 0.3, 0.45]),
      ...pts("CD14 Mono", 2, 2000, [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]),
      ...pts("CD14 Mono", 2, 5000, [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]),
    ]);
    expect(compositionSentence(short, "CD14 Mono", 2, 0.8)).toMatch(
      /^Even 16 donors per group give at most 45% power to detect a 2-fold increase in CD14 Mono's share/,
    );
    const none = compPlan(pts("CD14 Mono", 3, 500, [null, null, null, null, null, null]), {
      cells: [500], fold_changes: [3],
    });
    expect(compositionSentence(none, "CD14 Mono", 3, 0.8)).toBe(
      "A 3-fold increase is not possible for CD14 Mono: it would take its share past 98% of the cells.",
    );
  });

  it("keeps the dashed curve apart from the cells curves", () => {
    const plan = compPlan([
      ...pts("CD14 Mono", 2, 500, [0.3, 0.5, 0.7, 0.85, 0.95, 1]),
      ...pts("CD14 Mono", 2, 2000, [0.1, 0.2, 0.4, 0.6, 0.85, 0.95], true),
    ], { cells: [500] });
    const curves = compositionCurves(plan, "CD14 Mono", 2);
    expect(curves.map((c) => [c.cells, c.pessimistic])).toEqual([[500, false], [2000, true]]);
    expect(curves[0]!.power).toEqual([0.3, 0.5, 0.7, 0.85, 0.95, 1]);
  });
});

function dePts(L: number, cells: number, power: number[], tested = 1): DEPoint[] {
  return DONORS.map((donors, i) => ({
    log2fc: L, donors, cells, power: power[i]!, se: 0.01, fdp: 0.05, tested_share: tested,
    power_by_expression: [0.1, 0.5, 0.9],
  }));
}

function dePlan(points: DEPoint[]): DEPlan {
  return {
    kind: "de", group_key: "cluster", group: "CD14 Mono", condition_key: "stim", sample_key: "Donor",
    sample_key_detected: true, conditions: ["CTRL", "STIM"], reference: "CTRL",
    pilot: {
      design: "paired", n_samples: { CTRL: 8, STIM: 8 }, n_donors: 8,
      cells_per_sample: { median: 227, min: 99, max: 638 }, dropped_samples: [], n_tested: 3454,
      n_simulated: 2000, counts_source: "layers['counts']", n_significant: 2318,
    },
    dispersion: { pilot_median: 0.027, bio_median: 0.019, cell_median: 1, donor_sd_median: 0.06 },
    expression_tertiles: [0.05, 0.2], design: "paired", can_pair: true, log2fc: [1], donors: DONORS,
    cells: [50, 200, 1000], de_share: 0.1, alpha: 0.05, target_power: 0.8, reps: 11, points,
    needed: [], assumptions: [], notes: [], elapsed_s: 50,
  };
}

describe("deSentence", () => {
  it("answers from the curves, and notes genes lost at few cells", () => {
    const plan = dePlan([
      ...dePts(1, 50, [0.12, 0.17, 0.27, 0.33, 0.34, 0.34], 0.35),
      ...dePts(1, 200, [0.37, 0.53, 0.71, 0.78, 0.82, 0.83]),
      ...dePts(1, 1000, [0.75, 0.87, 0.95, 0.97, 0.98, 0.99]),
    ]);
    // 1,000 cells: 0.75 at 3, 0.87 at 4 -> 3 + 1 * (0.05 / 0.12) = 3.42 -> 4;
    // 200 cells: 0.78 at 8, 0.82 at 12 -> 8 + 4 * 0.5 = 10 exactly (not 11 by rounding error)
    expect(deSentence(plan, 1, 0.8)).toBe(
      "Finding 80% of the genes changed by log2FC 1 (2-fold) in CD14 Mono (FDR < 0.05) takes about 4 donors per " +
        "group at 1,000 CD14 Mono cells per donor; more cells still help (10 at 200 CD14 Mono cells per donor).",
    );
    expect(lostGenesNote(plan, 1)).toMatch(/^With 50 CD14 Mono cells per donor only 35% of the changed genes/);
  });
});

describe("helpers", () => {
  it("formats effects and shares and parses number lists", () => {
    expect(groupName("8")).toBe("cluster 8");
    expect(groupName("CD14 Mono")).toBe("CD14 Mono");
    expect(foldText(2)).toBe("a 2-fold increase");
    expect(foldText(0.5)).toBe("a 2-fold decrease");
    expect(foldText(1.5)).toBe("a 1.5-fold increase");
    expect(shareText(0.35)).toBe("35.0%");
    expect(shareText(0.0057)).toBe("0.57%");
    expect(parseNumbers("16, 3 4;4 8")).toEqual([3, 4, 8, 16]);
    expect(parseNumbers("3, x")).toBeNull();
    expect(parseNumbers("2.5", { integer: true })).toBeNull();
    expect(parseNumbers("1", { min: 2 })).toBeNull();
    expect(parseNumbers("")).toBeNull();
  });
});
