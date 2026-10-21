import { describe, expect, it } from "vitest";
import type { QcStateGroup, QcStates } from "@/api/qcStates";
import {
  FLAGGED_COLOR,
  jitter,
  levelColumns,
  MAX_LEVEL_COLUMNS,
  shareRange,
  stateTraces,
  stripAxis,
  summaryText,
  validStateName,
} from "./qcStates";

const group = (name: string, over: Partial<QcStateGroup> = {}): QcStateGroup => ({
  name,
  n_cells: 500,
  n_fitted: 500,
  fitted: true,
  n_states: 1,
  delta_bic: -5,
  separation: 0.8,
  flagged_label: null,
  flagged_share: null,
  minor_weight: 0.3,
  components: [{ label: "main", n_cells: 500, weight: 1, mean_rp_share: 30, mean_pct_mt: 4 }],
  per_condition: {},
  reason: "One state",
  ...over,
});

const twoState = group("LNCaP", {
  n_states: 2,
  delta_bic: 412,
  separation: 2.6,
  flagged_label: "low-RP high-mito",
  flagged_share: 0.38,
  per_condition: { ctrl: { n: 250, share: 0.25 }, drug: { n: 250, share: 0.65 }, few: { n: 5, share: 1 } },
});

const result = (over: Partial<QcStates> = {}): QcStates => ({
  column: "cell_type",
  reason: "celltype",
  n_cells: 1100,
  has_mt: true,
  fit: "rp_mito",
  note: "",
  rules: { min_cells: 100, min_delta_bic: 10, min_separation: 2, min_weight: 0.05 },
  condition: { column: "treatment", kind: "condition", levels: ["ctrl", "drug"] },
  groups: [twoState, group("CAF"), group("tiny", { fitted: false, n_cells: 40, n_fitted: 40, delta_bic: null })],
  two_state_groups: ["LNCaP"],
  overall: { n_fitted: 1000, n_flagged: 190, share: 0.19 },
  scatter: {
    x: [30, 10, 28, 25],
    y: [4, 20, 5, 3],
    group: [0, 0, 0, 1],
    state: [0, 1, 0, 0],
    x_label: "RP share of non-mito counts (%)",
    y_label: "% mito",
    n_shown: 4,
    max_per_group: 2000,
  },
  ...over,
});

const grey = () => [100, 100, 100] as const;

describe("stateTraces", () => {
  it("2-D, all groups: a trace per group and state, the flagged state as crosses", () => {
    const traces = stateTraces(result(), ["CAF", "LNCaP"], grey, null);
    expect(traces.map((t) => t.name)).toEqual(["CAF", "LNCaP", "LNCaP: low-RP high-mito"]);
    const flagged = traces[2]!;
    expect(flagged.marker.symbol).toBe("x");
    expect(flagged.x).toEqual([10]);
    expect(flagged.y).toEqual([20]);
    expect(traces[1]!.legendgroup).toBe("LNCaP");
  });

  it("2-D, one group: main in slate and the flagged state in red", () => {
    const traces = stateTraces(result(), [], grey, "LNCaP");
    expect(traces.map((t) => t.name)).toEqual(["main", "low-RP high-mito"]);
    expect(traces[1]!.marker.color).toBe(FLAGGED_COLOR);
    expect(traces[0]!.x).toEqual([30, 28]);
  });

  it("1-D (no mito): RP share on y, groups along x in the column's order", () => {
    const r = result({ has_mt: false, fit: "rp_only", scatter: { ...result().scatter!, y: null, y_label: null } });
    const traces = stateTraces(r, ["CAF", "LNCaP"], grey, null);
    expect(traces.map((t) => t.name)).toEqual(["main", "flagged state"]);
    const main = traces[0]!;
    expect(main.y).toEqual([30, 28, 25]);
    // LNCaP is second in the order (x near 1), CAF first (x near 0).
    expect(Math.abs(main.x[0]! - 1)).toBeLessThan(0.5);
    expect(Math.abs(main.x[2]!)).toBeLessThan(0.5);
    expect(stripAxis(r, ["CAF", "LNCaP"]).ticktext).toEqual(["CAF", "LNCaP", "tiny"]);
  });

  it("returns nothing without a scatter", () => {
    expect(stateTraces(result({ scatter: undefined }), [], grey, null)).toEqual([]);
  });
});

describe("helpers", () => {
  it("jitter is deterministic and bounded", () => {
    const xs = Array.from({ length: 500 }, (_, i) => jitter(i));
    expect(xs).toEqual(Array.from({ length: 500 }, (_, i) => jitter(i)));
    expect(Math.max(...xs.map(Math.abs))).toBeLessThanOrEqual(0.32);
    expect(new Set(xs.map((x) => x.toFixed(3))).size).toBeGreaterThan(200);
  });

  it("shareRange spans the levels with enough cells", () => {
    expect(shareRange(twoState)).toBe("25% (ctrl) – 65% (drug)");
    expect(shareRange(group("x"))).toBeNull();
  });

  it("levelColumns switches to a range past the cap", () => {
    expect(levelColumns(result())).toEqual({ levels: ["ctrl", "drug"], asRange: false });
    const many = Array.from({ length: MAX_LEVEL_COLUMNS + 1 }, (_, i) => `d${i}`);
    const r = result({ condition: { column: "Donor", kind: "sample", levels: many } });
    expect(levelColumns(r).asRange).toBe(true);
    expect(levelColumns(result({ condition: null })).levels).toEqual([]);
  });

  it("summaryText names the two-state groups and the unfitted ones", () => {
    expect(summaryText(result())).toBe(
      "Two states in 1 of 2 fitted groups of cell_type (LNCaP 38% in the flagged state). 1 group under 100 cells not fitted.",
    );
    const none = result({ groups: [group("CAF")], two_state_groups: [] });
    expect(summaryText(none)).toBe("No group of cell_type shows a second state: one state in each of the 1 fitted groups.");
  });

  it("validStateName follows the backend's rule", () => {
    expect(validStateName("qc_state")).toBe(true);
    expect(validStateName("qc.state-2")).toBe(true);
    for (const bad of ["", "1x", "a/b", "scview_x", "a b"]) expect(validStateName(bad)).toBe(false);
  });
});
