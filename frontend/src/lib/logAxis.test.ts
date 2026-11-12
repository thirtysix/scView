import { describe, it, expect } from "vitest";
import { MIN_TICK_PX, pseudoLog, pseudoLogTicks, summary, tickLabel } from "./logAxis";

describe("pseudoLog", () => {
  it("is log10(1 + v), 0 at 0, and symmetric for negatives", () => {
    expect(pseudoLog(0)).toBe(0);
    expect(pseudoLog(9)).toBeCloseTo(1);
    expect(pseudoLog(999)).toBeCloseTo(3);
    expect(pseudoLog(-9)).toBeCloseTo(-1);
  });
});

describe("tickLabel", () => {
  it("writes ticks short", () => {
    expect(tickLabel(0)).toBe("0");
    expect(tickLabel(0.05)).toBe("0.05");
    expect(tickLabel(20)).toBe("20");
    expect(tickLabel(2500)).toBe("2.5k");
    expect(tickLabel(10000)).toBe("10k");
    expect(tickLabel(1e6)).toBe("1M");
    expect(tickLabel(-5)).toBe("-5");
  });
});

describe("pseudoLogTicks", () => {
  const labels = (min: number, max: number) => pseudoLogTicks(min, max).ticktext;

  it("labels a count range in powers of ten, placed on the pseudo-log scale", () => {
    const t = pseudoLogTicks(0, 50_000);
    // "1" would sit 6% of the axis above "0"
    expect(t.ticktext).toEqual(["0", "10", "100", "1k", "10k"]);
    expect(t.tickvals[2]).toBeCloseTo(pseudoLog(100));
    expect(labels(0, 1000)).toEqual(["0", "1", "10", "100", "1k"]);
  });

  it("uses 1-2-5 steps over a short range, dropping ticks that bunch up near 0", () => {
    expect(labels(0, 6)).toEqual(["0", "0.5", "1", "2", "5"]); // log-normalised expression: no 0.1 / 0.2 pile at 0
    expect(labels(0, 100)).toEqual(["0", "1", "5", "10", "20", "50", "100"]);
    expect(labels(0, 1)).toEqual(["0", "0.1", "0.2", "0.5", "1"]); // a score in 0-1
  });

  it("mirrors the ticks for negative values", () => {
    expect(labels(-0.5, 1.5)).toEqual(["-0.5", "-0.2", "0", "0.2", "0.5", "1"]);
  });

  /** The smallest gap between neighbouring ticks, as a share of the axis. */
  const minShare = (min: number, max: number, px?: number) => {
    const v = pseudoLogTicks(min, max, px).tickvals;
    const span = pseudoLog(max) - pseudoLog(min);
    return Math.min(...v.slice(1).map((x, i) => (x - v[i]!) / span));
  };

  it("keeps every pair of ticks a tenth of the axis apart, and 0 and the outermost ticks", () => {
    const ranges: [number, number][] = [[0, 6], [0, 3], [0, 12], [0, 100], [0, 1], [0, 0.3], [-0.5, 1.5], [-3, 3], [0, 50_000]];
    for (const [min, max] of ranges) {
      expect(minShare(min, max), `${min}..${max}`).toBeGreaterThanOrEqual(0.1 - 1e-9);
      const t = labels(min, max);
      if (min <= 0 && max >= 0) expect(t, `${min}..${max}`).toContain("0");
    }
    expect(labels(0, 6).at(-1)).toBe("5");
    expect(labels(0, 50_000).at(-1)).toBe("10k");
    expect(labels(-3, 3)[0]).toBe("-2");
  });

  it("keeps the outermost tick when the one before it is too close (it drops that one instead)", () => {
    // 0..2.2: 1 and 2 both fit; on a 0..1.3 axis 1 is the outermost and 0.5 stays clear of it
    expect(labels(0, 1.3).at(-1)).toBe("1");
    expect(minShare(0, 1.3)).toBeGreaterThanOrEqual(0.1 - 1e-9);
  });

  it("keeps MIN_TICK_PX between ticks on a short axis", () => {
    const px = 120; // a short violin
    const v = pseudoLogTicks(0, 6, px).tickvals;
    const span = pseudoLog(6);
    const gaps = v.slice(1).map((x, i) => ((x - v[i]!) / span) * px);
    expect(Math.min(...gaps)).toBeGreaterThanOrEqual(MIN_TICK_PX);
    expect(pseudoLogTicks(0, 6, px).ticktext[0]).toBe("0");
    expect(pseudoLogTicks(0, 6, px).ticktext.at(-1)).toBe("5");
    // a tall axis is held to the share alone
    expect(pseudoLogTicks(0, 6, 1000).ticktext).toEqual(labels(0, 6));
  });

  it("copes with a single value and with no data", () => {
    expect(labels(0, 0)).toEqual(["0"]);
    expect(pseudoLogTicks(Infinity, -Infinity)).toEqual({ tickvals: [], ticktext: [] });
  });
});

describe("summary", () => {
  it("gives the range and the interpolated quartiles, skipping non-finite values", () => {
    expect(summary([4, 1, 3, 2, NaN, 5])).toEqual({ min: 1, q1: 2, median: 3, q3: 4, max: 5 });
    expect(summary([0, 10])).toEqual({ min: 0, q1: 2.5, median: 5, q3: 7.5, max: 10 });
    expect(summary([])).toBeNull();
  });
});
