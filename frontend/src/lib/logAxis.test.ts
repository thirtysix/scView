import { describe, it, expect } from "vitest";
import { pseudoLog, pseudoLogTicks, summary, tickLabel } from "./logAxis";

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
    expect(t.ticktext).toEqual(["0", "1", "10", "100", "1k", "10k"]);
    expect(t.tickvals[3]).toBeCloseTo(pseudoLog(100));
  });

  it("uses 1-2-5 steps over a short range, dropping ticks that bunch up near 0", () => {
    expect(labels(0, 6)).toEqual(["0", "0.2", "0.5", "1", "2", "5"]); // log-normalised expression
    expect(labels(0, 100)).toEqual(["0", "1", "2", "5", "10", "20", "50", "100"]);
    expect(labels(0, 1)).toEqual(["0", "0.1", "0.2", "0.5", "1"]); // a score in 0-1
  });

  it("mirrors the ticks for negative values", () => {
    expect(labels(-0.5, 1.5)).toEqual(["-0.5", "-0.2", "-0.1", "0", "0.1", "0.2", "0.5", "1"]);
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
