import { describe, expect, it } from "vitest";
import {
  addSample,
  fileAt,
  formatRate,
  formatSize,
  formatTimeLeft,
  percentSent,
  rateOf,
  RATE_AFTER_MS,
  RATE_WINDOW_MS,
  secondsLeft,
  sentOfTotal,
  type Sample,
} from "./uploadProgress";

const MB = 1e6;

/** Samples every 500 ms at `perSecond` bytes per second, from t = 0. */
function steady(perSecond: number, ms: number): Sample[] {
  let s: Sample[] = [];
  for (let t = 0; t <= ms; t += 500) s = addSample(s, { t, loaded: (perSecond * t) / 1000 });
  return s;
}

describe("percentSent", () => {
  it("floors, and says 100 only once every byte is sent", () => {
    expect(percentSent(0, 331 * MB)).toBe(0);
    expect(percentSent(152 * MB, 331 * MB)).toBe(45);
    expect(percentSent(331 * MB - 1, 331 * MB)).toBe(99);
    expect(percentSent(331 * MB, 331 * MB)).toBe(100);
    expect(percentSent(5, 0)).toBe(0);
  });
});

describe("fileAt", () => {
  it("names the file being sent from the bytes gone", () => {
    const sizes = [10, 0, 30, 5];
    expect(fileAt(sizes, 0)).toBe(0);
    expect(fileAt(sizes, 9)).toBe(0);
    expect(fileAt(sizes, 10)).toBe(2); // the empty file takes no time
    expect(fileAt(sizes, 44)).toBe(3);
    expect(fileAt(sizes, 45)).toBe(3);
  });
});

describe("speed and time left", () => {
  it("gives no speed before a few seconds of data", () => {
    expect(rateOf(steady(2 * MB, RATE_AFTER_MS - 500))).toBeNull();
    expect(rateOf(steady(2 * MB, RATE_AFTER_MS))).toBeCloseTo(2 * MB, -3);
  });

  it("follows the last seconds, so a slower link shows within the window", () => {
    let s = steady(4 * MB, 20_000);
    const at = 80 * MB;
    for (let t = 20_500; t <= 20_000 + RATE_WINDOW_MS; t += 500) s = addSample(s, { t, loaded: at + (1 * MB * (t - 20_000)) / 1000 });
    expect(rateOf(s)).toBeCloseTo(1 * MB, -3);
  });

  it("keeps few samples, but always the first", () => {
    const s = steady(1 * MB, 60_000);
    expect(s.length).toBeLessThan(25);
    expect(s[0]!.t).toBe(0);
  });

  it("shows a stall as a falling speed", () => {
    let s = steady(4 * MB, 10_000);
    for (let t = 10_500; t <= 15_000; t += 500) s = addSample(s, { t, loaded: 40 * MB });
    expect(rateOf(s)!).toBeCloseTo(1.5 * MB, -3); // 12 MB in the last 8 s
  });

  it("time left = what is left over the speed", () => {
    expect(secondsLeft(100 * MB, 331 * MB, 2 * MB)).toBeCloseTo(115.5);
    expect(secondsLeft(100 * MB, 331 * MB, null)).toBeNull();
    expect(secondsLeft(100 * MB, 331 * MB, 0)).toBeNull();
  });
});

describe("formatting", () => {
  it("rounds time left up, coarser as it grows", () => {
    expect(formatTimeLeft(3)).toBe("a few seconds");
    expect(formatTimeLeft(21)).toBe("about 25 s");
    expect(formatTimeLeft(56)).toBe("about 1 min");
    expect(formatTimeLeft(115.5)).toBe("about 2 min");
    expect(formatTimeLeft(3599)).toBe("about 1 h");
    expect(formatTimeLeft(3600)).toBe("about 1 h");
    expect(formatTimeLeft(4700)).toBe("about 1 h 20 min");
  });

  it("writes sizes in the total's unit", () => {
    expect(sentOfTotal(152.4 * MB, 331 * MB)).toBe("152 of 331 MB");
    expect(sentOfTotal(3.25 * MB, 12.5 * MB)).toBe("3.3 of 12.5 MB");
    expect(sentOfTotal(0.4e9, 2.4e9)).toBe("0.40 of 2.40 GB");
    expect(sentOfTotal(1200, 4000)).toBe("1 of 4 kB");
    expect(sentOfTotal(500 * MB, 331 * MB)).toBe("331 of 331 MB");
    expect(formatSize(331 * MB)).toBe("331 MB");
  });

  it("writes the speed", () => {
    expect(formatRate(3.24 * MB)).toBe("3.2 MB/s");
    expect(formatRate(450_400)).toBe("450 kB/s");
  });
});
