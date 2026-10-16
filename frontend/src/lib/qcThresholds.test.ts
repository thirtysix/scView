import { describe, it, expect } from "vitest";
import { parseThreshold, pct, relayoutLineValue, roundTo } from "./qcThresholds";

describe("relayoutLineValue", () => {
  it("reads a dragged line from both ends, or the one end that moved", () => {
    expect(relayoutLineValue({ "shapes[1].x0": 210, "shapes[1].x1": 210 }, 1)).toBe(210);
    expect(relayoutLineValue({ "shapes[0].x0": 200, "shapes[0].x1": 220 }, 0)).toBe(210);
    expect(relayoutLineValue({ "shapes[0].x1": 5.5 }, 0)).toBe(5.5);
  });

  it("ignores other shapes, zooms and non-numbers", () => {
    expect(relayoutLineValue({ "shapes[1].x0": 3 }, 0)).toBeNull();
    expect(relayoutLineValue({ "xaxis.range[0]": 3 }, 0)).toBeNull();
    expect(relayoutLineValue({ "shapes[0].x0": "3" }, 0)).toBeNull();
  });
});

describe("roundTo", () => {
  it("rounds to the step without float noise and never below zero", () => {
    expect(roundTo(345.6, 1)).toBe(346);
    expect(roundTo(0.2349, 0.005)).toBe(0.235);
    expect(roundTo(12.34, 0.1)).toBe(12.3);
    expect(roundTo(-4, 1)).toBe(0);
  });
});

describe("parseThreshold", () => {
  it("treats empty, zero and junk as off", () => {
    expect(parseThreshold("")).toBeNull();
    expect(parseThreshold("0")).toBeNull();
    expect(parseThreshold("abc")).toBeNull();
    expect(parseThreshold("250")).toBe(250);
  });
});

describe("pct", () => {
  it("formats a share, keeping a digit for tiny ones", () => {
    expect(pct(0.0312)).toBe("3.1%");
    expect(pct(0.0004)).toBe("0.04%");
    expect(pct(0)).toBe("0.0%");
    expect(pct(null)).toBe("—");
  });
});
