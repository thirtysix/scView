import { describe, it, expect } from "vitest";
import { formatBytes } from "./formatting";

describe("formatBytes", () => {
  it("uses decimal units with about three significant digits", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(950)).toBe("950 B");
    expect(formatBytes(12_345)).toBe("12.3 kB");
    expect(formatBytes(1_200_000_000)).toBe("1.20 GB");
    expect(formatBytes(345_000_000)).toBe("345 MB");
    expect(formatBytes(2.5e15)).toBe("2500 TB");
  });

  it("returns an empty string for nonsense input", () => {
    expect(formatBytes(-1)).toBe("");
    expect(formatBytes(Number.NaN)).toBe("");
  });
});
