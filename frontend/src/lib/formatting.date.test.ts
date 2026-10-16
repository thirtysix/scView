import { describe, it, expect } from "vitest";
import { formatDate } from "./formatting";

describe("formatDate", () => {
  it("formats an ISO timestamp as a short date", () => {
    expect(formatDate("2026-10-16T12:00:00+00:00", "en-GB")).toBe("16 Oct 2026");
  });

  it("returns an empty string for a missing or unparseable date", () => {
    expect(formatDate(undefined)).toBe("");
    expect(formatDate("")).toBe("");
    expect(formatDate("not a date")).toBe("");
  });
});
