import { describe, it, expect } from "vitest";
import { buildCsv } from "./csv";

describe("buildCsv", () => {
  it("joins headers and rows with newlines", () => {
    expect(buildCsv(["a", "b"], [[1, 2], [3, 4]])).toBe("a,b\n1,2\n3,4");
  });

  it("quotes and escapes cells containing commas, quotes, or newlines", () => {
    // The bug this fixed: enrichment terms / gene-set names with commas.
    const csv = buildCsv(
      ["term", "note"],
      [["Regulation of cell cycle, mitotic", 'has "quotes"'], ["line\nbreak", ""]],
    );
    expect(csv).toBe(
      'term,note\n"Regulation of cell cycle, mitotic","has ""quotes"""\n"line\nbreak",',
    );
  });

  it("renders null/undefined as empty cells", () => {
    expect(buildCsv(["a", "b", "c"], [[null, undefined, 0]])).toBe("a,b,c\n,,0");
  });
});
