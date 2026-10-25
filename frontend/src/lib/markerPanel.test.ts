import { describe, it, expect } from "vitest";
import { RESULT } from "./markerPanel.fixture";
import { confusionShares, curveTraces, hardestGroups, panelGenesText, panelText, parseGenes } from "./markerPanel";

describe("markerPanel helpers", () => {
  it("parses pasted genes on commas, spaces and new lines, once each", () => {
    expect(parseGenes("CD3E, CD19\nMS4A1;CD3E  ")).toEqual(["CD3E", "CD19", "MS4A1"]);
    expect(parseGenes("  ")).toEqual([]);
  });

  it("draws the selected curve and the baseline", () => {
    const t = curveTraces(RESULT.path, RESULT.baseline) as { name: string; y: number[] }[];
    expect(t.map((x) => x.name)).toEqual(["selected panel", "top markers in turn"]);
    expect(t[0]!.y).toEqual([0.6, 0.9]);
    expect(curveTraces(RESULT.path, [])).toHaveLength(1);
  });

  it("row-normalises the confusion counts", () => {
    const s = confusionShares([[3, 1], [0, 0]]);
    expect(s[0]).toEqual([0.75, 0.25]);
    expect(s[1]).toEqual([0, 0]);
  });

  it("writes the panel as text with the scheme and the baseline", () => {
    const text = panelText(RESULT);
    expect(text).toContain("Minimal marker panel for 'cell_type' (2 genes)");
    expect(text).toContain("stratified 3-fold CV");
    expect(text).toContain("Top 2 markers taken in turn from each group: 0.700");
    expect(text).toContain("GNLY\tNK");
    expect(panelGenesText(RESULT)).toBe("MS4A1\nGNLY");
  });

  it("names the hardest groups by recall", () => {
    expect(hardestGroups(RESULT, 2).map((h) => h.group)).toEqual(["T", "NK"]);
  });
});
