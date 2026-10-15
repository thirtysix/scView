import { describe, it, expect } from "vitest";
import { pickMarkerGenes, scalePerGene } from "./MarkerPlot";

describe("pickMarkerGenes", () => {
  it("takes the top up-regulated genes per group, each gene once, in group order", () => {
    const markers = {
      A: [
        { gene: "G1", logfoldchange: 3 },
        { gene: "DOWN", logfoldchange: -2 },
        { gene: "G2", logfoldchange: 1 },
        { gene: "G3", logfoldchange: 0.5 },
      ],
      B: [
        { gene: "G2", logfoldchange: 4 },
        { gene: "G4", logfoldchange: 2 },
      ],
    };
    expect(pickMarkerGenes(["A", "B"], markers, 2)).toEqual([
      { gene: "G1", group: "A" },
      { gene: "G2", group: "A" },
      { gene: "G4", group: "B" },
    ]);
  });
});

describe("scalePerGene", () => {
  it("scales each gene to 0-1 across groups and maps a flat gene to 0", () => {
    expect(scalePerGene([[1, 5], [3, 5]])).toEqual([[0, 0], [1, 0]]);
  });
});
