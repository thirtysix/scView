import { describe, it, expect } from "vitest";
import { mergeGeneLists, missingGenes, parseGeneList } from "./geneList";

describe("parseGeneList", () => {
  it("splits on commas, semicolons, spaces and newlines, keeping order", () => {
    expect(parseGeneList("CD3E, MS4A1;NKG7\n  LYZ\tCD14 ")).toEqual(["CD3E", "MS4A1", "NKG7", "LYZ", "CD14"]);
  });

  it("strips quotes, drops empties and repeats", () => {
    expect(parseGeneList(`"CD3E", 'CD3E',, MS4A1`)).toEqual(["CD3E", "MS4A1"]);
    expect(parseGeneList("  ")).toEqual([]);
  });

  it("keeps hyphenated and dotted names whole", () => {
    expect(parseGeneList("HLA-DRA MT-CO1 AC233755.1")).toEqual(["HLA-DRA", "MT-CO1", "AC233755.1"]);
  });
});

describe("mergeGeneLists", () => {
  it("appends new genes once and caps the list", () => {
    expect(mergeGeneLists(["A", "B"], ["B", "C"])).toEqual(["A", "B", "C"]);
    expect(mergeGeneLists(["A"], ["B", "C"], 2)).toEqual(["A", "B"]);
  });
});

describe("missingGenes", () => {
  it("lists requested genes the response does not have", () => {
    expect(missingGenes(["CD3E", "cd4", "MS4A1"], ["CD3E", "MS4A1"])).toEqual(["cd4"]);
  });
});
