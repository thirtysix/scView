import { describe, it, expect } from "vitest";
import {
  dedupeGenes,
  exportGeneLists,
  genePresence,
  genesByUpper,
  MAX_LIST_GENES,
  parseGeneListFile,
  sameGenes,
  uniqueListName,
} from "./geneLists";

describe("dedupeGenes", () => {
  it("keeps the first spelling and the order, dropping repeats ignoring case", () => {
    expect(dedupeGenes([" Cd3e", "CD3E", "Ms4a1", "cd3e", "", "Lyz2"])).toEqual(["Cd3e", "Ms4a1", "Lyz2"]);
  });

  it("drops names with spaces and caps the list", () => {
    expect(dedupeGenes(["CD3E CD3D", "A"])).toEqual(["A"]);
    const many = Array.from({ length: MAX_LIST_GENES + 10 }, (_, i) => `G${i}`);
    expect(dedupeGenes(many)).toHaveLength(MAX_LIST_GENES);
  });
});

describe("uniqueListName", () => {
  it("adds a number when the name is taken, ignoring case", () => {
    expect(uniqueListName("T cells", ["Other"])).toBe("T cells");
    expect(uniqueListName("T cells", ["t cells", "T cells (2)"])).toBe("T cells (3)");
    expect(uniqueListName("  \n ", [])).toBe("Gene list");
  });
});

describe("sameGenes", () => {
  it("compares order and names ignoring case", () => {
    expect(sameGenes(["CD3E", "MS4A1"], ["cd3e", "Ms4a1"])).toBe(true);
    expect(sameGenes(["CD3E", "MS4A1"], ["MS4A1", "CD3E"])).toBe(false);
  });
});

describe("parseGeneListFile", () => {
  it("makes one list per GMT set, dropping Enrichr weights and empty sets", () => {
    const text = "# comment\nT_CELL\tdesc\tCD3E\tCD3D,1.0\nEMPTY\tdesc\nB_CELL\t\tMS4A1\n";
    expect(parseGeneListFile("lib.gmt", text)).toEqual([
      { name: "T_CELL", genes: ["CD3E", "CD3D"], source: "file: lib.gmt" },
      { name: "B_CELL", genes: ["MS4A1"], source: "file: lib.gmt" },
    ]);
  });

  it("reads a gene_set,gene CSV as one list per set, and quoted names with commas", () => {
    const text = 'gene_set,gene\n"T, naive",CCR7\n"T, naive",SELL\nB,MS4A1\n';
    expect(parseGeneListFile("sets.csv", text).map((l) => [l.name, l.genes])).toEqual([
      ["T, naive", ["CCR7", "SELL"]],
      ["B", ["MS4A1"]],
    ]);
  });

  it("reads any other table's first column as one list named after the file", () => {
    const text = "gene,logfoldchange\nCD14,3.2\nLYZ,2.9\n";
    expect(parseGeneListFile("C:\\data\\mono_markers.csv", text)).toEqual([
      { name: "mono_markers", genes: ["CD14", "LYZ"], source: "file: mono_markers.csv" },
    ]);
  });

  it("reads a .txt as pasted names, or as GMT when it has tab-separated sets", () => {
    expect(parseGeneListFile("mine.txt", "\uFEFFCD3E\nCD3D, CD3G\n")[0]!.genes).toEqual(["CD3E", "CD3D", "CD3G"]);
    expect(parseGeneListFile("lib.txt", "S1\tna\tA\tB\n").map((l) => l.name)).toEqual(["S1"]);
  });

  it("throws when the file has no genes", () => {
    expect(() => parseGeneListFile("empty.gmt", "S\tdesc\n")).toThrow(/no genes/);
  });
});

describe("exportGeneLists", () => {
  const lists = [
    { name: "T, naive", genes: ["CCR7", "SELL"] },
    { name: "B", genes: ["MS4A1"], source: "Markers: B" },
  ];

  it("writes txt as one gene per line", () => {
    expect(exportGeneLists(lists, "txt")).toBe("CCR7\nSELL\n");
  });

  it("csv and gmt read back in as the same lists", () => {
    for (const format of ["csv", "gmt"] as const) {
      const back = parseGeneListFile(`out.${format}`, exportGeneLists(lists, format));
      expect(back.map((l) => [l.name, l.genes])).toEqual(lists.map((l) => [l.name, l.genes]));
    }
  });

  it("gmt puts the source in the description column", () => {
    expect(exportGeneLists(lists, "gmt").split("\n")[1]).toBe("B\tMarkers: B\tMS4A1");
  });
});

describe("genePresence", () => {
  it("matches names exactly, and says which missing genes exist in another case", () => {
    const mouse = ["Cd3e", "Ms4a1", "Lyz2"];
    const p = genePresence(["Cd3e", "MS4A1", "FOO"], new Set(mouse), genesByUpper(mouse));
    expect(p.present).toEqual(["Cd3e"]);
    expect(p.missing).toEqual(["MS4A1", "FOO"]);
    expect([...p.otherCase]).toEqual([["MS4A1", "Ms4a1"]]);
  });
});
