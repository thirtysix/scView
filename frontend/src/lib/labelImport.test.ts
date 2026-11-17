import { describe, expect, it } from "vitest";
import {
  BY_ORDER,
  defaultColumns,
  describeReport,
  inCardWords,
  parseHeader,
  readFirstLine,
  separatorOf,
  suggestImportName,
  type LabelImportReport,
} from "@/lib/labelImport";

describe("label table header", () => {
  it("splits as the backend does: tab, comma, semicolon, else by extension", () => {
    expect(separatorOf("a\tb,c", "x.csv")).toBe("\t");
    expect(separatorOf("a,b", "x.tsv")).toBe(",");
    expect(separatorOf("barcode", "x.csv.gz")).toBe(",");
    expect(separatorOf("barcode", "x.txt")).toBe("\t");
  });

  it("names columns as pandas does, R's empty first name included", () => {
    expect(parseHeader('"","sample","annotation"\r', "geo.csv").columns).toEqual(["Unnamed: 0", "sample", "annotation"]);
    expect(parseHeader("﻿cell\tlabel", "t.tsv")).toEqual({ columns: ["cell", "label"], sep: "\t" });
  });

  it("reads the first line of a plain or gzipped file", async () => {
    const text = "cell_id,annotation\nAAAC-1,M2\n";
    expect(await readFirstLine(new Blob([text]))).toBe("cell_id,annotation");
    const gz = new Blob([await new Response(new Response(text).body!.pipeThrough(new CompressionStream("gzip"))).arrayBuffer()]);
    expect(await readFirstLine(gz)).toBe("cell_id,annotation");
  });

  it("picks the first column for names and the next for labels; one column means by order", () => {
    expect(defaultColumns(["cell", "sample", "type"])).toEqual({ barcode: "cell", label: "sample" });
    expect(defaultColumns(["type"])).toEqual({ barcode: BY_ORDER, label: "type" });
  });

  it("suggests the label column's name when it is free", () => {
    expect(suggestImportName("annotation", ["leiden"])).toBe("annotation");
    expect(suggestImportName("cell type", ["cell type"])).toBe("imported_labels");
    expect(suggestImportName("scview_x", ["imported_labels"])).toBe("imported_labels_2");
    expect(suggestImportName("Unnamed: 0", [])).toBe("Unnamed_ 0");
  });
});

describe("match report", () => {
  const base: LabelImportReport = {
    column: "author", created: true, match: "name", n_rows: 13366, n_blank: 0, n_cells: 13836, n_matched: 13336,
    n_extra: 30, n_duplicates: 0, share_matched: 0.9978, extra_examples: ["BOGUS_0", "BOGUS_1"],
    n_unmatched_cells: 500, unmatched_examples: ["CTRL_A.1"], n_labels: 14, labels: [],
  };

  it("says what matched, what names no cell and what is left out", () => {
    expect(describeReport(base)).toEqual([
      "13,336 of 13,366 labelled rows name a cell of this dataset (99.8%).",
      "30 rows name no cell here (e.g. BOGUS_0, BOGUS_1).",
      "500 cells are not in the table and stay Unassigned (e.g. CTRL_A.1).",
    ]);
  });

  it("by order counts cells", () => {
    const r = { ...base, match: "order" as const, n_rows: 13836, n_matched: 13836, n_extra: 0, n_unmatched_cells: 0 };
    expect(describeReport(r)).toEqual(["13,836 of 13,836 cells get a label, row by row in the dataset's order."]);
  });
});

it("names the card's controls, not the API's options", () => {
  expect(inCardWords("13,336 of 13,366 rows would match: import with strip_suffix.")).toBe(
    '13,336 of 13,366 rows would match: tick "Ignore a -1 or .1 ending".',
  );
  expect(inCardWords("or lower min_match if the table")).toBe('or lower "Rows that must match" if the table');
});
