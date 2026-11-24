import { describe, expect, it } from "vitest";
import {
  BY_ORDER,
  defaultColumns,
  describeReport,
  inCardWords,
  parseHeader,
  readFirstLine,
  savedLabelsText,
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
    expect(describeReport(r)).toEqual([
      "13,836 of 13,836 cells get a label, row by row in the dataset's order.",
      "No sample column to check the order by: the rows must follow the dataset's cells exactly.",
    ]);
    const checked = { ...r, order_check: { table_column: "donor", dataset_column: "ind", n_samples: 8, lined_up: true } };
    expect(describeReport(checked)[1]).toBe(
      "The rows line up with the dataset's 'ind' sample by sample (table column 'donor', 8 samples).",
    );
  });

  it("into an existing column: the rest keep their label, and how many change", () => {
    const r = { ...base, created: false, column: "mine", n_changed: 42 };
    expect(describeReport(r)).toEqual([
      "13,336 of 13,366 labelled rows name a cell of this dataset (99.8%).",
      "30 rows name no cell here (e.g. BOGUS_0, BOGUS_1).",
      "500 cells are not in the table and keep their label in mine (e.g. CTRL_A.1).",
      "42 of the 13,336 matched cells get a different label in mine.",
    ]);
  });
});

it("names the card's controls, not the API's options", () => {
  expect(inCardWords("13,336 of 13,366 rows would match: import with strip_suffix.")).toBe(
    '13,336 of 13,366 rows would match: tick "Ignore a -1 or .1 ending".',
  );
  expect(inCardWords("or lower min_match if the table")).toBe('or lower "Rows that must match" if the table');
});

describe("savedLabelsText", () => {
  const r = {
    column: "sample", created: true, match: "name", n_rows: 2887, n_blank: 0, n_cells: 2887, n_matched: 2887,
    n_extra: 0, n_duplicates: 0, n_unmatched_cells: 0, n_labels: 8,
    labels: Array.from({ length: 8 }, (_, i) => ({ label: `L${i}`, n_cells: 1 })),
  } as LabelImportReport;
  it("says Unassigned only when a cell is (QA-8)", () => {
    expect(savedLabelsText({ ...r, n_unassigned: 0 })).toBe("8 labels");
    expect(savedLabelsText(r)).toBe("8 labels"); // an older backend: no count
    expect(savedLabelsText({ ...r, n_labels: 9, n_unassigned: 1 })).toBe("8 labels; 1 cell is Unassigned");
    expect(savedLabelsText({ ...r, labels: r.labels.slice(0, 1), n_imported_labels: 25, n_unassigned: 1200 })).toBe(
      "25 labels; 1,200 cells are Unassigned",
    );
  });
});
