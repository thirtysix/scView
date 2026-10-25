import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const { getStructure } = vi.hoisted(() => ({ getStructure: vi.fn() }));
vi.mock("@/api/structure", () => ({ getStructure }));

import type { DatasetStructure, MatrixSummary } from "@/api/structure";
import { structureText, valuesKind } from "@/lib/objectStructure";
import { ObjectInspector } from "./ObjectInspector";

const mat = (over: Partial<MatrixSummary> = {}): MatrixSummary => ({
  shape: [100, 20],
  dtype: "float32",
  format: "csr",
  on_disk: false,
  nnz: 400,
  density: 0.2,
  sample: { n_sampled: 400, min: 0, max: 5.2, integer: false, negative: false, n_nan: 0 },
  looks_like_counts: false,
  ...over,
});

const S: DatasetStructure = {
  n_obs: 100,
  n_vars: 20,
  mode: "memory",
  file_bytes: 2048,
  X: mat(),
  raw: null,
  layers: { counts: mat({ looks_like_counts: true, sample: { n_sampled: 400, min: 0, max: 30, integer: true, negative: false, n_nan: 0 } }) },
  sidecars: {},
  obs: {
    n_rows: 100,
    n_columns: 2,
    truncated: false,
    columns: [
      { name: "cluster", dtype: "category", kind: "categorical", n_unique: 3, n_missing: 0, all_missing: false, n_categories: 3, examples: ["a", "b", "c"] },
      { name: "empty", dtype: "float64", kind: "numeric", n_unique: 0, n_missing: 100, all_missing: true, min: null, max: null, examples: [] },
    ],
  },
  var: { n_rows: 20, n_columns: 0, truncated: false, columns: [] },
  obs_names: { n: 100, n_duplicated: 0, duplicated_examples: [], examples: ["c0", "c1"] },
  var_names: { n: 20, n_duplicated: 1, duplicated_examples: ["G1"], examples: ["G0", "G1"] },
  obsm: [{ key: "X_umap", shape: [100, 2], dtype: "float32", kind: "array" }],
  obsp: [],
  varm: [],
  varp: [],
  uns: [
    { key: "leiden", type: "dict", size: 1, more: 0, children: [{ key: "resolution", type: "float", value: 1 }] },
    { key: "big", type: "array", shape: [100000], dtype: "int64" },
  ],
  sampling: { values: 200000, rows: 500, seed: 0 },
  flags: [
    { level: "warn", where: "var_names", text: "1 duplicate gene names (e.g. G1): a gene looked up by name gets the first." },
    { level: "info", where: "obs", text: "1 obs column(s) are all missing: empty." },
  ],
};

beforeEach(() => {
  getStructure.mockReset().mockResolvedValue(S);
});

describe("ObjectInspector", () => {
  it("shows the flags at once and the details on Show", async () => {
    render(<ObjectInspector datasetId="ds" />);
    expect(await screen.findByTestId("object-flags")).toHaveTextContent("1 duplicate gene names");
    expect(screen.getByText(/100 x 20, in memory/)).toBeInTheDocument();
    expect(screen.queryByText("layers['counts']")).toBeNull();
    const show = screen.getByRole("button", { name: "Show" });
    fireEvent.click(show);
    expect(screen.getByRole("button", { name: "Hide" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("layers['counts']")).toBeInTheDocument();
    expect(screen.getByText("counts (integers)")).toBeInTheDocument();
    expect(screen.getByText("cluster")).toBeInTheDocument();
    expect(screen.getByText(/leiden: dict, 1 key/)).toBeInTheDocument();
    expect(screen.getByText(/n = 100 cells; 20 genes/)).toBeInTheDocument();
  });

  it("copies the summary as text", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    render(<ObjectInspector datasetId="ds" />);
    await screen.findByTestId("object-flags");
    fireEvent.click(screen.getByRole("button", { name: /Copy as text/ }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(writeText.mock.calls[0]![0]).toBe(structureText(S));
    expect(await screen.findByRole("status")).toHaveTextContent("Copied");
  });

  it("shows the server's reason", async () => {
    getStructure.mockRejectedValue(new Error("Dataset not found."));
    render(<ObjectInspector datasetId="ds" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Dataset not found.");
  });
});

describe("structureText", () => {
  it("lists the issues, matrices, tables and the uns tree", () => {
    const text = structureText(S);
    expect(text).toContain("AnnData object: 100 cells x 20 genes (in memory, file 2.0 KB)");
    expect(text).toContain("- 1 duplicate gene names");
    expect(text).toContain("X: 100 x 20 csr float32, 400 stored (20%); sample 0 to 5.2, non-integer (normalised?)");
    expect(text).toContain("layers['counts']: 100 x 20 csr float32");
    expect(text).toContain("cluster (category): 3 unique, e.g. a, b, c");
    expect(text).toContain("X_umap: 100 x 2 array float32");
    expect(text).toContain("    resolution: float = 1");
    expect(text).toContain("  big: array 100,000 int64");
  });

  it("names what the values look like", () => {
    expect(valuesKind(mat({ sample: { n_sampled: 1, min: -2, max: 3, integer: false, negative: true, n_nan: 0 } }))).toMatch(/scaled/);
    expect(valuesKind(mat({ sample: { n_sampled: 1, min: 0, max: 0, integer: true, negative: false, n_nan: 0 } }))).toMatch(/zero/);
  });
});
