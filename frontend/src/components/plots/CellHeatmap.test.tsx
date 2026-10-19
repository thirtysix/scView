import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { RecordBatch, Schema, Table, tableFromArrays, tableToIPC } from "apache-arrow";

const apiFetchBinary = vi.fn();
vi.mock("@/api/client", () => ({ apiFetchBinary: (...a: unknown[]) => apiFetchBinary(...a) }));
// Plotly needs a real browser; the test checks what is asked for and drawn.
const plotProps = vi.fn();
vi.mock("@/components/plots/Plot", () => ({
  default: (p: unknown) => {
    plotProps(p);
    return <div data-testid="plot" />;
  },
}));

import { CellHeatmap } from "./CellHeatmap";
import { useDatasetStore } from "@/stores/datasetStore";
import type { DatasetInfo } from "@/api/types";

function response(perGroup: number): ArrayBuffer {
  const meta = {
    groupby: "cell_type", layer: "raw", genes: ["CD3E", "LYZ"], missing_genes: [],
    groups: ["T", "Mono"], n_sampled: [2, 1], n_total: [50, 1], n_cells: 60, scale: "zscore",
    value_range: [-2.5, 2.5], seed: 0, per_group: perGroup, max_cells: 5000,
  };
  const t = tableFromArrays({
    cell: Int32Array.from([4, 8, 2]),
    group: Int32Array.from([0, 0, 1]),
    CD3E: Float32Array.from([1, 1, -2]),
    LYZ: Float32Array.from([-1, -1, 2.5]),
  });
  const schema = new Schema(t.schema.fields, new Map([["scview", JSON.stringify(meta)]]));
  const bytes = tableToIPC(new Table(schema, t.batches.map((b) => new RecordBatch(schema, b.data))), "stream");
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

const MARKERS = {
  T: [{ gene: "CD3E", logfoldchange: 2 }, { gene: "IL7R", logfoldchange: 1 }],
  Mono: [{ gene: "LYZ", logfoldchange: 3 }, { gene: "CD3E", logfoldchange: 1 }],
};
const DATASET = {
  id: "ds1", name: "ds1", obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["T", "Mono"] }],
  expression_layers: [{ key: "raw", label: "Log-normalized (raw)", n_genes: 10 }],
} as unknown as DatasetInfo;

const url = (i = -1) => new URL(`http://x${String(apiFetchBinary.mock.calls.at(i)![0])}`);

describe("CellHeatmap", () => {
  beforeEach(() => {
    apiFetchBinary.mockReset();
    plotProps.mockReset();
    useDatasetStore.setState({ currentDatasetId: "ds1", currentDataset: DATASET });
    apiFetchBinary.mockImplementation((u: string) =>
      Promise.resolve(response(Number(new URL(`http://x${u}`).searchParams.get("per_group")))),
    );
  });

  it("asks for the top markers in 200 cells per group, and says what it drew", async () => {
    render(<CellHeatmap datasetId="ds1" groupby="cell_type" groups={["T", "Mono"]} markers={MARKERS} perGroup={1} />);
    await screen.findByTestId("plot");
    const u = url();
    expect(u.pathname).toBe("/datasets/ds1/expression/heatmap-cells");
    expect(u.searchParams.get("genes")).toBe("CD3E,LYZ"); // one top marker each, CD3E once
    expect(u.searchParams.get("groupby")).toBe("cell_type");
    expect(u.searchParams.get("per_group")).toBe("200");
    expect(u.searchParams.get("scale")).toBe("zscore");
    expect(u.searchParams.get("seed")).toBe("0");
    expect(screen.getByTestId("cell-heatmap-caption")).toHaveTextContent(
      "200 cells per group (sampled), 3 of 60 cells, seed 0.",
    );
    expect(screen.getByTestId("cell-heatmap-caption")).toHaveTextContent("clipped to ±2.5");
    const p = plotProps.mock.calls.at(-1)![0] as { data: { y: unknown[]; colorscale: [number, string][] }[] };
    expect(p.data[0]!.y).toEqual(["CD3E", "LYZ"]);
    // the strip takes the groups' category colours, one band each
    expect(p.data[1]!.colorscale.length).toBe(4);
  });

  it("refetches for another number of cells per group and for the expression scale", async () => {
    render(<CellHeatmap datasetId="ds1" groupby="cell_type" groups={["T", "Mono"]} markers={MARKERS} perGroup={1} />);
    await screen.findByTestId("plot");
    fireEvent.change(screen.getByLabelText("Cells per group:"), { target: { value: "50" } });
    await waitFor(() => expect(url().searchParams.get("per_group")).toBe("50"));
    fireEvent.change(screen.getByLabelText("Colour:"), { target: { value: "none" } });
    await waitFor(() => expect(url().searchParams.get("scale")).toBe("none"));
  });

  it("sends at most 100 of the user's genes and says how many were left out", async () => {
    const genes = Array.from({ length: 120 }, (_, i) => `G${i}`);
    render(<CellHeatmap datasetId="ds1" groupby="cell_type" groups={[]} markers={{}} perGroup={5} genes={genes} />);
    await screen.findByTestId("plot");
    expect(url().searchParams.get("genes")!.split(",")).toHaveLength(100);
    expect(screen.getByText(/The first 100 genes are drawn per cell; 20 more/)).toBeTruthy();
  });

  it("says plainly when none of the genes are in the layer", async () => {
    apiFetchBinary.mockRejectedValue(new Error('API error 400: {"detail":"None of the genes are in layer \'raw\': NOPE."}'));
    render(<CellHeatmap datasetId="ds1" groupby="cell_type" groups={[]} markers={{}} perGroup={5} genes={["NOPE"]} />);
    expect(await screen.findByText(/None of the genes are in layer 'raw': NOPE/)).toHaveTextContent("case-sensitive");
  });
});
