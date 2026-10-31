import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({ getCoverage: vi.fn(), getFragments: vi.fn() }));
vi.mock("@/api/atacFragments", () => api);
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data, config }: { data: { name?: string }[]; config: { toImageButtonOptions?: { filename?: string } } }) => (
    <div data-testid="plot" data-file={config?.toImageButtonOptions?.filename}>{data.map((t) => t.name).join(";")}</div>
  ),
}));

import type { Coverage } from "@/api/atacFragments";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { asRequest, AtacCoverage } from "./AtacCoverage";

const COV: Coverage = {
  region: { chrom: "chr11", start: 1000, end: 3000 },
  bins: 2,
  bin_size: 1000,
  x: [1500, 2500],
  group_by: "cell_type",
  tracks: [{ name: "B", n_cells: 40, n_fragments_in_window: 3, total_fragments: 100, values: [1, 2] },
           { name: "T", n_cells: 60, n_fragments_in_window: 1, total_fragments: 100, values: [0, 1] }],
  ymax: 2,
  normalisation: "Mean fragment depth per bin / the group's fragments in cells x 1e6.",
  notes: [],
  n_fragments: 4,
  max_window: 2000000,
  peaks: [{ start: 1200, end: 1400 }],
  genes: [{ name: "MS4A1", id: "E", start: 1100, end: 2900, strand: "+", type: "protein_coding" }],
  gene: { name: "MS4A1", id: "E", start: 1100, end: 2900, strand: "+", type: "protein_coding" },
  genes_note: "Gene bodies from GENCODE, without exons.",
  seconds: 0.01,
};

beforeEach(() => {
  vi.clearAllMocks();
  useDatasetStore.setState({
    currentDataset: {
      id: "d",
      n_cells: 100,
      obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "T"] },
                    { name: "score", dtype: "float", n_unique: 90 }],
    } as unknown as DatasetInfo,
    currentDatasetId: "d",
  });
  useSettingsStore.setState({ colorBy: "cell_type" });
});

describe("AtacCoverage", () => {
  it("draws nothing until the files are attached", async () => {
    api.getFragments.mockResolvedValue({ attached: false });
    const { container } = render(<AtacCoverage datasetId="d" />);
    await waitFor(() => expect(api.getFragments).toHaveBeenCalled());
    expect(container.textContent).toBe("");
  });

  it("asks by gene, groups by the colour-by column and captions the scale", async () => {
    api.getFragments.mockResolvedValue({ attached: true, max_window: 2000000, gene_pad: 5000 });
    api.getCoverage.mockResolvedValue(COV);
    render(<AtacCoverage datasetId="d" />);
    const box = await screen.findByLabelText("Gene or region");
    expect(screen.getByText(/Up to 2 Mb/)).toBeTruthy();
    fireEvent.change(box, { target: { value: "MS4A1" } });
    await waitFor(() => expect((screen.getByLabelText("Group by") as HTMLSelectElement).value).toBe("cell_type"));
    fireEvent.click(screen.getByRole("button", { name: /Draw tracks/ }));
    const fig = await screen.findByTestId("atac-coverage");
    expect(api.getCoverage.mock.calls[0]?.[1]).toEqual({ gene: "MS4A1", groupBy: "cell_type" });
    expect(screen.getByTestId("plot").textContent).toBe("B;T");
    expect(fig.textContent).toContain("per bin");
    expect(fig.textContent).toContain("1 peaks marked");
    expect(fig.textContent).toContain("without exons");
  });

  it("shows the backend's sentence for a miss", async () => {
    api.getFragments.mockResolvedValue({ attached: true });
    api.getCoverage.mockRejectedValue(new Error("The fragments file has no chromosome 'chrQ'."));
    render(<AtacCoverage datasetId="d" />);
    const box = await screen.findByLabelText("Gene or region");
    fireEvent.change(box, { target: { value: "chrQ:1-100" } });
    fireEvent.click(screen.getByRole("button", { name: /Draw tracks/ }));
    expect(await screen.findByText(/no chromosome 'chrQ'/)).toBeTruthy();
    expect(api.getCoverage.mock.calls[0]?.[1]).toMatchObject({ region: "chrQ:1-100" });
  });

  it("an empty box is a sentence, not a request", async () => {
    api.getFragments.mockResolvedValue({ attached: true });
    render(<AtacCoverage datasetId="d" />);
    fireEvent.click(await screen.findByRole("button", { name: /Draw tracks/ }));
    expect(await screen.findByText(/Type a gene or a region/)).toBeTruthy();
    expect(api.getCoverage).not.toHaveBeenCalled();
  });

  it("tells a region from a gene", () => {
    expect(asRequest(" chr1:1,000-5,000 ")).toEqual({ region: "chr1:1,000-5,000" });
    expect(asRequest("CD3D")).toEqual({ gene: "CD3D" });
  });
});
