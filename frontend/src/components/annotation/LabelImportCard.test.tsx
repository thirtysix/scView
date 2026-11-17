import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const { importLabelFile, refresh } = vi.hoisted(() => ({ importLabelFile: vi.fn(), refresh: vi.fn() }));
vi.mock("@/api/labels", async (orig) => ({ ...(await orig<typeof import("@/api/labels")>()), importLabelFile }));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));

import { LabelImportCard } from "./LabelImportCard";
import { ApiDetailError } from "@/api/labels";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const REPORT = {
  column: "author_type", created: true, match: "name", n_rows: 13366, n_blank: 0, n_cells: 13836, n_matched: 13336,
  n_extra: 30, n_duplicates: 0, share_matched: 0.9978, extra_examples: ["BOGUS_0"], n_unmatched_cells: 500,
  unmatched_examples: ["CTRL_A.1"], n_labels: 14,
  labels: [{ label: "CD14 Mono", n_cells: 4081 }, { label: "B", n_cells: 962 }], dry_run: true,
};

function pick(text: string, name = "kang_author.csv") {
  fireEvent.change(screen.getByLabelText("Label table file"), { target: { files: [new File([text], name)] } });
}

beforeEach(() => {
  vi.clearAllMocks();
  useDatasetStore.setState({
    currentDataset: { id: "ds", name: "Kang", obs_columns: [{ name: "cluster" }, { name: "stim" }] } as never,
  });
  refresh.mockResolvedValue({});
});

describe("LabelImportCard", () => {
  it("reads the header, checks the match, then imports and offers to colour by it", async () => {
    importLabelFile.mockResolvedValueOnce(REPORT).mockResolvedValueOnce({ ...REPORT, dry_run: undefined });
    render(<LabelImportCard datasetId="ds" />);
    pick("cell_barcode,donor,author_type\nCTRL_A.1,d1,B\n");
    const labels = (await screen.findByLabelText("Label column")) as HTMLSelectElement;
    expect((screen.getByLabelText("Cell name column") as HTMLSelectElement).value).toBe("cell_barcode");
    expect(labels.value).toBe("donor");
    fireEvent.change(labels, { target: { value: "author_type" } });
    expect((screen.getByLabelText("New column name") as HTMLInputElement).value).toBe("author_type");

    const importButton = screen.getByRole("button", { name: /Import into author_type/ });
    expect(importButton).toBeDisabled(); // check first
    fireEvent.click(screen.getByRole("button", { name: "Check the match" }));
    const report = await screen.findByTestId("label-import-report");
    expect(report.textContent).toContain("13,336 of 13,366 labelled rows name a cell of this dataset (99.8%).");
    expect(report.textContent).toContain("30 rows name no cell here (e.g. BOGUS_0).");
    expect(report.textContent).toContain("500 cells are not in the table and stay Unassigned");
    expect(report.textContent).toContain("4,081");
    expect(importLabelFile.mock.calls[0]![2]).toEqual({
      column: "author_type", barcodeColumn: "cell_barcode", labelColumn: "author_type", stripSuffix: false,
      minMatch: 0.9, dryRun: true,
    });

    fireEvent.click(importButton);
    await screen.findByText(/Saved 13,336 cells into/);
    expect(importLabelFile.mock.calls[1]![2].dryRun).toBe(false);
    expect(refresh).toHaveBeenCalledWith("ds", "author_type");
    fireEvent.click(screen.getByRole("button", { name: "Colour by it" }));
    expect(useSettingsStore.getState().colorBy).toBe("author_type");
  });

  it("shows the server's refusal and clears a check when an input changes", async () => {
    importLabelFile.mockResolvedValueOnce(REPORT).mockRejectedValueOnce(
      new ApiDetailError("Only 0 of 12 labelled rows (0%) name a cell. import with strip_suffix.", 400),
    );
    render(<LabelImportCard datasetId="ds" />);
    pick("cell\tlabel\nA\tT\n", "t.tsv");
    await screen.findByLabelText("Label column");
    fireEvent.click(screen.getByRole("button", { name: "Check the match" }));
    await screen.findByTestId("label-import-report");
    fireEvent.click(screen.getByRole("checkbox"));
    expect(screen.queryByTestId("label-import-report")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Check the match" }));
    expect((await screen.findByRole("alert")).textContent).toContain("import with strip_suffix");
    expect(importLabelFile.mock.calls[1]![2].stripSuffix).toBe(true);
  });

  it("by order sends no barcode column, and refuses a name already taken", async () => {
    render(<LabelImportCard datasetId="ds" />);
    pick("cluster\nB\n", "order.csv");
    const names = (await screen.findByLabelText("Cell name column")) as HTMLSelectElement;
    expect(names.value).toBe("\u0000order");
    // "cluster" is taken, so the suggestion is another name; typing it is refused.
    expect((screen.getByLabelText("New column name") as HTMLInputElement).value).toBe("imported_labels");
    fireEvent.change(screen.getByLabelText("New column name"), { target: { value: "Cluster" } });
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("A column named 'Cluster' already exists."));
    expect(screen.getByRole("button", { name: "Check the match" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("New column name"), { target: { value: "author" } });
    importLabelFile.mockResolvedValueOnce({ ...REPORT, match: "order" });
    fireEvent.click(screen.getByRole("button", { name: "Check the match" }));
    await screen.findByTestId("label-import-report");
    expect(importLabelFile.mock.calls[0]![2].barcodeColumn).toBeNull();
  });
});
