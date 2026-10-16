import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { CustomCollection } from "@/api/genesets";
import { useCustomGeneSetsStore } from "@/stores/customGeneSetsStore";

const api = vi.hoisted(() => ({
  listCustomCollections: vi.fn(),
  uploadCustomCollection: vi.fn(),
  renameCustomCollection: vi.fn(),
  deleteCustomCollection: vi.fn(),
}));
vi.mock("@/api/genesets", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/genesets")>()),
  ...api,
}));
const client = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => client);

import { GeneSetUploadError } from "@/api/genesets";
import { CustomCollectionsGroup } from "./CustomCollectionsGroup";
import { MSigDBCollectionTree } from "./MSigDBCollectionTree";

const col = (id: string, name: string): CustomCollection => ({
  id,
  name,
  created_at: "",
  source_filename: `${name}.gmt`,
  format: "gmt",
  n_sets: 2,
  n_genes: 20,
  n_genes_present: 15,
  category: "Custom",
  available: true,
});

describe("CustomCollectionsGroup", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCustomGeneSetsStore.setState({ version: 0 });
    api.listCustomCollections.mockResolvedValue([col("custom-a", "Lab")]);
  });

  it("lists collections with how much of each the dataset measures", async () => {
    render(<CustomCollectionsGroup datasetId="ds1" />);
    expect(await screen.findByText("Lab")).toBeInTheDocument();
    expect(screen.getByText("2 sets · 15/20 genes in this dataset")).toBeInTheDocument();
    expect(api.listCustomCollections).toHaveBeenCalledWith("ds1");
    // Without a selection there are no checkboxes.
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("drops a deleted collection from the selection", async () => {
    const onChange = vi.fn();
    render(
      <CustomCollectionsGroup selected={new Set(["h.all", "custom-a", "custom-gone"])} onChange={onChange} />,
    );
    await waitFor(() => expect(onChange).toHaveBeenCalled());
    expect([...onChange.mock.calls[0]![0]]).toEqual(["h.all", "custom-a"]);
  });

  it("uploads under the chosen name and selects the new collection", async () => {
    api.uploadCustomCollection.mockResolvedValue({ ...col("custom-b", "Mine"), warnings: ["3 repeated genes were dropped."] });
    const onChange = vi.fn();
    render(<CustomCollectionsGroup selected={new Set(["h.all"])} onChange={onChange} />);
    await screen.findByText("Lab");
    const file = new File(["S\t\tA\n"], "my_sets.gmt");
    fireEvent.change(screen.getByLabelText("Gene-set file"), { target: { files: [file] } });
    const name = screen.getByLabelText("Collection name") as HTMLInputElement;
    expect(name.value).toBe("my_sets");
    fireEvent.change(name, { target: { value: "Mine" } });
    fireEvent.click(screen.getByText("Save"));
    expect(await screen.findByText(/Added "Mine" \(2 sets\)\. 3 repeated genes were dropped\./)).toBeInTheDocument();
    expect(api.uploadCustomCollection).toHaveBeenCalledWith(file, "Mine");
    expect([...onChange.mock.calls.at(-1)![0]]).toEqual(["h.all", "custom-b"]);
    expect(useCustomGeneSetsStore.getState().version).toBe(1);
  });

  it("shows every problem of a rejected file", async () => {
    api.uploadCustomCollection.mockRejectedValue(
      new GeneSetUploadError("bad.gmt: 2 problems found, nothing was saved.", [
        "line 2: set 'X' lists no genes.",
        "line 3: set 'Y' appears again (first on line 1).",
      ]),
    );
    render(<CustomCollectionsGroup />);
    await screen.findByText("Lab");
    fireEvent.change(screen.getByLabelText("Gene-set file"), {
      target: { files: [new File(["x"], "bad.gmt")] },
    });
    fireEvent.click(screen.getByText("Save"));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("2 problems found");
    expect(alert.textContent).toContain("line 2: set 'X' lists no genes.");
    expect(alert.textContent).toContain("appears again");
  });

  it("renames on Enter once, and deletes after confirming", async () => {
    api.renameCustomCollection.mockResolvedValue(col("custom-a", "Renamed"));
    api.deleteCustomCollection.mockResolvedValue(undefined);
    render(<CustomCollectionsGroup />);
    await screen.findByText("Lab");
    fireEvent.click(screen.getByLabelText("Rename Lab"));
    const input = screen.getByLabelText("Collection name");
    fireEvent.change(input, { target: { value: " Renamed " } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.blur(input);
    await waitFor(() => expect(api.renameCustomCollection).toHaveBeenCalledTimes(1));
    expect(api.renameCustomCollection).toHaveBeenCalledWith("custom-a", "Renamed");

    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    fireEvent.click(screen.getByLabelText("Delete Lab"));
    expect(api.deleteCustomCollection).not.toHaveBeenCalled();
    fireEvent.click(screen.getByLabelText("Delete Lab"));
    await waitFor(() => expect(api.deleteCustomCollection).toHaveBeenCalledWith("custom-a"));
    expect(confirm.mock.calls[0]![0]).toContain("will show it as deleted");
    confirm.mockRestore();
  });
});

describe("MSigDBCollectionTree with custom collections", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCustomGeneSetsStore.setState({ version: 0 });
    api.listCustomCollections.mockResolvedValue([col("custom-a", "Lab")]);
  });

  it("still offers custom collections when MSigDB is missing, and All selects them", async () => {
    client.apiFetch.mockResolvedValue({ collections: [], hierarchy: [], defaults: [] });
    const onChange = vi.fn();
    render(<MSigDBCollectionTree selected={new Set()} onChange={onChange} datasetId="ds1" />);
    expect(await screen.findByText(/No MSigDB collections available/)).toBeInTheDocument();
    expect(await screen.findByText("Lab")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Use Lab"));
    expect([...onChange.mock.calls.at(-1)![0]]).toEqual(["custom-a"]);
    fireEvent.click(screen.getByText("All"));
    expect([...onChange.mock.calls.at(-1)![0]]).toEqual(["custom-a"]);
  });
});
