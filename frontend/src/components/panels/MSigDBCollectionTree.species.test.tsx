import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { useCustomGeneSetsStore } from "@/stores/customGeneSetsStore";

const api = vi.hoisted(() => ({ listCustomCollections: vi.fn() }));
vi.mock("@/api/genesets", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/genesets")>()),
  ...api,
}));
const client = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => client);

import { CollectionOptions } from "./CustomCollectionsGroup";
import {
  DEFAULT_MOUSE_MSIGDB_COLLECTIONS,
  DEFAULT_MSIGDB_COLLECTIONS,
  MSigDBCollectionTree,
} from "./MSigDBCollectionTree";

const col = (id: string, category: string, name: string, species: "human" | "mouse", available = true) => ({
  id, category, subcategory: "", name, description: "", default: true, n_sets: 50, available, species,
});

const mouseListing = (state: string, available = true) => ({
  species: "mouse",
  defaults: DEFAULT_MOUSE_MSIGDB_COLLECTIONS,
  mouse_download: { state, release: "2026.2.Mm", message: state === "downloading" ? "Downloading the mouse MSigDB collections." : "" },
  collections: [],
  hierarchy: [
    { category: "MH", name: "Hallmark (mouse)", species: "mouse", children: [col("mh.all", "MH", "Hallmark (mouse)", "mouse", available)] },
    { category: "H", name: "Hallmark", species: "human", children: [col("h.all", "H", "Hallmark", "human")] },
  ],
});

describe("MSigDBCollectionTree by species", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCustomGeneSetsStore.setState({ version: 0 });
    api.listCustomCollections.mockResolvedValue([]);
  });

  it("asks for the dataset's collections and swaps untouched defaults to the mouse ones", async () => {
    client.apiFetch.mockResolvedValue(mouseListing("ready"));
    const onChange = vi.fn();
    render(<MSigDBCollectionTree selected={new Set(DEFAULT_MSIGDB_COLLECTIONS)} onChange={onChange} datasetId="mm 1" />);
    expect(await screen.findByText("Mouse gene sets")).toBeInTheDocument();
    expect(screen.getByText("Human gene sets (mapped to mouse)")).toBeInTheDocument();
    expect(client.apiFetch).toHaveBeenCalledWith("/enrichment/msigdb-collections?dataset_id=mm%201");
    await waitFor(() => expect(onChange).toHaveBeenCalled());
    expect([...onChange.mock.calls[0]![0]]).toEqual(DEFAULT_MOUSE_MSIGDB_COLLECTIONS);
  });

  it("keeps a selection the user made", async () => {
    client.apiFetch.mockResolvedValue(mouseListing("ready"));
    const onChange = vi.fn();
    render(<MSigDBCollectionTree selected={new Set(["h.all"])} onChange={onChange} datasetId="mm" />);
    await screen.findByText("Mouse gene sets");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("a human dataset shows no species headings and keeps the human defaults", async () => {
    client.apiFetch.mockResolvedValue({
      species: "human",
      defaults: DEFAULT_MSIGDB_COLLECTIONS,
      collections: [],
      hierarchy: [{ category: "H", name: "Hallmark", species: "human", children: [col("h.all", "H", "Hallmark", "human")] }],
    });
    const onChange = vi.fn();
    render(<MSigDBCollectionTree selected={new Set(DEFAULT_MSIGDB_COLLECTIONS)} onChange={onChange} datasetId="hs" />);
    expect(await screen.findByText("Hallmark")).toBeInTheDocument();
    expect(screen.queryByText("Mouse gene sets")).toBeNull();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("shows the first-use download and looks again until it is done", async () => {
    client.apiFetch
      .mockResolvedValueOnce(mouseListing("downloading", false))
      .mockResolvedValue(mouseListing("ready"));
    render(<MSigDBCollectionTree selected={new Set()} onChange={vi.fn()} datasetId="mm" />);
    expect(await screen.findByRole("status")).toHaveTextContent("Downloading the mouse MSigDB collections.");
    expect(screen.getByRole("checkbox", { name: /Hallmark \(mouse\)/ })).toBeDisabled();
    await waitFor(() => expect(screen.queryByRole("status")).toBeNull(), { timeout: 5000 });
    expect(screen.getByRole("checkbox", { name: /Hallmark \(mouse\)/ })).toBeEnabled();
    expect(client.apiFetch).toHaveBeenCalledTimes(2);
  });
});

describe("CollectionOptions by species", () => {
  it("groups mouse collections first and labels the human ones", () => {
    render(
      <select aria-label="Collection">
        <CollectionOptions
          collections={[
            { id: "mh.all", name: "Hallmark (mouse)", species: "mouse" },
            { id: "h.all", name: "Hallmark", species: "human" },
            { id: "custom-a", name: "Lab", category: "Custom" },
          ]}
        />
      </select>,
    );
    const groups = [...document.querySelectorAll("optgroup")].map((g) => g.label);
    expect(groups).toEqual(["MSigDB mouse", "MSigDB human (human symbols, case-matched)", "Custom"]);
  });

  it("is unchanged for human collections", () => {
    render(
      <select aria-label="Collection">
        <CollectionOptions collections={[{ id: "h.all", name: "Hallmark", species: "human" }]} />
      </select>,
    );
    expect(document.querySelectorAll("optgroup")).toHaveLength(0);
    expect(screen.getByRole("option", { name: "Hallmark" })).toBeInTheDocument();
  });
});
