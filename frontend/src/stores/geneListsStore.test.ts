import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  GENE_LISTS_STORAGE_KEY,
  sanitizeStored,
  useGeneListsStorageStatus,
  useGeneListsStore,
} from "./geneListsStore";
import { resetViewForDataset, useDatasetStore } from "./datasetStore";
import type { DatasetInfo } from "@/api/types";

const stored = () => JSON.parse(localStorage.getItem(GENE_LISTS_STORAGE_KEY) ?? "null")?.state;
const names = () => useGeneListsStore.getState().lists.map((l) => l.name);
const genesOf = (id: string) => useGeneListsStore.getState().lists.find((l) => l.id === id)!.genes;

describe("geneListsStore", () => {
  beforeEach(() => {
    localStorage.clear();
    useGeneListsStore.setState({ lists: [], selectedId: null });
    useGeneListsStorageStatus.setState({ failed: false });
  });
  afterEach(() => vi.restoreAllMocks());

  it("creates lists with unique names and selects the new one", () => {
    const s = useGeneListsStore.getState();
    const a = s.create("T cells", ["CD3E", "cd3e", "CD3D"], "Markers: T")!;
    const b = s.create("t cells")!;
    expect(names()).toEqual(["T cells", "t cells (2)"]);
    expect(genesOf(a)).toEqual(["CD3E", "CD3D"]);
    expect(useGeneListsStore.getState().lists[0]!.source).toBe("Markers: T");
    expect(useGeneListsStore.getState().selectedId).toBe(b);
  });

  it("renames (unique among the others), duplicates and deletes", () => {
    const s = useGeneListsStore.getState();
    const a = s.create("A", ["X"])!;
    const b = s.create("B", ["Y"])!;
    s.rename(b, "a");
    expect(names()).toEqual(["A", "a (2)"]);
    s.rename(a, "A"); // its own name is not a clash
    expect(names()[0]).toBe("A");
    s.rename(a, "   "); // a blank name is ignored
    expect(names()[0]).toBe("A");
    const c = s.duplicate(a)!;
    expect(names()).toEqual(["A", "a (2)", "A copy"]);
    expect(genesOf(c)).toEqual(["X"]);
    useGeneListsStore.getState().select(b);
    s.remove(b);
    expect(names()).toEqual(["A", "A copy"]);
    // the neighbour of a deleted selected list is selected
    expect(useGeneListsStore.getState().selectedId).toBe(c);
  });

  it("adds new genes only, removes and replaces", () => {
    const s = useGeneListsStore.getState();
    const id = s.create("L", ["Cd3e"])!;
    expect(s.addGenes(id, ["CD3E", "Ms4a1", "Lyz2"])).toBe(2);
    expect(genesOf(id)).toEqual(["Cd3e", "Ms4a1", "Lyz2"]);
    s.removeGene(id, "Ms4a1");
    expect(genesOf(id)).toEqual(["Cd3e", "Lyz2"]);
    s.replaceGenes(id, ["A", "a", "B"]);
    expect(genesOf(id)).toEqual(["A", "B"]);
  });

  it("imports pasted text and files", () => {
    const s = useGeneListsStore.getState();
    expect(s.importText("Pasted", "  ")).toBeNull();
    const id = s.importText("Pasted", "CD3E, CD3D\nCD3G")!;
    expect(genesOf(id)).toEqual(["CD3E", "CD3D", "CD3G"]);
    const ids = s.importFile("lib.gmt", "S1\t\tA\tB\nS2\t\tC\n");
    expect(ids.map(genesOf)).toEqual([["A", "B"], ["C"]]);
    expect(() => s.importFile("x.gmt", "S\t\n")).toThrow(/no genes/);
  });

  it("keeps the lists in localStorage, and reads them back on a reload", async () => {
    const id = useGeneListsStore.getState().create("Kept", ["CD3E"])!;
    expect(stored().lists.map((l: { name: string }) => l.name)).toEqual(["Kept"]);
    useGeneListsStore.setState({ lists: [], selectedId: null }, false);
    // what a reload does: the store reads the key again
    localStorage.setItem(
      GENE_LISTS_STORAGE_KEY,
      JSON.stringify({ state: { lists: [{ id, name: "Kept", genes: ["CD3E"], created: "t", updated: "t" }], selectedId: id }, version: 1 }),
    );
    await useGeneListsStore.persist.rehydrate();
    expect(names()).toEqual(["Kept"]);
    expect(useGeneListsStore.getState().selectedId).toBe(id);
  });

  it("is not touched by a dataset switch or resetViewForDataset", () => {
    useGeneListsStore.getState().create("Mine", ["CD3E"]);
    const ds = (id: string) => ({ id, name: id, obs_columns: [], available_embeddings: [] }) as unknown as DatasetInfo;
    useDatasetStore.getState().setCurrentDataset(ds("one"));
    useDatasetStore.getState().setCurrentDataset(ds("two"));
    resetViewForDataset(ds("two"));
    expect(names()).toEqual(["Mine"]);
  });

  it("keeps working when localStorage is blocked, and says the lists are not saved", async () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    const id = useGeneListsStore.getState().create("Here", ["CD3E"]);
    expect(id).not.toBeNull();
    expect(names()).toEqual(["Here"]);
    expect(useGeneListsStorageStatus.getState().failed).toBe(true);
    await expect(useGeneListsStore.persist.rehydrate()).resolves.toBeUndefined();
    vi.restoreAllMocks();
    useGeneListsStore.getState().create("Again");
    expect(useGeneListsStorageStatus.getState().failed).toBe(false);
  });

  it("survives a corrupt stored value", async () => {
    localStorage.setItem(GENE_LISTS_STORAGE_KEY, "{not json");
    await useGeneListsStore.persist.rehydrate();
    expect(useGeneListsStore.getState().lists).toEqual([]);
  });
});

describe("sanitizeStored", () => {
  it("drops malformed lists and an unknown selection", () => {
    const out = sanitizeStored({
      lists: [
        { id: "a", name: " A ", genes: ["X", 3, "x", "Y"] },
        { id: "a", name: "dup id", genes: [] },
        { id: "b", name: "", genes: ["X"] },
        { id: "c", name: "C", genes: "X" },
        null,
      ],
      selectedId: "zzz",
    });
    expect(out.lists.map((l) => [l.id, l.name, l.genes])).toEqual([["a", "A", ["X", "Y"]]]);
    expect(out.selectedId).toBeNull();
    expect(sanitizeStored("nope")).toEqual({ lists: [], selectedId: null });
  });
});
