import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useScoringMethodStore } from "@/stores/scoringMethodStore";

const client = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => client);
vi.mock("@/components/panels/CustomCollectionsGroup", () => ({
  CollectionOptions: () => null,
  CustomCollectionsGroup: () => null,
}));

import { UnifiedGeneSetsSubtab } from "./UnifiedGeneSetsSubtab";

const dataset = {
  id: "ds1", name: "kang", filename: "kang.h5ad", n_cells: 6, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], obs_columns: [], created_at: "",
} as DatasetInfo;

const sets = (n: number) =>
  Array.from({ length: n }, (_, i) => ({ name: `SET_${i}`, collection: "c2", genes: ["A"], n_genes: 1 }));

function respond(nSets: number, listed: number) {
  client.apiFetch.mockImplementation((url: string) => {
    if (url.includes("/genesets/collections")) {
      return Promise.resolve({
        dataset_id: "ds1",
        collections: [{ id: "c2", name: "C2", category: "MSigDB", n_sets: nSets, available: true }],
      });
    }
    if (url.includes("/genesets/search")) return Promise.resolve({ results: sets(listed) });
    return Promise.reject(new Error(`unexpected ${url}`));
  });
}

beforeEach(() => {
  client.apiFetch.mockReset();
  useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
  useScoringMethodStore.setState({ method: "score_genes" });
});

describe("UnifiedGeneSetsSubtab list cap", () => {
  it("says the list is the first 200 of the collection's sets when it has more", async () => {
    respond(3_050, 200);
    render(<UnifiedGeneSetsSubtab onScoreComplete={vi.fn()} groupByColumn="" setGroupByColumn={vi.fn()} />);
    await screen.findByText("SET_0");
    expect(screen.getByTestId("genesets-cap")).toHaveTextContent(
      "The first 200 of the collection's 3,050 sets, in file order; the filter above searches only these.",
    );
  });

  it("says nothing when the collection fits", async () => {
    respond(50, 50);
    render(<UnifiedGeneSetsSubtab onScoreComplete={vi.fn()} groupByColumn="" setGroupByColumn={vi.fn()} />);
    await screen.findByText("SET_0");
    await waitFor(() => expect(screen.queryByTestId("genesets-cap")).toBeNull());
  });
});

describe("UnifiedGeneSetsSubtab scoring method", () => {
  const scored = (method: string) => ({
    score_name: "SET_0", n_cells: 3, scores: method === "aucell" ? [0.1, 0.2, 0.3] : [1, 2, 3],
    min_score: 0, max_score: 1, genes_found: ["A"], genes_missing: [], method,
    max_rank: method === "aucell" ? 703 : null, source: "log-normalised .raw", genes_undetected: [],
    share_below_max_rank: method === "aucell" ? 0.64 : null,
  });
  function respondScoring() {
    client.apiFetch.mockImplementation((url: string, init?: { body?: string }) => {
      if (url.includes("/genesets/collections")) {
        return Promise.resolve({ dataset_id: "ds1", collections: [{ id: "c2", name: "C2", category: "MSigDB", n_sets: 1, available: true }] });
      }
      if (url.includes("/genesets/search")) return Promise.resolve({ results: sets(1) });
      if (url.endsWith("/genesets/score")) return Promise.resolve(scored(JSON.parse(init!.body!).method));
      return Promise.reject(new Error(`unexpected ${url}`));
    });
  }
  const scoreCalls = () => client.apiFetch.mock.calls.filter(([u]) => String(u).endsWith("/genesets/score"));

  it("scores with the chosen method, labels it, and keeps each method's score apart", async () => {
    respondScoring();
    const onScore = vi.fn();
    render(<UnifiedGeneSetsSubtab onScoreComplete={onScore} groupByColumn="" setGroupByColumn={vi.fn()} />);
    fireEvent.click(await screen.findByText("SET_0"));
    fireEvent.change(screen.getByRole("combobox", { name: "Gene-set scoring method" }), { target: { value: "aucell" } });
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(onScore).toHaveBeenCalledTimes(1));
    expect(JSON.parse(scoreCalls()[0]![1].body)).toEqual({ gene_set: ["A"], score_name: "SET_0", method: "aucell" });
    expect(onScore.mock.calls[0]!.slice(1)).toEqual(["SET_0", ["A"], undefined, "aucell"]);
    expect(screen.getByTestId("aucell-note")).toHaveTextContent(
      "AUCell over each cell's top 703 genes (5% of those detected); 64% of cells detect fewer",
    );

    // score_genes for the same set: computed, not the AUCell score from the cache
    fireEvent.change(screen.getByRole("combobox", { name: "Gene-set scoring method" }), { target: { value: "score_genes" } });
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(onScore).toHaveBeenCalledTimes(2));
    expect(JSON.parse(scoreCalls()[1]![1].body).method).toBe("score_genes");
    expect(Array.from(onScore.mock.calls[1]![0] as Float32Array)).toEqual([1, 2, 3]);
    expect(screen.queryByTestId("aucell-note")).toBeNull();

    // Re-selecting the set shows the cached score for the current method, without a request.
    fireEvent.click(screen.getAllByText("SET_0")[0]!); // the list row, not the heading
    expect(onScore).toHaveBeenCalledTimes(3);
    expect(onScore.mock.calls[2]![4]).toBe("score_genes");
    expect(scoreCalls()).toHaveLength(2);
  });

  it("shows how GSVA scored, and why a score was refused", async () => {
    let refuse = false;
    client.apiFetch.mockImplementation((url: string, init?: { body?: string }) => {
      if (url.includes("/genesets/collections")) {
        return Promise.resolve({ dataset_id: "ds1", collections: [{ id: "c2", name: "C2", category: "MSigDB", n_sets: 1, available: true }] });
      }
      if (url.includes("/genesets/search")) return Promise.resolve({ results: sets(1) });
      if (url.endsWith("/genesets/score")) {
        if (refuse) return Promise.reject(new Error("GSVA scores up to 300 million non-zero values in one request"));
        return Promise.resolve({
          ...scored(JSON.parse(init!.body!).method), genes_undetected: ["B"],
          note: "GSVA on the log-normalised X, 13,233 genes that vary; empirical CDF over all 13,836 cells",
        });
      }
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    const onScore = vi.fn();
    render(<UnifiedGeneSetsSubtab onScoreComplete={onScore} groupByColumn="" setGroupByColumn={vi.fn()} />);
    fireEvent.click(await screen.findByText("SET_0"));
    fireEvent.change(screen.getByRole("combobox", { name: "Gene-set scoring method" }), { target: { value: "gsva" } });
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(onScore).toHaveBeenCalledTimes(1));
    expect(onScore.mock.calls[0]![4]).toBe("gsva");
    expect(screen.getByTestId("score-note")).toHaveTextContent("empirical CDF over all 13,836 cells.");
    expect(screen.getByText("1 never detected")).toHaveAttribute("title", expect.stringContaining("GSVA leaves them out"));
    expect(screen.queryByTestId("aucell-note")).toBeNull();

    refuse = true;
    fireEvent.change(screen.getByRole("combobox", { name: "Gene-set scoring method" }), { target: { value: "ssgsea" } });
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("GSVA scores up to 300 million");
    expect(screen.queryByTestId("score-note")).toBeNull();
  });
});
