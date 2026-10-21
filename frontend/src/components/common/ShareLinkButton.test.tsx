import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

import type { DatasetInfo } from "@/api/types";
import { readShareHash } from "@/lib/shareUrl";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useGeneListsStore } from "@/stores/geneListsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { ShareLinkButton } from "./ShareLinkButton";

const kang: DatasetInfo = {
  id: "kang-id",
  name: "Kang 2018",
  filename: "k.h5ad",
  n_cells: 13836,
  n_genes: 5,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "cluster", dtype: "category", n_unique: 2, values: ["B", "CD14 Mono"] }],
  cells_fingerprint: "fp-kang",
  created_at: "",
};

const copyButton = () => screen.getByRole("button", { name: /copy link/i });
const linkBox = () => screen.getByLabelText("Link to this view") as HTMLInputElement;

let writeText: ReturnType<typeof vi.fn>;
beforeEach(() => {
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  useGeneListsStore.setState({ lists: [] });
  writeText = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
});
afterEach(() => {
  Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
});

function openKang() {
  act(() => {
    useDatasetStore.getState().setCurrentDataset(kang);
    useSettingsStore.getState().setEmbedding("X_umap");
    useSelectionStore.getState().setHighlight({ column: "cluster", value: "CD14 Mono" });
    useUnifiedViewStore.getState().setCamera({ datasetId: "kang-id", embedding: "X_umap", camera: { target: [1, 2], zoom: 4 } });
  });
}

describe("ShareLinkButton", () => {
  it("is disabled, with a reason, until a dataset is open", () => {
    render(<ShareLinkButton />);
    expect(copyButton()).toBeDisabled();
    expect(copyButton().parentElement).toHaveAttribute("title", "Open a dataset first");
    // A plain inline span around the flex button took its clicks in Chromium.
    expect(copyButton().parentElement).toHaveClass("inline-flex");
    openKang();
    expect(copyButton()).toBeEnabled();
  });

  it("copies a link that carries the view, and says what it restores", async () => {
    render(<ShareLinkButton />);
    openKang();
    act(() => useGeneListsStore.getState().addLists([{ name: "IFN", genes: ["ISG15"] }]));
    await act(async () => fireEvent.click(copyButton()));

    expect(await screen.findByText(/Link copied to the clipboard/)).toBeInTheDocument();
    const url = writeText.mock.calls[0]![0] as string;
    expect(url).toMatch(/^http:\/\/localhost:3000\/#view=v1\.z\./);
    expect(linkBox().value).toBe(url);
    const shared = await readShareHash(new URL(url).hash);
    expect(shared?.dataset).toMatchObject({ id: "kang-id", n_cells: 13836, cells_fingerprint: "fp-kang" });
    expect(shared?.view?.highlight).toEqual({ column: "cluster", value: "CD14 Mono" });
    expect(shared?.view?.camera).toEqual({ target: [1, 2], zoom: 4 });
    expect(shared).not.toHaveProperty("gene_lists");

    const dialog = screen.getByRole("dialog", { name: "Share link" });
    expect(dialog).toHaveTextContent("dataset “Kang 2018” (13,836 cells)");
    expect(dialog).toHaveTextContent("the highlighted group cluster = CD14 Mono");
    expect(dialog).toHaveTextContent("the link carries the view, not the data");
    expect(dialog).not.toHaveTextContent("Left out");
  });

  it("shows the link selected to copy by hand where the clipboard is not available (plain http)", async () => {
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
    render(<ShareLinkButton />);
    openKang();
    await act(async () => fireEvent.click(copyButton()));
    expect(await screen.findByText(/Copy this link \(Ctrl\+C/)).toBeInTheDocument();
    const box = linkBox();
    expect(box.value).toMatch(/#view=v1\./);
    expect(box).toHaveAttribute("readonly");
    expect([box.selectionStart, box.selectionEnd]).toEqual([0, box.value.length]);
  });

  it("falls back the same way when the browser refuses to copy", async () => {
    writeText.mockRejectedValue(new Error("NotAllowedError"));
    render(<ShareLinkButton />);
    openKang();
    await act(async () => fireEvent.click(copyButton()));
    expect(await screen.findByText(/Copy this link/)).toBeInTheDocument();
  });

  it("says when a large selection was left out", async () => {
    render(<ShareLinkButton />);
    act(() => {
      useDatasetStore.getState().setCurrentDataset({ ...kang, n_cells: 300_000 });
      // A lasso's cells are spread through obs order, so its bitmask does not compress.
      const cells = new Set<number>();
      let x = 7;
      while (cells.size < 100_000) {
        x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
        cells.add(x % 300_000);
      }
      useSelectionStore.getState().setSelection(cells);
    });
    await act(async () => fireEvent.click(copyButton()));
    const dialog = await screen.findByRole("dialog", { name: "Share link" });
    expect(dialog).toHaveTextContent(/Left out to keep the link short: the selection of 100,000 cells/);
    expect(linkBox().value.length).toBeLessThanOrEqual(6000);
  });

  it("closes on Escape", async () => {
    render(<ShareLinkButton />);
    openKang();
    await act(async () => fireEvent.click(copyButton()));
    await screen.findByRole("dialog", { name: "Share link" });
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Share link" })).not.toBeInTheDocument();
  });
});
