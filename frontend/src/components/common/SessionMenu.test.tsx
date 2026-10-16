import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, fireEvent, waitFor } from "@testing-library/react";

const restore = vi.hoisted(() => ({ restoreSession: vi.fn() }));
vi.mock("@/lib/sessionRestore", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/sessionRestore")>()),
  restoreSession: restore.restoreSession,
}));

import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { SessionMenu } from "./SessionMenu";

const ds: DatasetInfo = {
  id: "kang",
  name: "Kang 2018",
  filename: "k.h5ad",
  n_cells: 40,
  n_genes: 5,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [],
  cells_fingerprint: "fp1",
  created_at: "",
};

const pick = (file: File) =>
  fireEvent.change(screen.getByLabelText("Session file"), { target: { files: [file] } });
const jsonFile = (x: unknown) => new File([JSON.stringify(x)], "s.json", { type: "application/json" });

let saved: { blob: Blob | null; name: string };
beforeEach(() => {
  vi.clearAllMocks();
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  saved = { blob: null, name: "" };
  vi.stubGlobal("URL", {
    ...URL,
    createObjectURL: (b: Blob) => {
      saved.blob = b;
      return "blob:x";
    },
    revokeObjectURL: () => {},
  });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    saved.name = this.download;
  });
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("SessionMenu", () => {
  it("saves the workspace as a versioned session file named after the dataset", async () => {
    render(<SessionMenu />);
    fireEvent.click(screen.getByRole("button", { name: /session/i }));
    expect(screen.getByRole("menuitem", { name: /save session/i })).toBeDisabled(); // no dataset yet

    act(() => {
      useDatasetStore.getState().setCurrentDataset(ds);
      useSelectionStore.getState().setSelection(new Set([3, 1]));
    });
    fireEvent.click(screen.getByRole("menuitem", { name: /save session/i }));
    expect(saved.name).toMatch(/^scview_Kang_2018_session_\d{4}-\d{2}-\d{2}\.json$/);
    const file = JSON.parse(await saved.blob!.text());
    expect(file).toMatchObject({
      format: "scview-session",
      version: 1,
      dataset: { id: "kang", name: "Kang 2018", n_cells: 40, cells_fingerprint: "fp1" },
      selection: { n_cells: 40, indices: [1, 3] },
    });
  });

  it("opens a session and says what was not restored", async () => {
    restore.restoreSession.mockResolvedValue({
      headline: "Opened “Kang 2018”.",
      notes: ["Colour by “stim”: this dataset has no such column."],
      level: "warning",
    });
    render(<SessionMenu />);
    pick(jsonFile({ format: "scview-session", version: 1, dataset: { id: "kang" } }));
    const box = await screen.findByRole("status", { name: "Session restore" });
    expect(box).toHaveTextContent("Opened “Kang 2018”.");
    expect(box).toHaveTextContent("Not restored:");
    expect(box).toHaveTextContent("Colour by “stim”: this dataset has no such column.");
    expect(restore.restoreSession).toHaveBeenCalledWith({ format: "scview-session", version: 1, dataset: { id: "kang" } });
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByRole("status", { name: "Session restore" })).toBeNull();
  });

  it("refuses a file from another version or app without touching the workspace", async () => {
    render(<SessionMenu />);
    pick(jsonFile({ format: "scview-session", version: 3 }));
    expect(await screen.findByRole("alert")).toHaveTextContent("saved by a newer scView (format version 3)");
    pick(new File(["<html>"], "x.json"));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("not JSON"));
    pick(jsonFile({ format: "scview-session", version: 1, selection: { n_cells: 4, indices: [-1] } }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("selection.indices must hold whole numbers"));
    expect(restore.restoreSession).not.toHaveBeenCalled();
  });
});
