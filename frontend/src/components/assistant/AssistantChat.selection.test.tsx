import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import type { StreamHandlers, ViewContext } from "@/api/assistant";

const calls: ViewContext[] = [];
vi.mock("@/api/assistant", () => ({
  assistantChat: vi.fn(),
  checkClaims: vi.fn(),
  getMethods: vi.fn(),
  runPipelineSteps: vi.fn(),
  sendFeedback: vi.fn(),
  assistantChatStream: vi.fn(
    async (_ds: string | null, _q: string, _h: unknown, vc: ViewContext, h: StreamHandlers) => {
      calls.push(vc);
      h.onDelta?.("Opened the DE tab.");
      h.onDone?.({
        followups: [], model: null, claims: [],
        actions: [{ type: "open_de_selection", subtab: "de", label: "Opened the DE tab." }],
      } as never);
    },
  ),
}));

import { AssistantChat } from "./AssistantChat";
import { useCopilotSelectionStore } from "@/stores/copilotSelectionStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useDESettingsStore } from "@/stores/deSettingsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";

const ds: DatasetInfo = {
  id: "kang", name: "Kang", filename: "kang.h5ad", n_cells: 100, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], obs_columns: [], created_at: "",
};
const cells = new Set([4, 5, 6]);

beforeEach(() => {
  Element.prototype.scrollTo = () => {}; // not in jsdom
  calls.length = 0;
  localStorage.clear();
  useDatasetStore.setState({ currentDatasetId: "kang", currentDataset: ds });
  useSelectionStore.setState({ selectedCellIndices: cells, selectionFilter: null, selectionMode: "lasso" });
  useCopilotSelectionStore.setState({ excluded: null, lastSent: null });
  useDESettingsStore.getState().update({ mode: "groups" });
  useUnifiedViewStore.setState({ activeSubtab: "markers" });
  useViewStore.setState({ activePanel: "assistant", copilotOpen: false });
});

describe("co-pilot chat with a selection", () => {
  it("sends the selection, then marks it as sent; the DE action opens DE on the selection", async () => {
    render(<AssistantChat surface="panel" />);
    fireEvent.click(screen.getByRole("button", { name: "Run DE for this selection vs the rest" }));
    await waitFor(() => expect(calls).toHaveLength(1));
    expect(calls[0]!.selection).toEqual({
      n_cells: 3, source: "lasso", fresh: true, cells: { n_cells: 100, indices: [4, 5, 6] },
    });
    expect(useCopilotSelectionStore.getState().lastSent).toBe(cells);
    await waitFor(() => expect(useUnifiedViewStore.getState().activeSubtab).toBe("de"));
    expect(useDESettingsStore.getState().mode).toBe("selection");
    expect(useViewStore.getState().activePanel).toBe("unified");
  });

  it("leaves a removed selection out of the request", async () => {
    useCopilotSelectionStore.setState({ excluded: cells });
    render(<AssistantChat surface="panel" />);
    fireEvent.change(screen.getByRole("textbox", { name: "Message to the co-pilot" }), {
      target: { value: "What are these cells?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(calls).toHaveLength(1));
    expect(calls[0]!.selection).toBeUndefined();
  });
});
