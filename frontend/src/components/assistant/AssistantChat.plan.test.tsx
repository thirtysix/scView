import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import type { AssistantAction, StreamHandlers } from "@/api/assistant";

const STEPS: AssistantAction[] = [
  { type: "set_color_by", column: "stim", label: "Colored the plot by stim." },
  { type: "detect_doublets", step: "doublet_detection", requires_confirm: true, params: {},
    label: "Detect doublets (Scrublet).", estimate: "~15 s (Scrublet)" },
];
const { runStepStreamed, getDataset } = vi.hoisted(() => ({
  runStepStreamed: vi.fn(),
  getDataset: vi.fn(),
}));
vi.mock("@/api/assistant", () => ({
  assistantChat: vi.fn(),
  checkClaims: vi.fn(),
  getMethods: vi.fn(),
  runPipelineSteps: vi.fn(),
  sendFeedback: vi.fn(),
  assistantChatStream: vi.fn(async (_d: unknown, _q: unknown, _h: unknown, _v: unknown, h: StreamHandlers) => {
    h.onDelta?.("A plan of 2 steps. Check it, then click Run plan.");
    h.onDone?.({
      followups: [], model: null, claims: [],
      actions: [{ type: "run_plan", requires_confirm: true, steps: STEPS, label: "A plan of 2 steps." }],
    } as never);
  }),
}));
vi.mock("@/api/datasets", () => ({ getDataset }));
vi.mock("@/lib/copilotPlan", async (orig) => ({
  ...(await orig<typeof import("@/lib/copilotPlan")>()),
  runStepStreamed,
}));

import { AssistantChat } from "./AssistantChat";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

const ds = (cols: string[]): DatasetInfo => ({
  id: "kang", name: "Kang", filename: "kang.h5ad", n_cells: 100, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], created_at: "",
  obs_columns: cols.map((name) => ({ name, dtype: "category", n_unique: 2, values: ["a", "b"] })),
});

beforeEach(() => {
  Element.prototype.scrollTo = () => {}; // not in jsdom
  localStorage.clear();
  runStepStreamed.mockReset();
  getDataset.mockReset();
  useDatasetStore.setState({ currentDatasetId: "kang", currentDataset: ds(["stim"]) });
  useViewStore.setState({ activePanel: "assistant", copilotOpen: false });
  useSettingsStore.setState({ colorBy: "" });
});

async function askForPlan() {
  render(<AssistantChat surface="panel" />);
  fireEvent.change(screen.getByRole("textbox", { name: "Message to the co-pilot" }), {
    target: { value: "color by stim, then detect doublets" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  return screen.findByTestId("copilot-plan");
}

describe("co-pilot plans", () => {
  it("shows a numbered checklist and runs nothing until Run plan", async () => {
    const card = await askForPlan();
    const items = within(card).getAllByRole("listitem");
    expect(items.map((li) => li.textContent)).toEqual([
      "1.Colored the plot by stim.",
      "2.Detect doublets (Scrublet).~15 s (Scrublet)",
    ]);
    expect(useSettingsStore.getState().colorBy).toBe(""); // not applied yet
    expect(runStepStreamed).not.toHaveBeenCalled();

    getDataset.mockResolvedValue(ds(["stim", "predicted_doublet", "doublet_score"]));
    runStepStreamed.mockResolvedValue({ steps_run: ["doublet_detection"], status: "complete" });
    fireEvent.click(within(card).getByRole("button", { name: "Run plan" }));
    await waitFor(() => expect(within(card).getByRole("status").textContent).toBe("All 2 steps done."));
    expect(runStepStreamed).toHaveBeenCalledWith("kang", "doublet_detection", {});
    expect(useSettingsStore.getState().colorBy).toBe("stim");
    expect(useDatasetStore.getState().currentDataset?.obs_columns.map((c) => c.name)).toContain("predicted_doublet");
    expect(within(card).queryByRole("button", { name: "Run plan" })).toBeNull();
  });

  it("says which step failed and why", async () => {
    const card = await askForPlan();
    getDataset.mockResolvedValue(ds(["stim"])); // the run left no doublet columns
    runStepStreamed.mockResolvedValue({ steps_run: ["doublet_detection"], status: "complete" });
    fireEvent.click(within(card).getByRole("button", { name: "Run plan" }));
    await waitFor(() =>
      expect(within(card).getByRole("status").textContent).toBe(
        "Step 2 (Detect doublets (Scrublet)) failed: no doublet columns were added.",
      ),
    );
  });

  it("Cancel before running drops the plan", async () => {
    const card = await askForPlan();
    fireEvent.click(within(card).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByTestId("copilot-plan")).toBeNull());
    expect(runStepStreamed).not.toHaveBeenCalled();
  });
});
