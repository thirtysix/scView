import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { Claim } from "@/api/assistant";

const REFUTED = "Cells were clustered with Leiden at a resolution of 1.0.";
const SUPPORTED = "PCA was run with 50 components.";
const claim = (id: number, span: string, verdict: Claim["verdict"], reason: string): Claim => ({
  id, text: span, span, verdict, reason, cites: [], refs: ["provenance:recipe"],
  evidence: "- clustering (tool: scanpy.tl.leiden/louvain) — clustering_resolution=0.5",
  checked_by: "rules",
});

vi.mock("@/api/assistant", () => ({
  assistantChat: vi.fn(),
  assistantChatStream: vi.fn(),
  checkClaims: vi.fn(),
  runPipelineSteps: vi.fn(),
  sendFeedback: vi.fn(),
  getMethods: vi.fn(async () => ({
    methods: `${REFUTED} ${SUPPORTED}`,
    grounded: true,
    model: "m",
    claims: [
      claim(0, REFUTED, "refuted", "says 1.0 for resolution; the recipe has 0.5."),
      claim(1, SUPPORTED, "supported", "Matches the recipe: 50."),
    ],
  })),
}));

import { AssistantChat } from "./AssistantChat";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";

beforeEach(() => {
  Element.prototype.scrollTo = () => {}; // not in jsdom
  localStorage.clear();
  useDatasetStore.setState({ currentDatasetId: "kang", currentDataset: null });
  useViewStore.setState({ activePanel: "assistant", copilotOpen: false });
});

describe("Write methods", () => {
  it("shows the methods ledger with the refuted sentence in view", async () => {
    render(<AssistantChat surface="panel" />);
    fireEvent.click(screen.getByRole("button", { name: /Write methods/ }));
    await waitFor(() => expect(screen.getByText("1 refuted")).toBeInTheDocument());
    expect(screen.getByText("1 supported")).toBeInTheDocument();
    // refuted stays listed with the ledger collapsed, checked against the recipe
    expect(screen.getByText(/says 1.0 for resolution/)).toBeInTheDocument();
    expect(screen.getByText("the recorded recipe")).toBeInTheDocument();
  });
});
