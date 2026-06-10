import { describe, it, expect, beforeEach } from "vitest";
import { useViewStore } from "./viewStore";

describe("viewStore", () => {
  beforeEach(() => {
    useViewStore.setState({ copilotOpen: false, pendingAsk: null });
  });

  it("askCopilot opens the drawer and queues the question", () => {
    useViewStore.getState().askCopilot("What is this cluster?");
    const s = useViewStore.getState();
    expect(s.copilotOpen).toBe(true);
    expect(s.pendingAsk).toBe("What is this cluster?");
  });

  it("clearPendingAsk drains the queue without closing the drawer", () => {
    useViewStore.getState().askCopilot("hi");
    useViewStore.getState().clearPendingAsk();
    const s = useViewStore.getState();
    expect(s.pendingAsk).toBeNull();
    expect(s.copilotOpen).toBe(true);
  });
});
