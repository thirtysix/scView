import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

vi.mock("@/components/assistant/AssistantChat", () => ({
  AssistantChat: () => <textarea aria-label="Message to the co-pilot" />,
}));
vi.mock("@/components/assistant/RagSourcesNote", () => ({ RagSourcesNote: () => null }));

import { CoPilotDrawer } from "./CoPilotDrawer";
import { useViewStore } from "@/stores/viewStore";

beforeEach(() => useViewStore.setState({ copilotOpen: false, activePanel: "unified" }));

describe("co-pilot drawer focus", () => {
  it("is an inert complementary region while shut", () => {
    const { container } = render(<CoPilotDrawer />);
    const drawer = container.querySelector("aside")!;
    expect(drawer).toHaveAttribute("aria-label", "AI co-pilot");
    expect(drawer).toHaveAttribute("inert");
  });

  it("opening moves focus to the message box; Escape hides it and focus returns to the launcher", () => {
    render(<CoPilotDrawer />);
    const launcher = screen.getByRole("button", { name: "Ask the AI co-pilot" });
    launcher.focus();
    fireEvent.click(launcher);
    expect(screen.getByRole("textbox", { name: "Message to the co-pilot" })).toHaveFocus();
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(useViewStore.getState().copilotOpen).toBe(false);
    expect(screen.getByRole("button", { name: "Ask the AI co-pilot" })).toHaveFocus();
  });

  it("opened from elsewhere (a legend's ask button), focus goes back there", () => {
    render(
      <>
        <button type="button">Ask about NK</button>
        <CoPilotDrawer />
      </>,
    );
    const ask = screen.getByRole("button", { name: "Ask about NK" });
    ask.focus();
    act(() => useViewStore.setState({ copilotOpen: true }));
    expect(screen.getByRole("textbox")).toHaveFocus();
    fireEvent.click(screen.getByRole("button", { name: /Hide the co-pilot/ }));
    expect(ask).toHaveFocus();
  });
});
