import { describe, it, expect, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { SelectionAnnouncer, selectionAnnouncement } from "./SelectionAnnouncer";
import { useSelectionStore } from "@/stores/selectionStore";

beforeEach(() => useSelectionStore.getState().clearSelection());

describe("selection announcements", () => {
  it("reads out the number of cells selected, and a cleared selection", () => {
    expect(selectionAnnouncement(2481, null)).toBe("2,481 cells selected");
    expect(selectionAnnouncement(1, 5)).toBe("1 cell selected");
    expect(selectionAnnouncement(null, 5)).toBe("Selection cleared");
    expect(selectionAnnouncement(null, null)).toBe("");
  });

  it("a polite live region follows the selection store, silent until it changes", () => {
    render(<SelectionAnnouncer />);
    const region = screen.getByRole("status");
    expect(region).toHaveAttribute("aria-live", "polite");
    expect(region).toHaveTextContent("");
    act(() => useSelectionStore.getState().setSelection(new Set([1, 2, 3])));
    expect(region).toHaveTextContent("3 cells selected");
    act(() => useSelectionStore.getState().clearSelection());
    expect(region).toHaveTextContent("Selection cleared");
  });
});
