import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { SelectionChip } from "./SelectionChip";
import { AskSelectionButton } from "@/components/unified/AskSelectionButton";
import { useCopilotSelectionStore } from "@/stores/copilotSelectionStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useViewStore } from "@/stores/viewStore";

const cells = new Set(Array.from({ length: 1234 }, (_, i) => i));

beforeEach(() => {
  useSelectionStore.setState({ selectedCellIndices: cells, selectionFilter: null, selectionMode: "lasso" });
  useCopilotSelectionStore.setState({ excluded: null, lastSent: null });
  useViewStore.setState({ copilotOpen: false, activePanel: "unified" });
});

describe("selection chip in the co-pilot", () => {
  it("shows the selection with suggested questions, and nothing without one", () => {
    const onAsk = vi.fn();
    const { rerender } = render(<SelectionChip onAsk={onAsk} busy={false} />);
    expect(screen.getByText(/Selection: 1,234 cells/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "What are these cells?" }));
    expect(onAsk).toHaveBeenCalledWith("What are these cells?");
    rerender(<SelectionChip onAsk={onAsk} busy />);
    expect(screen.getByRole("button", { name: "How do they differ from the rest?" })).toBeDisabled();
    act(() => useSelectionStore.setState({ selectedCellIndices: null }));
    expect(screen.queryByText(/Selection:/)).toBeNull();
  });

  it("removes the selection from the context, and adds it back", () => {
    render(<SelectionChip onAsk={() => {}} busy={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Leave the selection out of the co-pilot's context" }));
    expect(useCopilotSelectionStore.getState().excluded).toBe(cells);
    expect(screen.queryByText(/Selection: 1,234 cells/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Add the selection (1,234 cells) to the co-pilot's context" }));
    expect(useCopilotSelectionStore.getState().excluded).toBeNull();
    expect(screen.getByText(/Selection: 1,234 cells/)).toBeInTheDocument();
  });

  it("the toolbar's Ask opens the drawer with the selection back in the context", () => {
    useCopilotSelectionStore.setState({ excluded: cells });
    render(<AskSelectionButton />);
    fireEvent.click(screen.getByRole("button", { name: "Ask the co-pilot about this selection" }));
    expect(useViewStore.getState().copilotOpen).toBe(true);
    expect(useCopilotSelectionStore.getState().excluded).toBeNull();
    expect(useViewStore.getState().pendingAsk).toBeNull(); // nothing asked by itself
  });
});
