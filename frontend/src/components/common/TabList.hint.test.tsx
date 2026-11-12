import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TabList } from "./TabList";

describe("TabList hints", () => {
  it("shows a tab's hint in the styled tip after a short hover, not as a browser title", () => {
    vi.useFakeTimers();
    render(
      <TabList
        tabs={[{ id: "de", label: "DE", hint: "Differential expression between groups." }, { id: "markers", label: "Markers" }]}
        active="markers"
        onChange={() => {}}
        label="Analysis"
        idPrefix="t"
        tabClassName={() => ""}
      />,
    );
    const tab = screen.getByRole("tab", { name: "DE" });
    expect(tab).not.toHaveAttribute("title");
    fireEvent.mouseEnter(tab);
    expect(screen.queryByRole("tooltip")).toBeNull();
    act(() => vi.advanceTimersByTime(400));
    expect(screen.getByRole("tooltip")).toHaveTextContent("Differential expression between groups.");
    fireEvent.mouseLeave(tab);
    expect(screen.queryByRole("tooltip")).toBeNull();
    fireEvent.mouseEnter(screen.getByRole("tab", { name: "Markers" }));
    act(() => vi.advanceTimersByTime(400));
    expect(screen.queryByRole("tooltip")).toBeNull(); // no hint, no tip
    vi.useRealTimers();
  });
});
