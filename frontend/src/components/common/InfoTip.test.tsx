import { describe, it, expect, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { HoverTip, InfoTip, placeTip } from "./InfoTip";

describe("InfoTip", () => {
  it("is hidden until hovered, focused or clicked, and reachable by keyboard", () => {
    render(<InfoTip label="About tau">Tau: 0 even across groups, 1 one group only.</InfoTip>);
    const button = screen.getByRole("button", { name: "About tau" });
    expect(screen.queryByRole("tooltip")).toBeNull();

    // keyboard focus (:focus-visible) opens it
    const visible = vi.spyOn(button, "matches").mockReturnValue(true);
    fireEvent.focus(button);
    expect(screen.getByRole("tooltip")).toHaveTextContent("Tau: 0 even across groups");
    expect(button).toHaveAttribute("aria-expanded", "true");
    expect(button).toHaveAttribute("aria-describedby", screen.getByRole("tooltip").id);
    fireEvent.blur(button);
    expect(screen.queryByRole("tooltip")).toBeNull();

    // a dialog focusing its first control after a click does not
    visible.mockReturnValue(false);
    fireEvent.focus(button);
    expect(screen.queryByRole("tooltip")).toBeNull();
    fireEvent.blur(button);

    fireEvent.mouseEnter(button);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    fireEvent.mouseLeave(button);
    expect(screen.queryByRole("tooltip")).toBeNull();

    fireEvent.click(button);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
  });

  it("opens on its preferred side when it fits, else flips, and stays inside the window", () => {
    const view = { width: 1000, height: 800 };
    const tip = { width: 300, height: 100 };
    const mid = { top: 400, bottom: 412, left: 100, right: 112 };
    expect(placeTip(mid, tip, view, "above", "left")).toEqual({ top: 294, left: 100, side: "above" });
    expect(placeTip(mid, tip, view, "below", "left")).toEqual({ top: 418, left: 100, side: "below" });
    // near the top of the window (a toolbar, a tab's first row): no room above, so below
    const top = { top: 20, bottom: 32, left: 100, right: 112 };
    expect(placeTip(top, tip, view, "above", "left").side).toBe("below");
    // near the right edge: pulled back inside; right-aligned ends at the icon's right edge
    const right = { top: 400, bottom: 412, left: 950, right: 962 };
    expect(placeTip(right, tip, view, "above", "left").left).toBe(1000 - 300 - 8);
    expect(placeTip(mid, tip, view, "above", "right").left).toBe(8);
  });

  it("draws the tip outside any clipping box (in a portal on the body)", () => {
    render(
      <div style={{ overflow: "hidden" }} data-testid="box">
        <InfoTip>text</InfoTip>
      </div>,
    );
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByTestId("box")).not.toContainElement(screen.getByRole("tooltip"));
    expect(screen.getByRole("tooltip").parentElement).toBe(document.body);
  });

  it("HoverTip opens on the thing itself after a short hover", () => {
    vi.useFakeTimers();
    render(<HoverTip tip="What a log fold change is.">Min |log2FC|</HoverTip>);
    fireEvent.mouseEnter(screen.getByText("Min |log2FC|"));
    expect(screen.queryByRole("tooltip")).toBeNull();
    act(() => vi.advanceTimersByTime(400));
    expect(screen.getByRole("tooltip")).toHaveTextContent("What a log fold change is.");
    fireEvent.mouseLeave(screen.getByText("Min |log2FC|"));
    expect(screen.queryByRole("tooltip")).toBeNull();
    vi.useRealTimers();
  });

  it("does not let its click reach the row behind it", () => {
    let rowClicks = 0;
    render(
      <div onClick={() => rowClicks++}>
        <InfoTip>text</InfoTip>
      </div>,
    );
    fireEvent.click(screen.getByRole("button"));
    expect(rowClicks).toBe(0);
  });
});
