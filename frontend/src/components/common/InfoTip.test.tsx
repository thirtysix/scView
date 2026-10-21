import { describe, it, expect } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { InfoTip } from "./InfoTip";

describe("InfoTip", () => {
  it("is hidden until hovered, focused or clicked, and reachable by keyboard", () => {
    render(<InfoTip label="About tau">Tau: 0 even across groups, 1 one group only.</InfoTip>);
    const button = screen.getByRole("button", { name: "About tau" });
    expect(screen.queryByRole("tooltip")).toBeNull();

    fireEvent.focus(button);
    expect(screen.getByRole("tooltip")).toHaveTextContent("Tau: 0 even across groups");
    expect(button).toHaveAttribute("aria-expanded", "true");
    expect(button).toHaveAttribute("aria-describedby", screen.getByRole("tooltip").id);
    fireEvent.blur(button);
    expect(screen.queryByRole("tooltip")).toBeNull();

    fireEvent.mouseEnter(button);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    fireEvent.mouseLeave(button);
    expect(screen.queryByRole("tooltip")).toBeNull();

    fireEvent.click(button);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
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
