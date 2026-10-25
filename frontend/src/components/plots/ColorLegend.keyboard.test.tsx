import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ColorLegend, legendKeyTarget } from "./ColorLegend";

const CATS = ["B", "NK", "T"];
const COLORS: [number, number, number][] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];

function renderLegend(props: Partial<Parameters<typeof ColorLegend>[0]> = {}) {
  const onCategoryClick = vi.fn();
  const onCategoryToggle = vi.fn();
  render(
    <ColorLegend
      type="categorical"
      categories={CATS}
      categoryColors={COLORS}
      label="cell_type"
      onCategoryClick={onCategoryClick}
      onCategoryToggle={onCategoryToggle}
      pickedCategories={[]}
      onAskAbout={() => {}}
      onRenameCategory={() => {}}
      {...props}
    />,
  );
  return { onCategoryClick, onCategoryToggle };
}
const entry = (name: string) => screen.getByRole("button", { name, pressed: undefined });

describe("legend by keyboard", () => {
  it("is one Tab stop: arrows, Home and End move between the entries", () => {
    renderLegend();
    expect(screen.getByRole("group", { name: /Legend: cell_type/ })).toBeInTheDocument();
    expect(entry("B")).toHaveAttribute("tabindex", "0");
    expect(entry("NK")).toHaveAttribute("tabindex", "-1");
    entry("B").focus();
    fireEvent.keyDown(entry("B"), { key: "ArrowDown" });
    expect(entry("NK")).toHaveFocus();
    expect(entry("NK")).toHaveAttribute("tabindex", "0");
    expect(entry("B")).toHaveAttribute("tabindex", "-1");
    fireEvent.keyDown(entry("NK"), { key: "End" });
    expect(entry("T")).toHaveFocus();
    fireEvent.keyDown(entry("T"), { key: "ArrowDown" });
    expect(entry("T")).toHaveFocus();
    fireEvent.keyDown(entry("T"), { key: "Home" });
    expect(entry("B")).toHaveFocus();
  });

  it("Enter or Space acts as a click (the browser's own button activation); with Shift it picks", () => {
    const { onCategoryClick, onCategoryToggle } = renderLegend();
    // A key's synthetic click carries no modifiers: a plain click on the entry.
    fireEvent.click(entry("NK"));
    expect(onCategoryClick).toHaveBeenCalledWith("NK");
    fireEvent.keyDown(entry("T"), { key: "Enter", shiftKey: true });
    fireEvent.keyDown(entry("B"), { key: " ", ctrlKey: true });
    expect(onCategoryToggle.mock.calls).toEqual([["T"], ["B"]]);
    fireEvent.keyDown(entry("NK"), { key: "Enter" });
    expect(onCategoryToggle).toHaveBeenCalledTimes(2);
  });

  it("announces the state: the selected group, or the picked ones, are pressed", () => {
    renderLegend({ selectedCategory: "NK" });
    expect(entry("NK")).toHaveAttribute("aria-pressed", "true");
    expect(entry("B")).toHaveAttribute("aria-pressed", "false");
    // The selected group starts in the Tab order.
    expect(entry("NK")).toHaveAttribute("tabindex", "0");
  });

  it("picks win over a selected group, and the hover-only actions have names", () => {
    renderLegend({ selectedCategory: "NK", pickedCategories: ["B", "T"] });
    expect(entry("B")).toHaveAttribute("aria-pressed", "true");
    expect(entry("NK")).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "Ask the co-pilot about NK" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Rename NK" })).toBeInTheDocument();
  });

  it("only the current row's actions are in the Tab order (Tab goes entry, its actions, out)", () => {
    renderLegend();
    expect(screen.getByRole("button", { name: "Rename B" })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("button", { name: "Rename NK" })).toHaveAttribute("tabindex", "-1");
    entry("B").focus();
    fireEvent.keyDown(entry("B"), { key: "ArrowDown" });
    expect(screen.getByRole("button", { name: "Ask the co-pilot about NK" })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("button", { name: "Ask the co-pilot about B" })).toHaveAttribute("tabindex", "-1");
  });
});

describe("legendKeyTarget", () => {
  it("clamps at the ends and ignores other keys", () => {
    expect(legendKeyTarget("ArrowDown", 0, 3)).toBe(1);
    expect(legendKeyTarget("ArrowRight", 2, 3)).toBe(2);
    expect(legendKeyTarget("ArrowUp", 0, 3)).toBe(0);
    expect(legendKeyTarget("End", 0, 3)).toBe(2);
    expect(legendKeyTarget("a", 0, 3)).toBeNull();
    expect(legendKeyTarget("ArrowDown", 0, 0)).toBeNull();
  });
});
