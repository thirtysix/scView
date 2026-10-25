import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ExportMenu } from "./ExportMenu";

const items = (onSelect = vi.fn()) => [
  { key: "a", label: "CSV", onSelect },
  { key: "b", label: "Excel", onSelect, disabled: true },
  { key: "c", label: "PNG", onSelect },
];

describe("export menu by keyboard", () => {
  it("opening focuses the first item; Up/Down/Home/End move over the enabled items", () => {
    render(<ExportMenu items={items()} />);
    fireEvent.click(screen.getByRole("button", { name: /Export/ }));
    const menu = screen.getByRole("menu", { name: "Export" });
    expect(screen.getByRole("menuitem", { name: "CSV" })).toHaveFocus();
    fireEvent.keyDown(menu, { key: "ArrowDown" });
    expect(screen.getByRole("menuitem", { name: "PNG" })).toHaveFocus(); // Excel is disabled
    fireEvent.keyDown(menu, { key: "ArrowDown" });
    expect(screen.getByRole("menuitem", { name: "CSV" })).toHaveFocus();
    fireEvent.keyDown(menu, { key: "ArrowUp" });
    expect(screen.getByRole("menuitem", { name: "PNG" })).toHaveFocus();
    fireEvent.keyDown(menu, { key: "Home" });
    expect(screen.getByRole("menuitem", { name: "CSV" })).toHaveFocus();
    fireEvent.keyDown(menu, { key: "End" });
    expect(screen.getByRole("menuitem", { name: "PNG" })).toHaveFocus();
  });

  it("Escape closes it and returns focus to its button; Tab closes it", () => {
    render(<ExportMenu items={items()} />);
    const button = screen.getByRole("button", { name: /Export/ });
    fireEvent.click(button);
    fireEvent.keyDown(screen.getByRole("menu"), { key: "Escape" });
    expect(screen.queryByRole("menu")).toBeNull();
    expect(button).toHaveFocus();
    expect(button).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(button);
    fireEvent.keyDown(screen.getByRole("menu"), { key: "Tab" });
    expect(screen.queryByRole("menu")).toBeNull();
  });
});
