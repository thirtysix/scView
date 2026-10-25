import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SelectByGroupControl } from "./SelectByGroup";

const source = (onSelect = vi.fn()) => ({
  column: "cell_type",
  categories: ["B", "NK", "T"],
  counts: [10, 20, 30],
  onSelect,
});

describe("Select by group (keyboard path to a lasso)", () => {
  it("opens a dialog with focus inside, keeps Tab in it, and selects the checked groups", () => {
    const onSelect = vi.fn();
    render(<SelectByGroupControl source={source(onSelect)} />);
    const opener = screen.getByRole("button", { name: "By group" });
    opener.focus();
    fireEvent.click(opener);
    expect(opener).toHaveAttribute("aria-expanded", "true");
    const dialog = screen.getByRole("dialog", { name: "Select cells by group" });
    const first = screen.getByRole("checkbox", { name: /^B/ });
    expect(first).toHaveFocus();

    // Tab from the last control wraps to the first; Shift+Tab from the first to the last.
    const cancel = screen.getByRole("button", { name: "Cancel" });
    const select = screen.getByRole("button", { name: "Select" });
    expect(select).toBeDisabled(); // nothing picked yet: Cancel is the last focusable
    cancel.focus();
    fireEvent.keyDown(dialog, { key: "Tab" });
    expect(first).toHaveFocus();
    fireEvent.keyDown(dialog, { key: "Tab", shiftKey: true });
    expect(cancel).toHaveFocus();

    fireEvent.click(screen.getByRole("checkbox", { name: /^NK/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /^T/ }));
    expect(screen.getByText("50 cells")).toBeInTheDocument();
    fireEvent.click(select);
    expect(onSelect).toHaveBeenCalledWith(["NK", "T"]);
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(opener).toHaveFocus();
  });

  it("Escape closes it, hands focus back, and goes no further (not the panel's clear-selection Escape)", () => {
    const outer = vi.fn();
    window.addEventListener("keydown", outer);
    render(<SelectByGroupControl source={source()} />);
    const opener = screen.getByRole("button", { name: "By group" });
    opener.focus();
    fireEvent.click(opener);
    fireEvent.keyDown(screen.getByRole("checkbox", { name: /^B/ }), { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(opener).toHaveFocus();
    expect(outer).not.toHaveBeenCalled();
    window.removeEventListener("keydown", outer);
  });

  it("is disabled, with a reason, when the plot is not coloured by groups", () => {
    render(<SelectByGroupControl source={null} />);
    const opener = screen.getByRole("button", { name: "By group" });
    expect(opener).toBeDisabled();
    expect(opener).toHaveAttribute("title", expect.stringContaining("Colour the plot by a grouping column"));
  });
});
