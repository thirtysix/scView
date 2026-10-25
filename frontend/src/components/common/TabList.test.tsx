import { describe, it, expect } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { TabList, TabPanel } from "./TabList";

const TABS = [
  { id: "a", label: "Alpha" },
  { id: "b", label: "Beta" },
  { id: "c", label: "Gamma" },
] as const;

function Harness() {
  const [active, setActive] = useState<"a" | "b" | "c">("a");
  return (
    <>
      <TabList tabs={TABS} active={active} onChange={setActive} label="Things" idPrefix="t" tabClassName={() => ""} />
      <TabPanel idPrefix="t" id={active}>
        content {active}
      </TabPanel>
    </>
  );
}

describe("TabList", () => {
  it("is a tablist whose open tab is selected, in the Tab order and labels its panel", () => {
    render(<Harness />);
    expect(screen.getByRole("tablist", { name: "Things" })).toBeInTheDocument();
    const alpha = screen.getByRole("tab", { name: "Alpha" });
    expect(alpha).toHaveAttribute("aria-selected", "true");
    expect(alpha).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("tab", { name: "Beta" })).toHaveAttribute("tabindex", "-1");
    const panel = screen.getByRole("tabpanel", { name: "Alpha" });
    expect(alpha).toHaveAttribute("aria-controls", panel.id);
  });

  it("arrow keys, Home and End move to and open a tab, wrapping at the ends", () => {
    render(<Harness />);
    const tab = (name: string) => screen.getByRole("tab", { name });
    tab("Alpha").focus();
    fireEvent.keyDown(tab("Alpha"), { key: "ArrowRight" });
    expect(tab("Beta")).toHaveAttribute("aria-selected", "true");
    expect(tab("Beta")).toHaveFocus();
    expect(screen.getByRole("tabpanel")).toHaveTextContent("content b");
    fireEvent.keyDown(tab("Beta"), { key: "End" });
    expect(tab("Gamma")).toHaveFocus();
    fireEvent.keyDown(tab("Gamma"), { key: "ArrowRight" });
    expect(tab("Alpha")).toHaveFocus();
    fireEvent.keyDown(tab("Alpha"), { key: "ArrowLeft" });
    expect(tab("Gamma")).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(tab("Gamma"), { key: "Home" });
    expect(tab("Alpha")).toHaveAttribute("aria-selected", "true");
  });

  it("a click opens a tab as before", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("tab", { name: "Gamma" }));
    expect(screen.getByRole("tabpanel", { name: "Gamma" })).toHaveTextContent("content c");
  });
});
