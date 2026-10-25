import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import type { Layout } from "plotly.js";

const { seen } = vi.hoisted(() => ({ seen: [] as Record<string, unknown>[] }));
// Plotly needs a real canvas: capture what the wrapper hands react-plotly.
vi.mock("react-plotly.js/factory", () => ({
  default: () => (props: Record<string, unknown>) => {
    seen.push(props);
    return <div data-testid="plotly" />;
  },
}));
vi.mock("@/lib/plotly", () => ({ default: {} }));

import { Plot } from "./Plot";
import { useThemeStore } from "@/lib/theme";
import { useA11yStore } from "@/lib/accessibility";

const layout: Partial<Layout> = { title: { text: "Cells per group" }, xaxis: { title: { text: "Group" } } };
const small = [{ type: "bar" as const, x: ["B", "NK"], y: [3, 4] }];
const last = () => seen[seen.length - 1]!;

beforeEach(() => {
  seen.length = 0;
  useThemeStore.setState({ preference: "light", systemDark: false });
  useA11yStore.setState({ enabled: false });
});

describe("Plot wrapper for screen readers", () => {
  it("names the plotting area (not the mode bar) as an image once Plotly has drawn it", () => {
    render(<Plot data={small} layout={layout} />);
    const gd = document.createElement("div");
    gd.innerHTML = '<div class="modebar"><a role="button">Download</a></div><svg class="main-svg"></svg>';
    (last().onInitialized as (f: unknown, g: HTMLElement) => void)({}, gd);
    const svg = gd.querySelector("svg.main-svg")!;
    expect(svg.getAttribute("role")).toBe("img");
    expect(svg.getAttribute("aria-label")).toBe("Bar chart: Cells per group. x axis: Group.");
    expect(gd.querySelector(".modebar")!.getAttribute("role")).toBeNull();
  });

  it("a small figure comes with its numbers as a visually hidden table", () => {
    render(<Plot data={small} layout={layout} />);
    const table = screen.getByTestId("plot-data-table");
    expect(table).toHaveClass("sr-only");
    expect(screen.getByRole("table", { name: /Cells per group.*as a table/ })).toBe(table);
    expect(screen.getAllByRole("row")).toHaveLength(3);
  });

  it("a big figure has no table", () => {
    const big = Array.from({ length: 500 }, (_, i) => i);
    render(<Plot data={[{ type: "scatter", x: big, y: big }]} layout={layout} />);
    expect(screen.queryByTestId("plot-data-table")).toBeNull();
  });

  it("under reduced motion (here: accessibility mode) layout transitions are off; otherwise untouched", () => {
    const { rerender } = render(<Plot data={small} layout={layout} />);
    expect(last().layout).toBe(layout);
    act(() => useA11yStore.setState({ enabled: true }));
    rerender(<Plot data={small} layout={layout} />);
    expect((last().layout as Partial<Layout>).transition).toEqual({ duration: 0, easing: "linear" });
  });
});
