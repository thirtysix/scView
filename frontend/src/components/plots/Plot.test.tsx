import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render } from "@testing-library/react";
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
import { DARK_SURFACE, plotTheme, useThemeStore } from "@/lib/theme";

const layout: Partial<Layout> = { paper_bgcolor: "white", font: { color: "#334155" } };
const data = [{ type: "bar" as const, x: [1], y: [2] }];
const last = () => seen[seen.length - 1]!;

beforeEach(() => {
  seen.length = 0;
  useThemeStore.setState({ preference: "light", systemDark: false });
});

describe("Plot wrapper", () => {
  it("passes the layout through untouched in light mode", () => {
    render(<Plot data={data} layout={layout} />);
    expect(last().layout).toBe(layout);
    expect(last().data).toBe(data);
  });

  it("gets a dark layout in dark mode, and redraws when the theme changes", () => {
    useThemeStore.setState({ preference: "dark" });
    render(<Plot data={data} layout={layout} />);
    const dark = last().layout as Partial<Layout>;
    expect(dark.paper_bgcolor).toBe(DARK_SURFACE);
    expect(dark.font!.color).toBe(plotTheme("dark").text);
    expect(dark.xaxis!.gridcolor).toBe(plotTheme("dark").grid);

    act(() => useThemeStore.setState({ preference: "light" }));
    expect(last().layout).toBe(layout);
    act(() => useThemeStore.setState({ preference: "system", systemDark: true }));
    expect((last().layout as Partial<Layout>).paper_bgcolor).toBe(DARK_SURFACE);
  });

  it("keeps the same layout object across re-renders, so Plotly does not redraw for nothing", () => {
    useThemeStore.setState({ preference: "dark" });
    const { rerender } = render(<Plot data={data} layout={layout} />);
    const first = last().layout;
    rerender(<Plot data={data} layout={layout} />);
    expect(last().layout).toBe(first);
  });

  it("still calls the plot's own onInitialized", () => {
    const onInitialized = vi.fn();
    render(<Plot data={data} layout={layout} onInitialized={onInitialized} />);
    const gd = document.createElement("div");
    (last().onInitialized as (f: unknown, g: HTMLElement) => void)({}, gd);
    expect(onInitialized).toHaveBeenCalledWith({}, gd);
  });
});
