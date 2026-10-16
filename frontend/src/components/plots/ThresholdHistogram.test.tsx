import { describe, it, expect, vi } from "vitest";
import { act, render } from "@testing-library/react";

type Listener = (ev: Record<string, number>) => void;

/** A stand-in for Plotly's plot div: the event methods it puts on it. */
class FakeDiv {
  listeners = new Map<string, Listener[]>();
  on(name: string, fn: Listener) {
    this.listeners.set(name, [...(this.listeners.get(name) ?? []), fn]);
  }
  removeListener(name: string, fn: Listener) {
    this.listeners.set(name, (this.listeners.get(name) ?? []).filter((f) => f !== fn));
  }
  emit(name: string, ev: Record<string, number>) {
    for (const fn of this.listeners.get(name) ?? []) fn(ev);
  }
  purge() {
    this.listeners.clear();
  }
}

type PlotProps = {
  data: { type: string; yaxis?: string }[];
  layout: { shapes?: { x0: number }[]; yaxis?: { type?: string }; annotations?: { text: string; xanchor?: string }[] };
  config: { edits?: { shapePosition?: boolean } };
  onInitialized?: (fig: unknown, gd: unknown) => void;
  onUpdate?: (fig: unknown, gd: unknown) => void;
};
const seen = vi.hoisted(() => ({ props: [] as PlotProps[] }));
vi.mock("@/components/plots/Plot", () => ({
  default: (props: PlotProps) => {
    seen.props.push(props);
    return <div data-testid="plot" />;
  },
}));

import { ThresholdHistogram } from "./ThresholdHistogram";

const EDGES = [0, 1, 2, 3];
const COUNTS = [5, 9, 2];

describe("ThresholdHistogram", () => {
  it("makes only the thresholds draggable shapes; fixed markers are traces", () => {
    seen.props = [];
    render(
      <ThresholdHistogram
        edges={EDGES}
        counts={COUNTS}
        lines={[{ id: "min", value: 1.5, label: "min" }]}
        refLines={[
          { value: 1, label: "median", color: "#000" },
          { value: 2, label: "Scrublet", color: "#555", showLabel: true },
        ]}
        plotName="t"
        logY
      />,
    );
    const p = seen.props.at(-1)!;
    expect(p.layout.shapes!.map((s) => s.x0)).toEqual([1.5]);
    expect(p.data.map((t) => t.type)).toEqual(["bar", "scatter", "scatter"]);
    expect(p.data[1]!.yaxis).toBe("y2");
    expect(p.layout.annotations!.map((a) => a.text)).toEqual(["min", "Scrublet"]);
    // Labels run away from each other, so lines at the same value don't overprint.
    expect(p.layout.annotations!.map((a) => a.xanchor)).toEqual(["left", "right"]);
    expect(p.config.edits?.shapePosition).toBe(true);
    expect(p.layout.yaxis?.type).toBe("log");
  });

  it("is not editable without thresholds", () => {
    seen.props = [];
    render(<ThresholdHistogram edges={EDGES} counts={COUNTS} plotName="t" />);
    const p = seen.props.at(-1)!;
    expect(p.config.edits).toBeUndefined();
    expect(p.onInitialized).toBeUndefined();
  });

  it("reports drags and drag ends, and keeps the figure while the values are unchanged", () => {
    seen.props = [];
    const onDrag = vi.fn();
    const onCommit = vi.fn();
    const lines = [
      { id: "min", value: 1, label: "min" },
      { id: "max", value: 2.5, label: "max" },
    ];
    const { rerender } = render(
      <ThresholdHistogram edges={EDGES} counts={COUNTS} lines={lines} plotName="t" onDrag={onDrag} onCommit={onCommit} />,
    );
    const first = seen.props.at(-1)!;
    const gd = new FakeDiv();
    first.onInitialized!({}, gd);
    gd.emit("plotly_relayouting", { "shapes[1].x0": 2.2, "shapes[1].x1": 2.2 });
    expect(onDrag).toHaveBeenCalledWith("max", 2.2);
    gd.emit("plotly_relayout", { "xaxis.range[0]": 0 }); // a zoom: no threshold moved
    expect(onCommit).not.toHaveBeenCalled();

    // A parent re-rendering mid-drag with new arrays of the same values: same figure,
    // so react-plotly does not redraw the plot under the pointer.
    rerender(
      <ThresholdHistogram
        edges={EDGES}
        counts={COUNTS}
        lines={lines.map((l) => ({ ...l }))}
        refLines={[]}
        plotName="t"
        onDrag={onDrag}
        onCommit={onCommit}
      />,
    );
    const again = seen.props.at(-1)!;
    expect(again.layout).toBe(first.layout);
    expect(again.data).toBe(first.data);
    expect(again.config).toBe(first.config);

    act(() => gd.emit("plotly_relayout", { "shapes[0].x0": 0.7, "shapes[0].x1": 0.7 }));
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenCalledWith("min", 0.7);
    // A new layout after the drag, so the line snaps to the parent's value.
    expect(seen.props.at(-1)!.layout).not.toBe(first.layout);
  });

  it("tells Plotly to recompute the ranges when a threshold moves", () => {
    // Without it a layout-only change left the x axis on Plotly's empty [-1, 6].
    seen.props = [];
    const { rerender } = render(
      <ThresholdHistogram edges={EDGES} counts={COUNTS} lines={[{ id: "t", value: 1, label: "t" }]} plotName="t" />,
    );
    const before = seen.props.at(-1)!.layout as { datarevision?: string };
    rerender(<ThresholdHistogram edges={EDGES} counts={COUNTS} lines={[{ id: "t", value: 2, label: "t" }]} plotName="t" />);
    const after = seen.props.at(-1)!.layout as { datarevision?: string };
    expect(before.datarevision).toBeTruthy();
    expect(after.datarevision).not.toBe(before.datarevision);
  });

  it("listens again after a redraw purged the plot, without doubling up", () => {
    seen.props = [];
    const onCommit = vi.fn();
    render(
      <ThresholdHistogram edges={EDGES} counts={COUNTS} lines={[{ id: "t", value: 1, label: "t" }]} plotName="t" onCommit={onCommit} />,
    );
    const p = seen.props.at(-1)!;
    const gd = new FakeDiv();
    p.onInitialized!({}, gd);
    gd.purge(); // Plotly.purge / newPlot drop every listener
    p.onUpdate!({}, gd);
    p.onUpdate!({}, gd);
    act(() => gd.emit("plotly_relayout", { "shapes[0].x0": 2, "shapes[0].x1": 2 }));
    expect(onCommit).toHaveBeenCalledTimes(1);
  });
});
