import { useCallback, useMemo, useRef, useState } from "react";
import type { Data, Layout, PlotRelayoutEvent } from "plotly.js";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import { relayoutLineValue } from "@/lib/qcThresholds";

/** A threshold the user drags on the plot. */
export interface ThresholdLine {
  id: string;
  value: number;
  label: string;
  color?: string;
}

/** A fixed vertical marker (the median, the method's own threshold). */
export interface RefLine {
  value: number;
  label: string;
  color: string;
  dash?: "dot" | "dash" | "solid";
  /** Write the label above the line (default: hover-free and unlabelled). */
  showLabel?: boolean;
}

interface Props {
  edges: number[];
  counts: number[];
  lines?: ThresholdLine[];
  refLines?: RefLine[];
  logY?: boolean;
  plotName: string;
  barColor?: string;
  /** While a line is dragged (the plot is not redrawn meanwhile). */
  onDrag?: (id: string, value: number) => void;
  /** When a drag ends; the parent sets the line's value (rounded as it likes). */
  onCommit?: (id: string, value: number) => void;
}

const FILL_STYLE = { width: "100%", height: "100%" } as const;

/** The event methods Plotly puts on a plot's div. */
interface PlotlyEmitter {
  on: (event: string, fn: (ev: PlotRelayoutEvent) => void) => void;
  removeListener: (event: string, fn: (ev: PlotRelayoutEvent) => void) => void;
}

/**
 * A histogram with draggable vertical threshold lines (Plotly editable shapes).
 *
 * Only the thresholds are shapes, so only they can be dragged; fixed markers are
 * traces on a hidden 0-1 axis, so they span the plot on a log axis too. The
 * figure is memoised on the committed values: a drag updates only the parent's
 * live value and does not redraw the plot under the pointer.
 */
export function ThresholdHistogram({
  edges,
  counts,
  lines = [],
  refLines = [],
  logY = false,
  plotName,
  barColor = "#60a5fa",
  onDrag,
  onCommit,
}: Props) {
  // Keys, not the arrays: a parent re-rendering mid-drag passes new arrays with
  // the same values, and a new figure would redraw the plot under the pointer.
  const refKey = refLines.map((r) => `${r.value}:${r.label}:${r.color}:${r.dash ?? ""}:${r.showLabel ?? ""}`).join("|");
  const data = useMemo<Data[]>(() => {
    const mids = counts.map((_, i) => (edges[i]! + edges[i + 1]!) / 2);
    const width = edges.length > 1 ? edges[1]! - edges[0]! : 1;
    const bars: Data = {
      type: "bar",
      x: mids,
      y: counts,
      width,
      marker: { color: barColor },
      hovertemplate: "%{x:.3~f}: %{y} cells<extra></extra>",
    };
    const refs: Data[] = refLines.map((r) => ({
      type: "scatter",
      mode: "lines",
      x: [r.value, r.value],
      y: [0, 1],
      yaxis: "y2",
      name: r.label,
      line: { color: r.color, width: 1, dash: r.dash ?? "dot" },
      hoverinfo: "skip",
      showlegend: false,
    }));
    return [bars, ...refs];
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refKey stands for `refLines`
  }, [edges, counts, refKey, barColor]);

  const lineKey = lines.map((l) => `${l.id}:${l.value}:${l.color ?? ""}:${l.label}`).join("|");
  // Bumped when a drag ends: Plotly wrote the dragged position into the layout it
  // was given, so a fresh one is needed to snap the line to the parent's value.
  const [commits, setCommits] = useState(0);
  const layout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      margin: { l: 40, r: 10, t: lines.length > 0 || refLines.some((r) => r.showLabel) ? 16 : 8, b: 24 },
      xaxis: { tickfont: { size: 9 } },
      yaxis: { tickfont: { size: 9 }, type: logY ? "log" : "linear" },
      yaxis2: { overlaying: "y", range: [0, 1], visible: false, fixedrange: true },
      bargap: 0.02,
      showlegend: false,
      shapes: lines.map((l) => ({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: l.value,
        x1: l.value,
        y0: 0,
        y1: 1,
        line: { color: l.color ?? "#dc2626", width: 2, dash: "dash" },
      })),
      annotations: [
        ...lines.map((l) => ({ value: l.value, label: l.label, color: l.color ?? "#dc2626" })),
        ...refLines.filter((r) => r.showLabel),
      ].map((l) => ({
        xref: "x" as const,
        yref: "paper" as const,
        x: l.value,
        y: 1,
        yanchor: "bottom" as const,
        text: l.label,
        showarrow: false,
        font: { size: 9, color: l.color },
      })),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the keys stand for `lines` / `refLines`
    [lineKey, refKey, logY, commits],
  );

  const draggable = lines.length > 0;
  const config = useMemo(
    () => (draggable ? { ...plotConfig(plotName), edits: { shapePosition: true } } : plotConfig(plotName)),
    [plotName, draggable],
  );

  // The listeners read the latest lines and callbacks through a ref.
  const latest = useRef({ lines, onDrag, onCommit });
  latest.current = { lines, onDrag, onCommit };
  const handlers = useMemo(() => {
    const handle = (ev: PlotRelayoutEvent, final: boolean) => {
      const { lines: current, onDrag: drag, onCommit: commit } = latest.current;
      let moved = false;
      current.forEach((l, i) => {
        const v = relayoutLineValue(ev as unknown as Record<string, unknown>, i);
        if (v === null) return;
        moved = true;
        (final ? commit : drag)?.(l.id, v);
      });
      if (final && moved) setCommits((c) => c + 1);
    };
    return {
      relayout: (ev: PlotRelayoutEvent) => handle(ev, true),
      relayouting: (ev: PlotRelayoutEvent) => handle(ev, false),
    };
  }, []);
  // Attached to the plot's div by hand after every draw, not through react-plotly's
  // onRelayout prop: purging the plot (StrictMode's dev remount does) removes every
  // listener, and react-plotly re-adds a handler only when its identity changed,
  // so a stable one was silently lost (found live: drags did nothing).
  const listen = useCallback(
    (_figure: unknown, gd: unknown) => {
      const el = gd as Partial<PlotlyEmitter>;
      if (typeof el.on !== "function" || typeof el.removeListener !== "function") return;
      for (const [name, fn] of [
        ["plotly_relayout", handlers.relayout],
        ["plotly_relayouting", handlers.relayouting],
      ] as const) {
        el.removeListener(name, fn);
        el.on(name, fn);
      }
    },
    [handlers],
  );

  return (
    <Plot
      data={data}
      layout={layout}
      config={config}
      style={FILL_STYLE}
      useResizeHandler
      onInitialized={draggable ? listen : undefined}
      onUpdate={draggable ? listen : undefined}
    />
  );
}
