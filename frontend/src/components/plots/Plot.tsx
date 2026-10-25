import { useCallback, useMemo, useRef } from "react";
import type { Data, Layout } from "plotly.js";
import type { Figure, PlotParams } from "react-plotly.js";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "@/lib/plotly";
import { rememberBaseFigure, themedData, themedLayout, useResolvedTheme } from "@/lib/theme";
import { a11yLayout, useA11y } from "@/lib/accessibility";
import { plotAriaLabel, plotTable, type PlotTable } from "@/lib/plotA11y";
import { useReducedMotion } from "@/lib/reducedMotion";

// A drop-in replacement for the default `react-plotly.js` export, but bound to
// our slimmed custom Plotly bundle (src/lib/plotly.ts). Use this everywhere
// instead of `import Plot from "react-plotly.js"`.
const BasePlot = createPlotlyComponent(Plotly);

/**
 * The plot in the app theme (lib/theme.ts): layout and trace neutrals are
 * swapped for dark ones in dark mode and left alone in light mode. A theme
 * change hands Plotly a new layout, so the plot redraws in place. In
 * accessibility mode (lib/accessibility.ts) no text is below 12 px.
 *
 * For screen readers (lib/plotA11y.ts) the plotting area is an image named
 * from the title, kind and axis titles (the mode bar stays reachable), and a
 * small figure (<= 200 points of bar/scatter) comes with its numbers as a
 * visually hidden table. Under reduced motion, layout transitions are off.
 */
export function Plot(props: PlotParams) {
  const theme = useResolvedTheme();
  const a11y = useA11y();
  const { data, layout, onInitialized, onUpdate } = props;
  const reducedMotion = useReducedMotion();
  const themedLayoutValue = useMemo(() => {
    const out = a11yLayout(themedLayout(layout, theme), a11y) as Partial<Layout>;
    return reducedMotion ? { ...out, transition: { duration: 0, easing: "linear" as const } } : out;
  }, [layout, theme, a11y, reducedMotion]);
  const label = useMemo(() => plotAriaLabel(data as Data[], layout as Partial<Layout>), [data, layout]);
  const table = useMemo(() => plotTable(data as Data[], layout as Partial<Layout>), [data, layout]);
  const labelRef = useRef(label);
  labelRef.current = label;
  const themedDataValue = useMemo(() => themedData(data, theme), [data, theme]);

  // Downloads re-render the light figure (lib/plotConfig.ts -> exportFigure).
  const remember = useCallback(
    (gd: HTMLElement) => {
      rememberBaseFigure(gd, data as Data[], layout as Partial<Layout>);
      nameFigure(gd, labelRef.current);
    },
    [data, layout],
  );
  const handleInitialized = useCallback(
    (figure: Readonly<Figure>, gd: Readonly<HTMLElement>) => {
      remember(gd as HTMLElement);
      onInitialized?.(figure, gd);
    },
    [remember, onInitialized],
  );
  const handleUpdate = useCallback(
    (figure: Readonly<Figure>, gd: Readonly<HTMLElement>) => {
      remember(gd as HTMLElement);
      onUpdate?.(figure, gd);
    },
    [remember, onUpdate],
  );

  const plot = (
    <BasePlot
      {...props}
      data={themedDataValue}
      layout={themedLayoutValue}
      onInitialized={handleInitialized}
      onUpdate={handleUpdate}
    />
  );
  if (!table) return plot;
  return (
    <>
      {plot}
      <PlotDataTable table={table} />
    </>
  );
}

/** The plotting area (Plotly's first svg; the mode bar is outside it) as a named image. */
function nameFigure(gd: HTMLElement, label: string): void {
  const svg = gd.querySelector?.("svg.main-svg");
  if (!svg) return;
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", label);
}

/** A small figure's numbers, for screen readers only (sr-only takes no space). */
function PlotDataTable({ table }: { table: PlotTable }) {
  return (
    <table className="sr-only" data-testid="plot-data-table">
      <caption>{table.caption}</caption>
      <thead>
        <tr>
          {table.columns.map((c, i) => (
            <th key={i} scope="col">{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {table.rows.map((row, i) => (
          <tr key={i}>
            {row.map((v, j) => (
              <td key={j}>{v}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default Plot;
