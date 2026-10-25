import { useCallback, useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import type { Figure, PlotParams } from "react-plotly.js";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "@/lib/plotly";
import { rememberBaseFigure, themedData, themedLayout, useResolvedTheme } from "@/lib/theme";

// A drop-in replacement for the default `react-plotly.js` export, but bound to
// our slimmed custom Plotly bundle (src/lib/plotly.ts). Use this everywhere
// instead of `import Plot from "react-plotly.js"`.
const BasePlot = createPlotlyComponent(Plotly);

/**
 * The plot in the app theme (lib/theme.ts): layout and trace neutrals are
 * swapped for dark ones in dark mode and left alone in light mode. A theme
 * change hands Plotly a new layout, so the plot redraws in place.
 */
export function Plot(props: PlotParams) {
  const theme = useResolvedTheme();
  const { data, layout, onInitialized, onUpdate } = props;
  const themedLayoutValue = useMemo(() => themedLayout(layout, theme), [layout, theme]);
  const themedDataValue = useMemo(() => themedData(data, theme), [data, theme]);

  // Downloads re-render the light figure (lib/plotConfig.ts -> exportFigure).
  const remember = useCallback(
    (gd: HTMLElement) => rememberBaseFigure(gd, data as Data[], layout as Partial<Layout>),
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

  return (
    <BasePlot
      {...props}
      data={themedDataValue}
      layout={themedLayoutValue}
      onInitialized={handleInitialized}
      onUpdate={handleUpdate}
    />
  );
}

export default Plot;
