import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "@/lib/plotly";

// A drop-in replacement for the default `react-plotly.js` export, but bound to
// our slimmed custom Plotly bundle (src/lib/plotly.ts). Use this everywhere
// instead of `import Plot from "react-plotly.js"`.
export const Plot = createPlotlyComponent(Plotly);
export default Plot;
