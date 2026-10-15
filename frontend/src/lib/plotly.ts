// Custom partial Plotly bundle — register only the trace types scView actually
// uses (bar, scatter, scattergl, violin, heatmap) instead of pulling the full
// plotly.js-dist-min (~4.9 MB). `box` is included because `violin` depends on it.
import Plotly from "plotly.js/lib/core";
import bar from "plotly.js/lib/bar";
import scatter from "plotly.js/lib/scatter";
import scattergl from "plotly.js/lib/scattergl";
import box from "plotly.js/lib/box";
import violin from "plotly.js/lib/violin";
import heatmap from "plotly.js/lib/heatmap";

Plotly.register([bar, scatter, scattergl, box, violin, heatmap]);

export default Plotly;
