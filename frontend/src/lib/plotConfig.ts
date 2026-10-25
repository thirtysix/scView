/**
 * Shared Plotly config: a minimal modebar with three figure downloads, PNG at
 * PNG_SCALE x the on-screen size, SVG (vector) and PDF (vector, from the SVG:
 * lib/pdfExport.ts, loaded on first use), named `scview_<dataset>_<plot>`.
 *
 * Use as `config={plotConfig("volcano")}`. The buttons are cached per plot
 * name: Plotly.react compares config values by reference and does a full
 * newPlot when a click handler changes, so new closures every render would
 * redraw the plot from scratch on each update.
 */
import type { Config, DownloadImgopts, ModeBarButton, PlotlyHTMLElement } from "plotly.js";
import Plotly from "@/lib/plotly";
import { exportFilename } from "@/lib/exportFilename";
import { useDatasetStore } from "@/stores/datasetStore";
import { exportFigure } from "@/lib/theme";
import { saveBlob, svgToPdf } from "@/lib/pdfExport";

export { exportFilename };

/** PNG downloads render at this multiple of the on-screen size. */
export const PNG_SCALE = 3;

/** Save the plot in `gd` as PNG (at PNG_SCALE) or SVG, on a white background
 *  (in dark mode too: exportFigure puts the light colours back). */
export function downloadPlot(gd: PlotlyHTMLElement, format: "png" | "svg", filename: string): Promise<void> {
  const opts = {
    format,
    filename,
    width: null,
    height: null,
    scale: format === "png" ? PNG_SCALE : 1,
    // Plots with a transparent paper (e.g. the marker dot plot) would otherwise
    // come out see-through and unreadable on a dark slide.
    setBackground: "opaque",
  };
  const figure = exportFigure(gd);
  return Plotly.downloadImage(figure as PlotlyHTMLElement, opts as DownloadImgopts).then(
    () => undefined,
    (e: unknown) => console.error("[plotConfig] download failed:", e),
  );
}

/** The plot's SVG export as text (Plotly's `imageDataOnly` SVG, or its data URL). */
export function plotlySvgText(image: string): string {
  if (!image.startsWith("data:")) return image;
  const comma = image.indexOf(",");
  const body = image.slice(comma + 1);
  return image.slice(0, comma).endsWith(";base64") ? atob(body) : decodeURIComponent(body);
}

/** Save the plot in `gd` as a vector PDF (its SVG export, converted) on a white background
 *  (in dark mode too: exportFigure puts the light colours back). */
export function downloadPlotPdf(gd: PlotlyHTMLElement, filename: string): Promise<void> {
  const opts = { format: "svg", width: null, height: null, setBackground: "opaque", imageDataOnly: true };
  const figure = exportFigure(gd);
  return Plotly.toImage(figure as PlotlyHTMLElement, opts as unknown as Parameters<typeof Plotly.toImage>[1])
    .then((image) => svgToPdf(plotlySvgText(image)))
    .then((blob) => saveBlob(blob, filename, "pdf"))
    .catch((e: unknown) => console.error("[plotConfig] PDF download failed:", e));
}

/** A page with lines: the modebar's PDF button. */
const PDF_ICON = {
  width: 1000,
  height: 1000,
  path:
    "M150 0H650L900 250V1000H150Z M230 80V920H820V300H600V80Z " +
    "M330 450H720V520H330Z M330 620H720V690H330Z M330 790H600V860H330Z",
};

const buttonCache = new Map<string, ModeBarButton[]>();

function downloadButtons(plot: string, dataset: string | undefined): ModeBarButton[] {
  const key = `${plot}\u0000${dataset ?? ""}`;
  const cached = buttonCache.get(key);
  if (cached) return cached;
  // The dataset name is read at click time, so a rename shows up straight away.
  const filename = () =>
    exportFilename(dataset ?? useDatasetStore.getState().currentDataset?.name, plot);
  const buttons: ModeBarButton[] = [
    {
      name: "scviewDownloadPng",
      title: `Download PNG (${PNG_SCALE}x resolution)`,
      icon: Plotly.Icons.camera,
      click: (gd) => void downloadPlot(gd, "png", filename()),
    },
    {
      name: "scviewDownloadSvg",
      title: "Download SVG (vector, editable)",
      icon: Plotly.Icons.disk,
      click: (gd) => void downloadPlot(gd, "svg", filename()),
    },
    {
      name: "scviewDownloadPdf",
      title: "Download PDF (vector, text selectable)",
      icon: PDF_ICON,
      click: (gd) => void downloadPlotPdf(gd, filename()),
    },
  ];
  buttonCache.set(key, buttons);
  return buttons;
}

/**
 * Plotly `config` for every scView plot. `plot` names the figure in the
 * filename; `dataset` overrides the current dataset's name (e.g. for a plot
 * comparing two datasets).
 */
export function plotConfig(plot: string, dataset?: string): Partial<Config> {
  return {
    responsive: true,
    displaylogo: false,
    modeBarButtons: [downloadButtons(plot, dataset)],
  };
}
