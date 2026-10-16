/**
 * Shared Plotly config: a minimal modebar with two figure downloads, PNG at
 * PNG_SCALE x the on-screen size and SVG (vector), named
 * `scview_<dataset>_<plot>`.
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

export { exportFilename };

/** PNG downloads render at this multiple of the on-screen size. */
export const PNG_SCALE = 3;

/** Save the plot in `gd` as PNG (at PNG_SCALE) or SVG, on a white background. */
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
  return Plotly.downloadImage(gd, opts as DownloadImgopts).then(
    () => undefined,
    (e: unknown) => console.error("[plotConfig] download failed:", e),
  );
}

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
