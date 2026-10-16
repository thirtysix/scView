import { describe, it, expect, vi, beforeEach } from "vitest";
import type { ModeBarButton, PlotlyHTMLElement } from "plotly.js";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";

const plotly = vi.hoisted(() => ({
  Icons: { camera: { name: "camera" }, disk: { name: "disk" } },
  downloadImage: vi.fn(),
}));
vi.mock("@/lib/plotly", () => ({ default: plotly }));

import { PNG_SCALE, exportFilename, plotConfig } from "./plotConfig";

const gd = {} as PlotlyHTMLElement;
const buttons = (plot: string, dataset?: string) =>
  (plotConfig(plot, dataset).modeBarButtons as ModeBarButton[][])[0]!;
const click = (b: ModeBarButton) => b.click(gd, new MouseEvent("click"));

describe("exportFilename", () => {
  it("joins scview, the dataset and the plot into a safe name", () => {
    expect(exportFilename("Kang 2018 (IFN-β)", "volcano")).toBe("scview_Kang_2018_IFN_volcano");
    expect(exportFilename("pbmc_3k", "qc_% Mitochondrial")).toBe("scview_pbmc_3k_qc_Mitochondrial");
    expect(exportFilename("../../x", "violin")).toBe("scview_x_violin");
  });

  it("leaves out an empty dataset name", () => {
    expect(exportFilename(null, "violin")).toBe("scview_violin");
    expect(exportFilename("ÄÖ", "violin")).toBe("scview_violin");
  });
});

describe("plotConfig", () => {
  beforeEach(() => {
    plotly.downloadImage.mockReset().mockResolvedValue("ok");
    useDatasetStore.setState({ currentDataset: { id: "a", name: "pbmc 3k" } as DatasetInfo });
  });

  it("shows only the two download buttons, without the Plotly logo", () => {
    const cfg = plotConfig("volcano");
    expect(cfg.displaylogo).toBe(false);
    expect(cfg.displayModeBar).toBeUndefined(); // default: on hover
    expect(buttons("volcano").map((b) => b.name)).toEqual(["scviewDownloadPng", "scviewDownloadSvg"]);
  });

  it("returns the same buttons for the same plot, so Plotly doesn't redraw on every render", () => {
    expect(buttons("volcano")).toBe(buttons("volcano"));
    expect(buttons("volcano")).not.toBe(buttons("violin"));
  });

  it("downloads a high-resolution PNG and an SVG on a white background", async () => {
    const [png, svg] = buttons("volcano");
    click(png!);
    click(svg!);
    const [pngGd, pngOpts] = plotly.downloadImage.mock.calls[0]!;
    expect(pngGd).toBe(gd);
    expect(pngOpts).toMatchObject({
      format: "png",
      scale: PNG_SCALE,
      filename: "scview_pbmc_3k_volcano",
      setBackground: "opaque",
    });
    expect(plotly.downloadImage.mock.calls[1]![1]).toMatchObject({ format: "svg", filename: "scview_pbmc_3k_volcano" });
  });

  it("names the file after the dataset at click time, or an explicit one", () => {
    const [png] = buttons("violin");
    useDatasetStore.setState({ currentDataset: { id: "a", name: "renamed" } as DatasetInfo });
    click(png!);
    expect(plotly.downloadImage.mock.calls[0]![1].filename).toBe("scview_renamed_violin");
    click(buttons("composition", "Dataset A")[0]!);
    expect(plotly.downloadImage.mock.calls[1]![1].filename).toBe("scview_Dataset_A_composition");
  });
});
