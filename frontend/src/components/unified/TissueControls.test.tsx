import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { SpatialInfo } from "@/api/types";
import { useTissueViewStore } from "@/stores/tissueViewStore";
import { SPATIAL_HELP } from "@/lib/help/spatial";
import { hoverTip, unhover } from "@/test/hoverTip";
import { TissueControls } from "./TissueControls";

const library = { id: "a", n_obs: 3, images: { lowres: { width: 2, height: 2, scalef: 1 } }, spot_diameter_fullres: 29.2, bounds: [0, 0, 1, 1] as [number, number, number, number] };

function renderFor(spatial: Partial<SpatialInfo>, openPlotTip = true) {
  render(<TissueControls spatial={{ technology: "visium", unit: "spot", library_column: "library_id", libraries: [library], ...spatial } as SpatialInfo} />);
  if (openPlotTip) fireEvent.click(screen.getByRole("button", { name: "About the tissue plot" }));
}

describe("TissueControls", () => {
  it("calls a Visium HD dot a bin of its size, not a spot", () => {
    renderFor({ technology: "visium_hd", unit: "bin", bin_size_um: 8 });
    expect(screen.getAllByText(/Visium HD bin \(8 µm square\)/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Visium spot/)).toBeNull();
    expect(screen.getByTitle(/under the bins/)).toBeTruthy();
  });

  it("offers Draw only for the outlines the dataset has", () => {
    useTissueViewStore.setState({ draw: "points" });
    renderFor({ technology: "xenium", unit: "cell", outlines: ["cell"] });
    expect(screen.getByRole("group", { name: "Draw" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Points" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByRole("button", { name: "Nuclei" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Cells" }));
    expect(useTissueViewStore.getState().draw).toBe("cell");
  });

  it("hides Draw without outlines (a merged run, any other dataset), and says why when an import left them out", () => {
    renderFor({ technology: "xenium", unit: "cell", outlines: [], outlines_note: "cell_boundaries.parquet matches 3 of 9 cells." });
    expect(screen.queryByRole("group", { name: "Draw" })).toBeNull();
    expect(screen.getAllByText(/matches 3 of 9 cells/).length).toBeGreaterThan(0);
  });

  it("keeps the Visium sentence for spots", () => {
    renderFor({});
    expect(screen.getAllByText(/Visium spot \(55 µm\)/).length).toBeGreaterThan(0);
  });

  it("explains section, image and draw from the help catalog on hover, and the controls keep their labels", async () => {
    renderFor({ technology: "xenium", unit: "cell", outlines: ["cell"], libraries: [library, { ...library, id: "b" }] }, false);
    for (const [text, id] of [["Section", "tissue.section"], ["Image", "tissue.image"], ["Draw", "tissue.draw"]] as const) {
      const el = screen.getByText(text);
      expect(await hoverTip(el)).toBe(SPATIAL_HELP[id]);
      unhover(el);
    }
    expect(screen.getByLabelText("Section").tagName).toBe("SELECT");
    expect(screen.getByRole("checkbox", { name: "Image" })).toBeTruthy();
    expect(screen.getByRole("group", { name: "Draw" })).toBeTruthy();
  });
});
