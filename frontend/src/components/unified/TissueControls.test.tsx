import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { SpatialInfo } from "@/api/types";
import { TissueControls } from "./TissueControls";

const library = { id: "a", n_obs: 3, images: { lowres: { width: 2, height: 2, scalef: 1 } }, spot_diameter_fullres: 29.2, bounds: [0, 0, 1, 1] as [number, number, number, number] };

function renderFor(spatial: Partial<SpatialInfo>) {
  render(<TissueControls spatial={{ technology: "visium", unit: "spot", library_column: "library_id", libraries: [library], ...spatial } as SpatialInfo} />);
  fireEvent.click(screen.getByRole("button", { name: "About the tissue plot" }));
}

describe("TissueControls", () => {
  it("calls a Visium HD dot a bin of its size, not a spot", () => {
    renderFor({ technology: "visium_hd", unit: "bin", bin_size_um: 8 });
    expect(screen.getAllByText(/Visium HD bin \(8 µm square\)/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Visium spot/)).toBeNull();
    expect(screen.getByTitle(/under the bins/)).toBeTruthy();
  });

  it("keeps the Visium sentence for spots", () => {
    renderFor({});
    expect(screen.getAllByText(/Visium spot \(55 µm\)/).length).toBeGreaterThan(0);
  });
});
