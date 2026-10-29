import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import type { BundleFile } from "@/api/ingest";
import { SequencingQcSideFiles } from "./SequencingQcSideFiles";

const side = (name: string, kind: string): BundleFile => ({ path: `/x/${name}`, name, kind, role: "side", confidence: 0.95 });

describe("SequencingQcSideFiles", () => {
  it("lists each side file with what it is and that it will be attached", () => {
    render(
      <SequencingQcSideFiles
        files={[
          side("pbmc_1k_v3_metrics_summary.csv", "tenx_metrics"),
          side("pbmc_1k_v3_raw_feature_bc_matrix.h5", "tenx_h5"),
          side("s1_raw_feature_bc_matrix_barcodes.tsv.gz", "tenx_barcodes"),
        ]}
      />,
    );
    const items = screen.getByTestId("seqqc-side-files").querySelectorAll("li");
    expect(Array.from(items).map((li) => li.textContent)).toEqual([
      "pbmc_1k_v3_metrics_summary.csv(Cell Ranger metrics): will be attached",
      "pbmc_1k_v3_raw_feature_bc_matrix.h5(raw (unfiltered) matrix): will be attached",
      "s1_raw_feature_bc_matrix_barcodes.tsv.gz(raw matrix part): will be attached",
    ]);
    expect(items[0]!.getAttribute("title")).toContain("not loaded as cells");
  });

  it("renders nothing without side files (and for an older backend's units)", () => {
    const { container } = render(<SequencingQcSideFiles files={[]} />);
    expect(container.innerHTML).toBe("");
    const { container: c2 } = render(<SequencingQcSideFiles />);
    expect(c2.innerHTML).toBe("");
  });

  it("names Space Ranger spatial files by what they hold", () => {
    const sp = (name: string, part: BundleFile["spatial_part"]): BundleFile =>
      ({ ...side(name, "spatial"), side_kind: "spatial", spatial_part: part });
    render(
      <SequencingQcSideFiles
        files={[sp("tissue_positions.csv", "positions"), sp("scalefactors_json.json", "scalefactors"),
          sp("tissue_hires_image.png", "hires"), sp("LN_spatial.tar.gz", "archive"), sp("odd.csv", null)]}
      />,
    );
    const items = Array.from(screen.getByTestId("seqqc-side-files").querySelectorAll("li"));
    expect(items.map((li) => li.textContent)).toEqual([
      "tissue_positions.csv(spot positions): will be attached",
      "scalefactors_json.json(image scale factors): will be attached",
      "tissue_hires_image.png(high-resolution tissue image): will be attached",
      "LN_spatial.tar.gz(Space Ranger spatial folder): will be attached",
      "odd.csv(Space Ranger spatial file): will be attached",
    ]);
    expect(items[0]!.getAttribute("title")).toContain("tissue image");
  });
});
