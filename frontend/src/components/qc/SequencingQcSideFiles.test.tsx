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
});
