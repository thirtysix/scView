import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { seqQcSideFiles, type BundleFile, type IngestUnit } from "@/api/ingest";
import { ClonotypeUnitFiles } from "./ClonotypeUnitFiles";

const unit = (files: IngestUnit["files"], side_files: BundleFile[] = []): IngestUnit => ({
  label: "pbmc", format: "tenx_h5", files, complete: true, missing_roles: [], issues: [], side_files,
});
const h5: BundleFile = { path: "/a", name: "pbmc.h5", kind: "tenx_h5", role: "data", confidence: 1 };
const contigs = (name: string, receptor: string | null): BundleFile => ({
  path: `/${name}`, name, kind: "vdj_contigs", role: "side", confidence: 0.95, side_kind: "vdj", receptor,
});
const metrics: BundleFile = {
  path: "/m", name: "metrics_summary.csv", kind: "tenx_metrics", role: "side", confidence: 0.95, side_kind: "metrics",
};

describe("ClonotypeUnitFiles", () => {
  it("lists the contig tables among a sample's side files and nothing else", () => {
    const { container } = render(<ClonotypeUnitFiles unit={unit([h5], [metrics])} />);
    expect(container).toBeEmptyDOMElement();
    render(
      <ClonotypeUnitFiles
        unit={unit([h5], [metrics, contigs("t_filtered_contig_annotations.csv", "T"), contigs("b.csv", "B"), contigs("x.csv", null)])}
      />,
    );
    const items = Array.from(screen.getByTestId("unit-vdj-files").querySelectorAll("li")).map((li) => li.textContent);
    expect(items).toEqual([
      "t_filtered_contig_annotations.csv (TCR contigs)",
      "b.csv (BCR contigs)",
      "x.csv (TCR/BCR contigs)",
    ]);
  });

  it("keeps contig tables out of the sequencing-QC side files", () => {
    const u = unit([h5], [metrics, contigs("t.csv", "T")]);
    expect(seqQcSideFiles(u).map((f) => f.name)).toEqual(["metrics_summary.csv"]);
  });
});
