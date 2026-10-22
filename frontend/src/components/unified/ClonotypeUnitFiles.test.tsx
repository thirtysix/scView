import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import type { IngestUnit } from "@/api/ingest";
import { ClonotypeUnitFiles } from "./ClonotypeUnitFiles";

const unit = (files: IngestUnit["files"]): IngestUnit => ({
  label: "pbmc", format: "tenx_h5", files, complete: true, missing_roles: [], issues: [],
});

describe("ClonotypeUnitFiles", () => {
  it("lists the contig tables bundled with a sample and nothing else", () => {
    const { container } = render(
      <ClonotypeUnitFiles unit={unit([{ path: "/a", name: "pbmc.h5", kind: "tenx_h5", role: "data", confidence: 1 }])} />,
    );
    expect(container).toBeEmptyDOMElement();
    render(
      <ClonotypeUnitFiles
        unit={unit([
          { path: "/a", name: "pbmc.h5", kind: "tenx_h5", role: "data", confidence: 1 },
          { path: "/b", name: "pbmc_t_filtered_contig_annotations.csv", kind: "vdj_contigs", role: "vdj_contigs", confidence: 0.95 },
        ])}
      />,
    );
    expect(screen.getByTestId("unit-vdj-files")).toHaveTextContent("pbmc_t_filtered_contig_annotations.csv (TCR/BCR contigs)");
  });
});
