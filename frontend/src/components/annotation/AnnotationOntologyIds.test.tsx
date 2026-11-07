import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const { getLabelMatches } = vi.hoisted(() => ({ getLabelMatches: vi.fn() }));
vi.mock("@/api/ontology", () => ({ getLabelMatches }));

import { AnnotationOntologyIds } from "./AnnotationOntologyIds";

describe("AnnotationOntologyIds", () => {
  it("names a single-cell match (DC2 = cDC2, not CL's legacy pDC synonym)", async () => {
    getLabelMatches.mockResolvedValue({
      column: "cell_type", n_labels: 2, n_matched: 2, annotator: true, version: "cl/releases/2026-06-08/cl-basic.owl",
      labels: [
        { label: "DC2", n_cells: 40, cl_id: "CL:0002399", cl_name: "CD1c-positive myeloid dendritic cell",
          how: "single_cell", how_text: "what single-cell references mean by this name (CL lists it under another term)",
          source: "model", model_cl_id: "CL:0002399", url: "https://www.ebi.ac.uk/ols4/ontologies/cl/classes?obo_id=CL:0002399" },
        { label: "DC1", n_cells: 9, cl_id: "CL:0002394", cl_name: "CD141-positive myeloid dendritic cell",
          how: "single_cell", how_text: "what single-cell references mean by this name (CL lists it under another term)",
          source: "label", model_cl_id: null, url: "https://www.ebi.ac.uk/ols4/ontologies/cl/classes?obo_id=CL:0002394" },
      ],
    });
    render(<AnnotationOntologyIds datasetId="d1" column="cell_type" />);
    expect(await screen.findByText("DC2")).toBeInTheDocument();
    expect(screen.getByText("DC2").closest("tr")).toHaveTextContent("AI · single-cell name");
    expect(screen.getByText("DC1").closest("tr")).toHaveTextContent("single-cell name");
    expect(screen.getByText("DC1").closest("tr")).not.toHaveTextContent("undefined");
  });
});
