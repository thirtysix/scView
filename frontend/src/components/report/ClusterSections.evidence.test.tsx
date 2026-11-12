import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ClusterReport, ReportSection } from "@/api/report";
import { ClusterSectionBody } from "./ClusterSections";

const ok = { status: "ok" as const };

describe("cluster report QC section", () => {
  it("lists the group's evidence, with the lines where a rule fired marked", () => {
    const section: ReportSection = {
      id: "qc", title: "Quality control", caption: "", method: "", status: "ok", note: null,
      data: {
        medians: [], has_mt: true,
        flags: { ...ok, flags: [{ flag: "low_quality", reason: "Likely low quality.", help: "" }], notes: [] },
        doublets: { status: "not_computed", note: "Not computed yet." },
        states: { status: "not_computed", note: "" },
        removed: { status: "not_applicable", note: "" },
        evidence: { ...ok, notes: ["No marker genes for 'leiden'."], lines: [
          { kind: "qc", fired: true, text: "1,115 genes vs 1,370 typical; 11.2% vs 5.9% mito." },
          { kind: "decision", fired: false, text: "The AI named it T cells in 'cell_type'." },
        ] },
      },
    };
    render(<>{ClusterSectionBody({ section, report: {} as ClusterReport, onGene: () => {} })}</>);
    const list = screen.getByTestId("report-evidence");
    expect(list.textContent).toContain("11.2% vs 5.9% mito");
    expect(screen.getByText(/11.2% vs 5.9% mito/)).toHaveClass("text-amber-800");
    expect(screen.getByText("The AI named it T cells in 'cell_type'.")).not.toHaveClass("text-amber-800");
    expect(screen.getByText("No marker genes for 'leiden'.")).toBeInTheDocument();
  });
});
