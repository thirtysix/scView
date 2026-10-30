import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TargetedPanelNote, type TargetedPanelInfo } from "./TargetedPanelNote";
import { filterThresholds } from "@/components/panels/DataAssessmentPanel";

const info: TargetedPanelInfo = {
  n_genes: 313,
  technology: "xenium",
  panel_name: "Breast TME",
  sentences: ["Targeted panel of 313 genes: the run uses the panel's own settings unless you set them.", "No selection."],
  control: { median_pct: 0, share_above: 0.004, above_pct: 5, flag: null },
  defaults: { target_sum: 166, min_genes: 0, min_counts: 10, qc_counts_floor: 10 },
  all_genes: true,
};

describe("TargetedPanelNote", () => {
  it("says nothing without a panel", () => {
    const { container } = render(<TargetedPanelNote info={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("names the panel, the defaults and the control share", () => {
    render(<TargetedPanelNote info={info} />);
    expect(screen.getByTestId("targeted-panel-note")).toHaveTextContent("Targeted panel of 313 genes");
    expect(screen.getByText(/Breast TME/)).toBeInTheDocument();
    expect(screen.getByText(/0.4% of cells above 5%/)).toBeInTheDocument();
    expect(screen.getByText(/turns the selection back on/)).toBeInTheDocument();
  });

  it("points at a high control share", () => {
    render(<TargetedPanelNote info={{ ...info, control: { ...info.control!, flag: "Many cells are high." } }} />);
    expect(screen.getByText("Many cells are high.")).toBeInTheDocument();
  });
});

describe("filterThresholds on a panel", () => {
  it("takes the panel's defaults under what the user set, and sends a floor only when set", () => {
    const panel = filterThresholds(undefined, info.defaults);
    expect(panel.min_genes).toBe(0);
    expect(panel.min_counts).toBe(10);
    expect(panel.qc_counts_floor).toBeUndefined();
    const set = filterThresholds({ min_genes: 5, qc_counts_floor: 3 }, info.defaults);
    expect(set.min_genes).toBe(5);
    expect(set.qc_counts_floor).toBe(3);
    expect(filterThresholds(undefined).min_genes).toBe(200);
  });
});
