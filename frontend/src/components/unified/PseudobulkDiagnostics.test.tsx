import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { PseudobulkHetero, PseudobulkRuv } from "@/api/de";

vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { name?: string }[] }) => (
    <div data-testid="plot">{data.map((t) => t.name ?? "").join(",")}</div>
  ),
}));

import { PseudobulkDiagnostics, PseudobulkModelRows } from "./PseudobulkDiagnostics";

const HETERO: PseudobulkHetero = {
  groups: [
    { group: "STIM", n_samples: 8, median_variance: 0.4, trend: [[1, 1.2], [5, 0.6]] },
    { group: "CTRL", n_samples: 8, median_variance: 0.1, trend: [[1, 1.0], [5, 0.4]] },
  ],
  ratio: 4,
  higher: "STIM",
  flag: true,
  threshold: 2,
  message: "STIM samples vary 4.0x as much as CTRL samples. Try voomByGroup.",
};

const RUV: PseudobulkRuv = {
  k: 1,
  n_controls: 1000,
  control_source: "the 1,000 genes least associated with the condition in a first limma-voom pass",
  factors: [{ name: "W1", r_condition: 0.62, r_log_depth: -0.1, r_n_cells: null, r2_donor: 0 }],
  samples: [
    { donor: "d1", group: "STIM", n_cells: 120, log10_depth: 5.1, W: [0.3] },
    { donor: "d1", group: "CTRL", n_cells: 100, log10_depth: 5.0, W: [-0.3] },
  ],
  warning: "W1 follows the condition (r = 0.62): RUV may remove real signal.",
};

describe("PseudobulkDiagnostics", () => {
  it("shows the flagged variance check with one trend per group", () => {
    render(<PseudobulkDiagnostics hetero={HETERO} />);
    expect(screen.getByText(/vary 4.0x as much/)).toBeTruthy();
    expect(screen.getByTestId("plot").textContent).toBe("STIM (8),CTRL (8)");
    expect(screen.getByText(/n = 8 STIM samples; 8 CTRL samples/)).toBeTruthy();
    expect(screen.queryByTestId("pb-ruv")).toBeNull();
  });

  it("tabulates the RUV factors against the covariates, with the warning", () => {
    render(<PseudobulkDiagnostics ruv={RUV} />);
    expect(screen.getByText(/W1 follows the condition/)).toBeTruthy();
    expect(screen.getByText("0.62")).toBeTruthy();
    expect(screen.getAllByText("n/a").length).toBe(1);
    expect(screen.getByText(/n = 2 samples; 1,000 control genes/)).toBeTruthy();
    expect(screen.getByText("Factor values per sample")).toBeTruthy();
  });

  it("renders nothing without a result", () => {
    const { container } = render(<PseudobulkDiagnostics />);
    expect(container.innerHTML).toBe("");
  });
});

describe("PseudobulkModelRows", () => {
  it("offers the three models and RUV k 0-3 as labelled selects", () => {
    const onChange = vi.fn();
    render(<PseudobulkModelRows pbMethod="deseq2" ruvK={0} onChange={onChange} selectCls="" />);
    const model = screen.getByLabelText("Model") as HTMLSelectElement;
    expect(Array.from(model.options).map((o) => o.text)).toEqual(["DESeq2", "limma-voom", "voomByGroup"]);
    fireEvent.change(model, { target: { value: "voombygroup" } });
    expect(onChange).toHaveBeenCalledWith({ pbMethod: "voombygroup" });
    const ruv = screen.getByLabelText("RUV") as HTMLSelectElement;
    expect(ruv.options.length).toBe(4);
    fireEvent.change(ruv, { target: { value: "2" } });
    expect(onChange).toHaveBeenCalledWith({ ruvK: 2 });
  });
});
