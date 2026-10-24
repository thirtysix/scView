import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { RegressOutChoice } from "./RegressOutChoice";
import { regressOptions } from "@/lib/regressOut";

const options = regressOptions([]);

describe("RegressOutChoice", () => {
  it("is off by default and adds or removes a covariate per checkbox", () => {
    const onChange = vi.fn();
    const { rerender } = render(<RegressOutChoice options={options} value={[]} onChange={onChange} />);
    expect(screen.getByText("None: off (the default).")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "S − G2M (cell-cycle difference)" }));
    expect(onChange).toHaveBeenLastCalledWith(["cc_difference"]);
    rerender(<RegressOutChoice options={options} value={["cc_difference", "total_counts"]} onChange={onChange} />);
    expect(screen.getByText(/2 covariates/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "Total counts" }));
    expect(onChange).toHaveBeenLastCalledWith(["cc_difference"]);
  });

  it("says when it helps and when it hurts", () => {
    render(<RegressOutChoice options={options} value={[]} onChange={() => {}} />);
    fireEvent.mouseEnter(screen.getByRole("button", { name: "When regress-out helps" }));
    const tip = screen.getByRole("tooltip");
    expect(tip).toHaveTextContent("removes real biology in proliferating tissue");
    expect(tip).toHaveTextContent("usually redundant after normalisation");
    expect(tip).toHaveTextContent("#526");
  });

  it("is disabled with Pearson residuals, and says why", () => {
    render(<RegressOutChoice options={options} value={[]} onChange={() => {}} pearson />);
    expect(screen.getByRole("checkbox", { name: "Total counts" })).toBeDisabled();
    expect(screen.getByText(/Off with Pearson residuals/)).toBeInTheDocument();
  });
});
