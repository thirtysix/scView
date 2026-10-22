import { describe, it, expect, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { useScoringMethodStore } from "@/stores/scoringMethodStore";
import { ScoringMethodChoice } from "./ScoringMethodChoice";

beforeEach(() => {
  useScoringMethodStore.setState({ method: "score_genes" });
});

describe("ScoringMethodChoice", () => {
  it("shows the shared choice and sets it for every scoring control", () => {
    render(<><ScoringMethodChoice /><ScoringMethodChoice label="with" /></>);
    const [a, b] = screen.getAllByRole("combobox", { name: "Gene-set scoring method" });
    expect(a).toHaveValue("score_genes");
    fireEvent.change(a!, { target: { value: "aucell" } });
    expect(useScoringMethodStore.getState().method).toBe("aucell");
    expect(b).toHaveValue("aucell");
    expect(screen.getByText("Score with")).toBeInTheDocument();
  });

  it("explains both methods in its tooltip", () => {
    render(<ScoringMethodChoice />);
    fireEvent.mouseEnter(screen.getByRole("button", { name: "How gene sets are scored" }));
    const tip = screen.getByRole("tooltip");
    expect(tip).toHaveTextContent("score_genes (scanpy): the set's mean log expression");
    expect(tip).toHaveTextContent("AUCell (Aibar et al. 2017): genes are ranked within each cell");
  });
});
