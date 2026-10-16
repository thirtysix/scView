import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { OpenInUnifiedView, SkippedSteps, SpeciesNote, StepCaution } from "./DataAssessmentPanel";
import { useViewStore } from "@/stores/viewStore";

describe("StepCaution", () => {
  it("shows a step's caution in full", () => {
    const text =
      "13,162 of 13,836 cells (95%) have both S and G2M scores within ±0.1 of 0: phase calls for these cells are not reliable.";
    render(<StepCaution caution={text} />);
    expect(screen.getByText(text)).toBeTruthy();
  });

  it("renders nothing without one (or from an older backend)", () => {
    const { container } = render(
      <>
        <StepCaution caution={null} />
        <StepCaution />
      </>,
    );
    expect(container.textContent).toBe("");
  });
});

describe("SkippedSteps", () => {
  it("lists a skipped step by its label, with the reason", () => {
    render(<SkippedSteps notes={{ dissociation_stress: "Only 5 of the 22 genes are in this dataset." }} />);
    expect(screen.getByText(/Dissociation Stress: Only 5 of the 22 genes/)).toBeTruthy();
  });

  it("renders nothing when no step was skipped", () => {
    const { container } = render(<SkippedSteps notes={{}} />);
    expect(container.textContent).toBe("");
  });
});

describe("SpeciesNote", () => {
  it("names the species and the evidence for it", () => {
    render(
      <SpeciesNote
        call={{ species: "mouse", method: "symbol_case", evidence: "Gene symbols are title case." }}
      />,
    );
    expect(screen.getByText("mouse")).toBeTruthy();
    expect(screen.getByText(/Gene symbols are title case/)).toBeTruthy();
  });

  it("renders nothing for an older backend without species", () => {
    const { container } = render(<SpeciesNote call={undefined} />);
    expect(container.textContent).toBe("");
  });
});

describe("OpenInUnifiedView", () => {
  beforeEach(() => useViewStore.setState({ activePanel: "assessment" }));

  it("opens Unified View only when clicked", () => {
    render(<OpenInUnifiedView />);
    expect(useViewStore.getState().activePanel).toBe("assessment");
    fireEvent.click(screen.getByRole("button", { name: /open in unified view/i }));
    expect(useViewStore.getState().activePanel).toBe("unified");
  });
});
