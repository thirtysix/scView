import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { OpenInUnifiedView, SpeciesNote } from "./DataAssessmentPanel";
import { useViewStore } from "@/stores/viewStore";

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
