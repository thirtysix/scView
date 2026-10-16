import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { OpenInUnifiedView } from "./DataAssessmentPanel";
import { useViewStore } from "@/stores/viewStore";

describe("OpenInUnifiedView", () => {
  beforeEach(() => useViewStore.setState({ activePanel: "assessment" }));

  it("opens Unified View only when clicked", () => {
    render(<OpenInUnifiedView />);
    expect(useViewStore.getState().activePanel).toBe("assessment");
    fireEvent.click(screen.getByRole("button", { name: /open in unified view/i }));
    expect(useViewStore.getState().activePanel).toBe("unified");
  });
});
