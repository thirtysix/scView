import { describe, it, expect, beforeEach } from "vitest";
import { useUnifiedViewStore } from "./unifiedViewStore";

describe("unifiedViewStore", () => {
  beforeEach(() => {
    useUnifiedViewStore.setState({
      scatterOverlay: null,
      overlayValues: null,
      overlayLabel: "",
      overlayDatasetId: null,
      violinData: {},
      violinTitle: "",
      activeSubtab: "markers",
    });
  });

  it("sets and reads an expression overlay", () => {
    useUnifiedViewStore.getState().setScatterOverlay({ type: "expression", gene: "CD8A" });
    expect(useUnifiedViewStore.getState().scatterOverlay).toEqual({ type: "expression", gene: "CD8A" });
  });

  it("clearOverlayState resets all overlay + violin fields", () => {
    const s = useUnifiedViewStore.getState();
    s.setScatterOverlay({ type: "score", name: "HALLMARK" });
    s.setOverlayValues(new Float32Array([1, 2, 3]));
    s.setOverlayLabel("HALLMARK");
    s.setViolinTitle("x expression");
    s.clearOverlayState();
    const after = useUnifiedViewStore.getState();
    expect(after.scatterOverlay).toBeNull();
    expect(after.overlayValues).toBeNull();
    expect(after.overlayLabel).toBe("");
    expect(after.overlayDatasetId).toBeNull();
    expect(after.violinData).toEqual({});
    expect(after.violinTitle).toBe("");
  });

  it("switches the active subtab (incl. the new DE tab)", () => {
    useUnifiedViewStore.getState().setActiveSubtab("de");
    expect(useUnifiedViewStore.getState().activeSubtab).toBe("de");
  });
});
