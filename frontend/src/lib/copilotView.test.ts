import { describe, expect, it } from "vitest";
import { screenContext } from "@/lib/copilotView";

const de = { id: "r1", label: "CD14 Mono: STIM vs CTRL (Wilcoxon)" };
const enrichment = { column: "cluster", group: "B" };

describe("screenContext", () => {
  it("names the DE result only while the DE tab is open", () => {
    const slots = { de, enrichment: null, compareTab: null };
    expect(screenContext("unified", "de", slots)).toEqual({
      subtab: "de",
      on_screen: { kind: "de", id: "r1", label: de.label },
    });
    // a DE result left behind is not "this result" on the Markers tab
    expect(screenContext("unified", "markers", slots)).toEqual({ subtab: "markers" });
  });

  it("names the enrichment group on the Enrichment tab", () => {
    const slots = { de: null, enrichment, compareTab: null };
    expect(screenContext("unified", "enrichment", slots).on_screen).toEqual({
      kind: "enrichment",
      column: "cluster",
      group: "B",
    });
    expect(screenContext("unified", "de", slots)).toEqual({ subtab: "de" });
  });

  it("names the open Compare tab, and nothing on other panels", () => {
    const slots = { de, enrichment, compareTab: "composition" };
    expect(screenContext("compare", "de", slots)).toEqual({
      on_screen: { kind: "compare", tab: "composition" },
    });
    expect(screenContext("assessment", "de", slots)).toEqual({});
  });
});
