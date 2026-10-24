import { describe, it, expect } from "vitest";
import type { AmbientResponse } from "@/api/ambient";
import { correctionText, flagIndex, flagTitle, isCorrectionSamples, pct, rhoText } from "./ambient";
import { paramText } from "./provenanceParams";

describe("ambient RNA words", () => {
  it("percents and rho with its range", () => {
    expect(pct(0.053)).toBe("5.3%");
    expect(pct(0.0042)).toBe("0.42%");
    expect(pct(null)).toBe("—");
    expect(rhoText({ rho: 0.191, rho_low: 0.138, rho_high: 0.253 })).toBe("19.1% (13.8-25.3%)");
    expect(rhoText({ rho: 0.05, rho_low: null, rho_high: null })).toBe("5.0%");
  });

  it("indexes the flags by group and gene, and explains each", () => {
    const resp = {
      markers: { groups: { "0": [{ gene: "LYZ", soup_rank: 2, ratio: 1.2 }] }, approximate: false },
    } as unknown as AmbientResponse;
    const idx = flagIndex(resp);
    const lyz = idx.get("0")?.get("LYZ");
    expect(lyz?.ratio).toBe(1.2);
    expect(idx.get("1")).toBeUndefined();
    expect(flagIndex(null).size).toBe(0);
    expect(flagIndex({} as AmbientResponse).size).toBe(0);
    expect(flagTitle(lyz!, "0", false)).toContain("gene 2 of the soup");
    expect(flagTitle(lyz!, "0", false)).toContain("1.2x what ambient RNA alone");
    expect(flagTitle({ gene: "LYZ", soup_rank: 3, ratio: null }, "0", true)).toContain("Possibly ambient");
  });

  it("the correction step's History chips", () => {
    const rows = [
      { sample: "s1", rho: 0.053, rho_source: "estimated" as const, removed_share: 0.051 },
      { sample: "s2", rho: 0.1, rho_source: "set" as const },
    ];
    expect(isCorrectionSamples("samples", rows)).toBe(true);
    expect(isCorrectionSamples("samples", ["a"])).toBe(false);
    expect(isCorrectionSamples("other", rows)).toBe(false);
    expect(correctionText(rows)).toBe("ambient RNA: s1: rho 5.3% (estimated, 5.1% removed); s2: rho 10.0% (set)");
    expect(paramText("samples", rows)).toBe(correctionText(rows));
    expect(paramText("ambient_rho", null)).toBe("rho: each sample's own estimate");
    expect(paramText("ambient_rho", 0.1)).toBe("rho set: 0.1");
    expect(paramText("removed_share", 0.0512)).toBe("5.1% of the UMIs removed as ambient");
  });
});
