import { describe, it, expect, beforeEach } from "vitest";
import { useScoringMethodStore } from "./scoringMethodStore";

const KEY = "scview.scoringMethod.v1";

beforeEach(() => {
  localStorage.clear();
  useScoringMethodStore.setState({ method: "score_genes" });
});

describe("scoring method store", () => {
  it("defaults to score_genes and remembers the choice in localStorage", () => {
    expect(useScoringMethodStore.getState().method).toBe("score_genes");
    useScoringMethodStore.getState().setMethod("aucell");
    expect(JSON.parse(localStorage.getItem(KEY)!).state).toEqual({ method: "aucell" });
  });

  it("reads a stored choice back, and ignores one it does not know", async () => {
    localStorage.setItem(KEY, JSON.stringify({ state: { method: "aucell" }, version: 0 }));
    await useScoringMethodStore.persist.rehydrate();
    expect(useScoringMethodStore.getState().method).toBe("aucell");
    localStorage.setItem(KEY, JSON.stringify({ state: { method: "gsva" }, version: 0 }));
    await useScoringMethodStore.persist.rehydrate();
    expect(useScoringMethodStore.getState().method).toBe("gsva");
    localStorage.setItem(KEY, JSON.stringify({ state: { method: "zscore" }, version: 0 }));
    useScoringMethodStore.setState({ method: "score_genes" });
    await useScoringMethodStore.persist.rehydrate();
    expect(useScoringMethodStore.getState().method).toBe("score_genes");
  });
});
