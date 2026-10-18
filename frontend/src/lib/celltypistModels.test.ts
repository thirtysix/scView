import { describe, it, expect } from "vitest";
import type { CellTypistModel } from "@/api/annotation";
import { defaultModel, orderModels, speciesMismatch } from "./celltypistModels";

const m = (model: string, species: CellTypistModel["species"]): CellTypistModel => ({ model, description: "", species });
const CATALOG = [
  m("Immune_All_Low.pkl", "human"),
  m("Adult_Mouse_Gut.pkl", "mouse"),
  m("Adult_Pig_Hippocampus.pkl", "other"),
  m("Healthy_Human_Liver.pkl", "human"),
  m("Mouse_Whole_Brain.pkl", "mouse"),
];

describe("CellTypist model choice", () => {
  it("groups by species, the dataset's first, keeping catalog order within each group", () => {
    expect(orderModels(CATALOG, "mouse").map((x) => x.model)).toEqual([
      "Adult_Mouse_Gut.pkl", "Mouse_Whole_Brain.pkl",
      "Immune_All_Low.pkl", "Healthy_Human_Liver.pkl", "Adult_Pig_Hippocampus.pkl",
    ]);
    // Unknown species: still one run per species, so each optgroup appears once.
    expect(orderModels(CATALOG, "unknown").map((x) => x.species)).toEqual(["human", "human", "mouse", "mouse", "other"]);
  });

  it("defaults to a model of the dataset's species", () => {
    expect(defaultModel(CATALOG, "human", "Immune_All_Low.pkl")).toBe("Immune_All_Low.pkl");
    expect(defaultModel(CATALOG, "mouse", "Immune_All_Low.pkl")).toBe("Adult_Mouse_Gut.pkl");
    expect(defaultModel(CATALOG, "unknown", "Immune_All_Low.pkl")).toBe("Immune_All_Low.pkl");
    expect(defaultModel([m("Immune_All_Low.pkl", "human")], "mouse", "Immune_All_Low.pkl")).toBe("Immune_All_Low.pkl");
  });

  it("warns only when the species is known and differs", () => {
    expect(speciesMismatch(CATALOG[0], "human")).toBe("");
    expect(speciesMismatch(CATALOG[0], "unknown")).toBe("");
    expect(speciesMismatch(undefined, "mouse")).toBe("");
    expect(speciesMismatch(CATALOG[0], "mouse")).toMatch(/looks mouse, but this model was trained on human cells/);
    expect(speciesMismatch(CATALOG[2], "human")).toMatch(/cells of another species/);
  });
});
