import type { CellTypistModel } from "@/api/annotation";

export type DatasetSpecies = "human" | "mouse" | "unknown";

function known(species: DatasetSpecies | undefined): species is "human" | "mouse" {
  return species === "human" || species === "mouse";
}

/** The catalog with the dataset's species first, each part in catalog order. */
export function orderModels(models: CellTypistModel[], species: DatasetSpecies | undefined): CellTypistModel[] {
  if (!known(species)) return models;
  return [...models.filter((m) => m.species === species), ...models.filter((m) => m.species !== species)];
}

/** The model to start from: the server's default when it suits the dataset's
 *  species (or the species is unknown), else the first model of that species. */
export function defaultModel(
  models: CellTypistModel[],
  species: DatasetSpecies | undefined,
  serverDefault: string,
): string {
  if (!known(species)) return serverDefault;
  const def = models.find((m) => m.model === serverDefault);
  if (def?.species === species) return serverDefault;
  return models.find((m) => m.species === species)?.model ?? serverDefault;
}

/** A warning when the picked model's species is not the dataset's ("" when it
 *  is, or when the dataset's species is unknown). */
export function speciesMismatch(model: CellTypistModel | undefined, species: DatasetSpecies | undefined): string {
  if (!model || !known(species) || model.species === species) return "";
  const cells = model.species === "other" ? "cells of another species" : `${model.species} cells`;
  return (
    `This dataset looks ${species}, but this model was trained on ${cells}. Gene symbols are ` +
    "matched exactly (human CD3E, mouse Cd3e), so few genes will be shared and the labels are unreliable."
  );
}
