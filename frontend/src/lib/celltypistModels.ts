import type { CellTypistModel } from "@/api/annotation";

export type DatasetSpecies = "human" | "mouse" | "unknown";

function known(species: DatasetSpecies | undefined): species is "human" | "mouse" {
  return species === "human" || species === "mouse";
}

const SPECIES_ORDER: CellTypistModel["species"][] = ["human", "mouse", "other"];

/** The catalog grouped by species, the dataset's species first, each group in
 *  catalog order (the picker draws one optgroup per run of a species, and the
 *  catalog interleaves them). */
export function orderModels(models: CellTypistModel[], species: DatasetSpecies | undefined): CellTypistModel[] {
  const order = known(species) ? [species, ...SPECIES_ORDER.filter((s) => s !== species)] : SPECIES_ORDER;
  const rank = (s: CellTypistModel["species"]) => {
    const i = order.indexOf(s);
    return i < 0 ? order.length : i;
  };
  return models
    .map((m, i) => ({ m, i }))
    .sort((a, b) => rank(a.m.species) - rank(b.m.species) || a.i - b.i)
    .map(({ m }) => m);
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
