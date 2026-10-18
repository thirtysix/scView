import type { CoexpressedGene } from "@/api/coexpression";

export type CoexpressedSort = "gene" | "r" | "n";

/** Genes of a co-expression list in table order: by |r| (strongest first by
 *  default), by name, or by expressing cells; ties by name. */
export function sortCoexpressed(
  genes: CoexpressedGene[],
  field: CoexpressedSort,
  desc: boolean,
): CoexpressedGene[] {
  const sign = desc ? -1 : 1;
  return [...genes].sort((a, b) => {
    if (field === "gene") return sign * a.gene.localeCompare(b.gene);
    if (field === "n") return sign * (a.n_expressing - b.n_expressing) || a.gene.localeCompare(b.gene);
    return sign * (Math.abs(a.r) - Math.abs(b.r)) || a.gene.localeCompare(b.gene);
  });
}
