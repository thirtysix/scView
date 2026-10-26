/**
 * DSB helpers for the Protein tab: which layer a normalisation reads, the empty
 * droplets a window takes (from the binned density the backend sends), and the
 * one-line note on how DSB values were made.
 */
import type { ProteinInfo } from "@/api/types";
import type { DsbDensity } from "@/api/dsb";

export type ProteinNorm = "clr" | "dsb";

/** The layer a normalisation reads: DSB only once computed, else CLR. */
export function proteinLayerFor(protein: ProteinInfo | null | undefined, norm: ProteinNorm): string | undefined {
  if (!protein) return undefined;
  return norm === "dsb" && protein.dsb_layer ? protein.dsb_layer : protein.layer;
}

/** "DSB" or "CLR": the short name of the layer's normalisation, for labels. */
export function normLabel(layer: string | undefined | null): string {
  return layer === "protein_dsb" ? "DSB" : "CLR";
}

/** Droplets with lo <= log10 UMIs < hi for RNA and protein (the backend's rule, on its bins). */
export function dropletsInWindow(
  density: DsbDensity,
  rnaWindow: [number, number],
  protWindow: [number, number],
): number {
  const w = density.bin_width;
  const inside = (b: number, [lo, hi]: [number, number]) => b * w >= lo - 1e-9 && (b + 1) * w <= hi + 1e-9;
  let n = 0;
  for (let i = 0; i < density.n.length; i++) {
    if (inside(density.rna_bin[i]!, rnaWindow) && inside(density.prot_bin[i]!, protWindow)) n += density.n[i]!;
  }
  return n;
}

/** How the dataset's DSB values were made, in one sentence. */
export function dsbNote(dsb: NonNullable<ProteinInfo["dsb"]>): string {
  const background =
    dsb.mode === "empty_droplets"
      ? `against ${dsb.n_droplets.toLocaleString()} empty droplets`
      : "against a background estimated from the cells (no empty droplets)";
  const technical = !dsb.denoise
    ? "no per-cell noise removal"
    : dsb.isotypes.length
      ? `per-cell noise removed with isotypes ${dsb.isotypes.join(", ")}`
      : "per-cell noise removed with the background mean alone (no isotype controls)";
  return `DSB ${background}; ${technical}.`;
}
