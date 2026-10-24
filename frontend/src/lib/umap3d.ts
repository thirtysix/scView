/**
 * The 3D UMAP: the pipeline's `umap_3d` (backend core/pipeline.py). Automatic makes it in
 * memory and leaves it out in large-dataset mode, where it adds about half the UMAP
 * step's time (25 min at 1.2 million cells, measured on Roihu 2026-10-24).
 */

export type Umap3dSetting = "auto" | "on" | "off";

/** The UMAP step's select: value -> label. */
export const UMAP_3D_LABELS: Record<string, string> = {
  auto: "Automatic (off for large datasets)",
  on: "On",
  off: "Off",
};

/** A recorded `umap_3d` in words, for the History chip. */
export function umap3dText(value: unknown): string {
  if (value === "on") return "3D UMAP: made";
  if (value === "off") return "3D UMAP: left out";
  return `umap_3d=${String(value)}`;
}
