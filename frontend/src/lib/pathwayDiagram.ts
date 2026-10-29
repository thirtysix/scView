/**
 * Pure helpers of the Reactome pathway diagram (components/unified/PathwayDiagram):
 * which enrichment rows have a diagram, label wrapping, node outlines per class,
 * and the log fold change colour scale (lib/colorScale.ts, always centred on 0).
 */
import type { DiagramNode, DiagramShape, PathwayValues } from "@/api/pathways";
import { colormapColor, type DivergingColormapName } from "@/lib/colors";
import { overlayStats, resolveColorScale, type ClipSetting, type ResolvedColorScale } from "@/lib/colorScale";
import { LIGHT_SLATE, DARK_SLATE, DARK_SURFACE, type ResolvedTheme } from "@/lib/theme";

const REACTOME_ID = /^R-(HSA|MMU)-\d+$/;

/** Where an enrichment row's gene set came from, by MSigDB's exactSource. */
export type SourceKind = "reactome" | "kegg" | "wikipathways" | "other" | null;

export function sourceKind(sourceId: string | undefined | null): SourceKind {
  if (!sourceId) return null;
  if (REACTOME_ID.test(sourceId)) return "reactome";
  if (/^WP\d+$/.test(sourceId)) return "wikipathways";
  if (/^(N|hsa|mmu|map)\d+$/.test(sourceId)) return "kegg";
  return "other";
}

export const isReactomeId = (sourceId: string | undefined | null): boolean => sourceKind(sourceId) === "reactome";

/** "Open in KEGG" for a row's outbound link (null: no link to show). */
export function outboundLabel(sourceId: string | undefined | null, url: string | undefined | null): string | null {
  if (!url || !/^https?:\/\//.test(url)) return null;
  const kind = sourceKind(sourceId);
  if (kind === "kegg") return "Open in KEGG";
  if (kind === "wikipathways") return "Open in WikiPathways";
  if (kind === "reactome") return "Open in Reactome";
  return "Open the source";
}

export const LABEL_FONT = 9;
const CHAR_W = 0.56; // average glyph width / font size (sans-serif)

/** `label` wrapped into lines that fit `width` x `height` at `font` px; the last line gets "…" when cut. */
export function wrapLabel(label: string, width: number, height: number, font = LABEL_FONT): string[] {
  const perLine = Math.max(3, Math.floor((width - 4) / (font * CHAR_W)));
  const maxLines = Math.max(1, Math.floor((height - 2) / (font * 1.15)));
  const words = label.split(/(?<=[\s:/,-])/).map((w) => w.trimEnd()).filter(Boolean);
  const lines: string[] = [];
  let cur = "";
  const push = (s: string) => lines.push(s);
  for (let w of words) {
    while (w.length > perLine) {
      // a word longer than a line: break it
      if (cur) {
        push(cur);
        cur = "";
      }
      push(w.slice(0, perLine));
      w = w.slice(perLine);
    }
    const sep = cur && !/[:/,-]$/.test(cur) ? " " : "";
    if ((cur + sep + w).length <= perLine) cur = cur + sep + w;
    else {
      if (cur) push(cur);
      cur = w;
    }
  }
  if (cur) push(cur);
  if (lines.length <= maxLines) return lines;
  const kept = lines.slice(0, maxLines);
  const last = kept[maxLines - 1]!;
  kept[maxLines - 1] = (last.length >= perLine ? last.slice(0, perLine - 1) : last) + "…";
  return kept;
}

/** An octagon (Reactome's complex) as an SVG path. */
export function octagonPath(x: number, y: number, w: number, h: number, cut = 6): string {
  const c = Math.min(cut, w / 4, h / 4);
  return `M${x + c},${y}H${x + w - c}L${x + w},${y + c}V${y + h - c}L${x + w - c},${y + h}H${x + c}L${x},${y + h - c}V${y + c}Z`;
}

/** The SVG outline element of each node class: what the renderer draws. */
export function nodeShapeKind(node: DiagramNode): "rounded" | "octagon" | "double" | "ellipse" | "gene" | "rna" | "rect" | "process" | "encapsulated" {
  switch (node.class) {
    case "Protein": return "rounded";
    case "Complex": return "octagon";
    case "EntitySet": return "double";
    case "Chemical": return "ellipse";
    case "Gene": return "gene";
    case "RNA": return "rna";
    case "ProcessNode": return "process";
    case "EncapsulatedNode": return "encapsulated";
    default: return "rect";
  }
}

export interface DiagramPalette {
  surface: string;
  ink: string;
  muted: string;
  line: string;
  nodeStroke: string;
  /** a node with genes none of which the DE result measured */
  notMeasured: string;
  /** a node without genes (a small molecule) */
  noGenes: string;
  compartmentFill: string;
  compartmentStroke: string;
  pathwayFill: string;
  highlight: string;
}

/** Explicit colours for the SVG (attributes, not classes, so a download carries them). Downloads use light. */
export function diagramPalette(theme: ResolvedTheme): DiagramPalette {
  const s = theme === "dark" ? DARK_SLATE : LIGHT_SLATE;
  return {
    surface: theme === "dark" ? DARK_SURFACE : "#ffffff",
    ink: s[800]!,
    muted: s[500]!,
    line: s[500]!,
    nodeStroke: s[600]!,
    notMeasured: theme === "dark" ? "#4b5563" : "#d1d5db",
    noGenes: theme === "dark" ? DARK_SURFACE : "#ffffff",
    compartmentFill: theme === "dark" ? "rgba(217,170,90,0.07)" : "rgba(250,236,210,0.45)",
    compartmentStroke: theme === "dark" ? "#a37d3c" : "#d4a95f",
    pathwayFill: theme === "dark" ? "#22324a" : "#e8f0fb",
    highlight: theme === "dark" ? "#60a5fa" : "#2563eb",
  };
}

/** The colour scale of the log fold changes: diverging, centred on 0, symmetric (lib/colorScale.ts rules,
 *  the user's diverging map and clipping). The values are mirrored so a one-signed result still gets 0 in
 *  the middle: a fold change is never "low" and "high", only down and up. */
export function lfcScale(
  values: PathwayValues | null,
  diverging: DivergingColormapName,
  clip: ClipSetting,
): ResolvedColorScale | null {
  if (!values) return null;
  const v = Object.values(values.nodes)
    .map((n) => n.value)
    .filter((x): x is number => x !== null && Number.isFinite(x));
  if (v.length === 0) return null;
  const stats = overlayStats([...v, ...v.map((x) => -x)]);
  const scale = resolveColorScale("score", stats, {
    sequential: "viridis",
    diverging,
    signedMode: "diverging",
    clip,
  });
  if (!(scale.max > scale.min)) return { ...scale, min: -1, max: 1, centre: 0, diverging: true, colormap: diverging };
  return { ...scale, n: v.length };
}

export function rgb([r, g, b]: [number, number, number]): string {
  return `rgb(${r},${g},${b})`;
}

/** The fill of a value on the scale. */
export function valueColour(value: number, scale: ResolvedColorScale): [number, number, number] {
  const t = (value - scale.min) / (scale.max - scale.min);
  return colormapColor(scale.colormap, t);
}

/** Black or white text on a fill. */
export function inkOn([r, g, b]: [number, number, number]): string {
  return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? "#111827" : "#ffffff";
}

/** "3 of 5 genes changed (adjusted p < 0.05)", "2 of 5 genes measured", ... for a tooltip. */
export function nodeSummary(nv: PathwayValues["nodes"][string] | undefined, cutoff: number): string {
  if (!nv) return "";
  if (nv.n_measured === 0) return `None of its ${nv.n_genes} gene${nv.n_genes === 1 ? "" : "s"} measured`;
  const measured = `${nv.n_measured} of ${nv.n_genes} gene${nv.n_genes === 1 ? "" : "s"} measured`;
  if (nv.n_changed === null) return `${measured}; no adjusted p-values`;
  return `${nv.n_changed} of ${nv.n_measured} changed (adjusted p < ${cutoff}); ${measured}`;
}

/** Points "x,y x,y" of a triangle end shape. */
export function trianglePoints(s: DiagramShape): string {
  return [s.a, s.b, s.c].filter(Boolean).map((p) => `${p![0]},${p![1]}`).join(" ");
}

/** The bounds with a margin, as a viewBox. */
export function fitViewBox(b: { minX: number; maxX: number; minY: number; maxY: number }, pad = 20) {
  const w = Math.max(b.maxX - b.minX, 10);
  const h = Math.max(b.maxY - b.minY, 10);
  return { x: b.minX - pad, y: b.minY - pad, w: w + pad * 2, h: h + pad * 2 };
}

/** Zoom a viewBox by `factor` (< 1 zooms in) about the point (px, py) in diagram units. */
export function zoomViewBox(
  vb: { x: number; y: number; w: number; h: number },
  factor: number,
  px: number,
  py: number,
  limits: { min: number; max: number },
) {
  const w = Math.min(Math.max(vb.w * factor, limits.min), limits.max);
  const f = w / vb.w;
  return { x: px - (px - vb.x) * f, y: py - (py - vb.y) * f, w, h: vb.h * f };
}
