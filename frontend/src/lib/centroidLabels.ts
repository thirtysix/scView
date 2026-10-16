/**
 * Group labels drawn on the embedding: each group's name at its centroid (the
 * median x and y of its drawn cells, so a few outlying cells do not pull the
 * label off the group), as a deck.gl TextLayer with a halo that reads on light
 * and dark points alike.
 */
import { TextLayer } from "@deck.gl/layers";

/** Labels are on by default for a column with at most this many groups. */
export const LABELS_AUTO_MAX_GROUPS = 40;
const MAX_LABEL = 32;

/** The user's explicit choice for one colouring (its key); else the default applies. */
export interface LabelChoice {
  key: string;
  show: boolean;
}

/** The label-choice key of a compare-colourings set (one choice for all its panels). */
export const compareLabelKey = (columns: readonly string[]) => `compare:${columns.join("\u0000")}`;

/** Whether a colouring with `nGroups` groups shows its labels. */
export function labelsShown(choice: LabelChoice | null, key: string, nGroups: number): boolean {
  if (choice && choice.key === key) return choice.show;
  return nGroups > 0 && nGroups <= LABELS_AUTO_MAX_GROUPS;
}

export interface GroupCentroid {
  /** The group's code (index in `labels`) */
  code: number;
  label: string;
  /** Median position: [x, y] or [x, y, z] */
  position: number[];
  /** Drawn cells in the group */
  count: number;
}

function medianOf(values: Float32Array): number {
  values.sort();
  const n = values.length;
  const mid = n >> 1;
  return n % 2 ? values[mid]! : (values[mid - 1]! + values[mid]!) / 2;
}

/**
 * The centroid of every group with at least one drawn cell, in the column's
 * order. `codes` holds each cell's group (-1 = none); `indices` limits the
 * cells to those drawn (a sample, or one split panel's), null = all cells.
 * Cells with non-finite coordinates are skipped.
 */
export function groupCentroids(
  positions: Float32Array,
  stride: 2 | 3,
  codes: ArrayLike<number>,
  labels: readonly string[],
  indices: ArrayLike<number> | null = null,
): GroupCentroid[] {
  const k = labels.length;
  const n = indices ? indices.length : Math.min(codes.length, positions.length / stride);
  const cellAt = (s: number) => (indices ? indices[s]! : s);
  const ok = (i: number) => {
    const c = codes[i]!;
    if (!(c >= 0 && c < k) || !Number.isInteger(c)) return false;
    for (let d = 0; d < stride; d++) if (!Number.isFinite(positions[i * stride + d]!)) return false;
    return true;
  };
  const counts = new Uint32Array(k);
  for (let s = 0; s < n; s++) {
    const i = cellAt(s);
    if (ok(i)) counts[codes[i]!] = counts[codes[i]!]! + 1;
  }
  const coords = Array.from({ length: k }, (_, c) =>
    Array.from({ length: stride }, () => new Float32Array(counts[c]!)),
  );
  const fill = new Uint32Array(k);
  for (let s = 0; s < n; s++) {
    const i = cellAt(s);
    if (!ok(i)) continue;
    const c = codes[i]!;
    for (let d = 0; d < stride; d++) coords[c]![d]![fill[c]!] = positions[i * stride + d]!;
    fill[c] = fill[c]! + 1;
  }
  const out: GroupCentroid[] = [];
  for (let c = 0; c < k; c++) {
    if (counts[c] === 0) continue;
    out.push({ code: c, label: labels[c]!, position: coords[c]!.map(medianOf), count: counts[c]! });
  }
  return out;
}

const truncate = (s: string) => (s.length > MAX_LABEL ? `${s.slice(0, MAX_LABEL - 1)}…` : s);

/**
 * TextLayer(s) for the labels. With a highlighted group, the others are drawn
 * faded in a second layer (a layer's opacity fades the text and its halo
 * together, which a per-label colour cannot do with an SDF outline).
 */
export function centroidLabelLayers(
  id: string,
  centroids: readonly GroupCentroid[],
  opts: { dark: boolean; highlight?: string | null },
): TextLayer<GroupCentroid>[] {
  const { dark, highlight = null } = opts;
  const hasHighlight = highlight != null && centroids.some((c) => c.label === highlight);
  const make = (layerId: string, data: readonly GroupCentroid[], opacity: number) =>
    new TextLayer<GroupCentroid>({
      id: layerId,
      data,
      getPosition: (d) => d.position as [number, number] | [number, number, number],
      getText: (d) => truncate(d.label),
      getSize: 12,
      sizeUnits: "pixels",
      getColor: dark ? [248, 250, 252, 255] : [15, 23, 42, 255],
      getTextAnchor: "middle",
      getAlignmentBaseline: "center",
      fontFamily: "Inter, system-ui, sans-serif",
      fontWeight: 600,
      characterSet: "auto",
      // A wide SDF radius and buffer leave room for a ~2 px halo at 12 px.
      fontSettings: { sdf: true, fontSize: 64, buffer: 12, radius: 16, smoothing: 0.15 },
      outlineWidth: 12,
      outlineColor: dark ? [15, 23, 42, 235] : [255, 255, 255, 235],
      opacity,
      pickable: false,
      // Over the points, also in 3D where nearer points would hide a label.
      parameters: { depthCompare: "always", depthWriteEnabled: false },
    });
  if (!hasHighlight) return [make(id, centroids, 1)];
  return [
    make(id, centroids.filter((c) => c.label !== highlight), 0.3),
    make(`${id}-hl`, centroids.filter((c) => c.label === highlight), 1),
  ];
}
