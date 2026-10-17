/**
 * The PAGA graph drawn over the embedding (Trajectory panel): one node per
 * group at the median of its cells, sized by its cell count, and an edge
 * between two groups whose PAGA connectivity reaches the threshold, wider and
 * darker the stronger it is. A branch (the groups on PAGA's spanning tree from
 * the pseudotime root to a chosen end group) is drawn in amber.
 */
import { LineLayer, ScatterplotLayer } from "@deck.gl/layers";
import type { Layer } from "@deck.gl/core";
import type { PagaRecord } from "@/api/trajectory";
import { centroidLabelLayers } from "@/lib/centroidLabels";

/** Edges at or above this connectivity are drawn until the user moves the slider. */
export const DEFAULT_PAGA_THRESHOLD = 0.1;
const NODE_MIN_PX = 5;
const NODE_MAX_PX = 20;
const AMBER: [number, number, number] = [217, 119, 6];

export interface PagaEdge {
  i: number;
  j: number;
  weight: number;
  /** On PAGA's maximum spanning tree */
  tree: boolean;
}

const pairKey = (i: number, j: number) => (i < j ? `${i}-${j}` : `${j}-${i}`);

/** The edges to draw: connectivity >= `threshold`, and only spanning-tree
 *  edges when `treeOnly`. Strongest first. */
export function visibleEdges(record: PagaRecord, threshold: number, treeOnly = false): PagaEdge[] {
  const tree = new Set(record.tree.map(([i, j]) => pairKey(i, j)));
  return record.edges
    .map(([i, j, weight]) => ({ i, j, weight, tree: tree.has(pairKey(i, j)) }))
    .filter((e) => e.weight >= threshold && (!treeOnly || e.tree))
    .sort((a, b) => b.weight - a.weight);
}

/** Node radius (px): its area grows with the group's cells. */
export function nodeRadius(size: number, maxSize: number): number {
  if (maxSize <= 0) return NODE_MIN_PX;
  return NODE_MIN_PX + (NODE_MAX_PX - NODE_MIN_PX) * Math.sqrt(Math.max(0, size) / maxSize);
}

function treeNeighbours(record: PagaRecord): number[][] {
  const adj: number[][] = record.groups.map(() => []);
  for (const [i, j] of record.tree) {
    adj[i]?.push(j);
    adj[j]?.push(i);
  }
  return adj;
}

/** The groups along PAGA's spanning tree from `from` to `to` (both included),
 *  or null when either is unknown or the tree does not join them. */
export function treePath(record: PagaRecord, from: string, to: string): string[] | null {
  const start = record.groups.indexOf(from);
  const end = record.groups.indexOf(to);
  if (start < 0 || end < 0) return null;
  const adj = treeNeighbours(record);
  const prev = new Map<number, number>([[start, -1]]);
  const queue = [start];
  while (queue.length) {
    const u = queue.shift()!;
    if (u === end) break;
    for (const v of adj[u] ?? []) {
      if (!prev.has(v)) {
        prev.set(v, u);
        queue.push(v);
      }
    }
  }
  if (!prev.has(end)) return null;
  const path: string[] = [];
  for (let u = end; u !== -1; u = prev.get(u)!) path.push(record.groups[u]!);
  return path.reverse();
}

/** Where a branch from `from` can end: every other group the tree reaches, in
 *  the column's order. */
export function branchEnds(record: PagaRecord, from: string): string[] {
  return record.groups.filter((g) => g !== from && treePath(record, from, g) !== null);
}

interface PagaNode {
  i: number;
  group: string;
  position: number[];
  size: number;
  radius: number;
  onPath: boolean;
}

interface DrawnEdge extends PagaEdge {
  from: number[];
  to: number[];
  onPath: boolean;
}

export interface PagaLayerOptions {
  record: PagaRecord;
  /** Each group's position on the embedding drawn (null: no cells there) */
  positions: (number[] | null)[];
  threshold: number;
  treeOnly?: boolean;
  /** Groups on the chosen branch, in order */
  path?: string[] | null;
  dark?: boolean;
}

/** deck.gl layers for the graph: edges, nodes, then group names above the nodes.
 *  Not pickable, so the cells underneath keep their hover and click. */
export function pagaLayers(opts: PagaLayerOptions): Layer[] {
  const { record, positions, threshold, treeOnly = false, path = null, dark = false } = opts;
  const onPath = new Set(path ?? []);
  const pathPairs = new Set<string>();
  for (let k = 1; k < (path?.length ?? 0); k++) {
    pathPairs.add(pairKey(record.groups.indexOf(path![k - 1]!), record.groups.indexOf(path![k]!)));
  }
  const maxSize = Math.max(0, ...record.sizes);
  const nodes: PagaNode[] = [];
  record.groups.forEach((group, i) => {
    const position = positions[i];
    if (!position) return;
    const size = record.sizes[i] ?? 0;
    nodes.push({ i, group, position, size, radius: nodeRadius(size, maxSize), onPath: onPath.has(group) });
  });
  const edges: DrawnEdge[] = [];
  for (const e of visibleEdges(record, threshold, treeOnly)) {
    const from = positions[e.i];
    const to = positions[e.j];
    if (from && to) edges.push({ ...e, from, to, onPath: pathPairs.has(pairKey(e.i, e.j)) });
  }
  const ink: [number, number, number] = dark ? [226, 232, 240] : [30, 41, 59];
  const over = { depthCompare: "always", depthWriteEnabled: false } as const;
  return [
    new LineLayer<DrawnEdge>({
      id: "paga-edges",
      data: edges,
      getSourcePosition: (d) => d.from as [number, number],
      getTargetPosition: (d) => d.to as [number, number],
      getWidth: (d) => (d.onPath ? 2.5 : 1) + 6 * d.weight,
      widthUnits: "pixels",
      getColor: (d) => (d.onPath ? [...AMBER, 235] : [...ink, Math.round(70 + 150 * d.weight)]),
      pickable: false,
      parameters: over,
    }),
    new ScatterplotLayer<PagaNode>({
      id: "paga-nodes",
      data: nodes,
      getPosition: (d) => d.position as [number, number],
      getRadius: (d) => d.radius,
      radiusUnits: "pixels",
      filled: true,
      stroked: true,
      getFillColor: dark ? [30, 41, 59, 230] : [255, 255, 255, 235],
      getLineColor: (d) => (d.onPath ? [...AMBER, 255] : [...ink, 230]),
      getLineWidth: (d) => (d.onPath ? 3 : 1.5),
      lineWidthUnits: "pixels",
      pickable: false,
      parameters: over,
    }),
    ...centroidLabelLayers(
      "paga-labels",
      nodes.map((n) => ({
        code: n.i,
        label: n.group,
        position: n.position,
        count: n.size,
        offset: [0, -(n.radius + 9)] as [number, number],
      })),
      { dark },
    ),
  ];
}
