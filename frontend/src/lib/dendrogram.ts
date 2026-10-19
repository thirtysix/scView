/**
 * A dendrogram drawn as Plotly lines (Plotly has no dendrogram trace), from
 * the scipy linkage rows GET /clustering/distances returns as `tree` (C043).
 *
 * Coordinates are (position, height): a leaf's position is its index in the
 * leaf order (0 = first row of the heatmap beside it), heights are the merge
 * distances. The caller maps them onto x and y for its orientation.
 */
import type { Data, Layout } from "plotly.js";

/** One linkage row: [node a, node b, height, groups under the new node].
 *  Node ids below n are groups; n + k is the node made by row k. */
export type LinkageRow = [number, number, number, number];

export interface DendrogramLines {
  /** Leaf-order positions, null between segments */
  pos: (number | null)[];
  /** Heights, null between segments */
  height: (number | null)[];
  maxHeight: number;
  /** Number of leaves */
  n: number;
}

/** Line segments of the tree, or null when `tree` and `order` don't describe
 *  the same n groups (an older server, or a tree for other groups). */
export function dendrogramSegments(tree: readonly (readonly number[])[], order: readonly number[]): DendrogramLines | null {
  const n = order.length;
  if (n < 2 || tree.length !== n - 1) return null;
  const leafPos = new Map<number, number>();
  order.forEach((leaf, i) => leafPos.set(leaf, i));
  if (leafPos.size !== n || [...leafPos.keys()].some((k) => k < 0 || k >= n)) return null;
  const nodePos: number[] = [];
  const nodeHeight: number[] = [];
  const at = (id: number): [number, number] | null => {
    if (id < n) {
      const p = leafPos.get(id);
      return p === undefined ? null : [p, 0];
    }
    const k = id - n;
    return k < nodePos.length ? [nodePos[k]!, nodeHeight[k]!] : null;
  };
  const pos: (number | null)[] = [];
  const height: (number | null)[] = [];
  let maxHeight = 0;
  for (const row of tree) {
    const [a, b, h] = row as LinkageRow;
    const pa = at(a);
    const pb = at(b);
    if (!pa || !pb || !Number.isFinite(h)) return null;
    pos.push(pa[0], pa[0], pb[0], pb[0], null);
    height.push(pa[1], h, h, pb[1], null);
    nodePos.push((pa[0] + pb[0]) / 2);
    nodeHeight.push(h);
    maxHeight = Math.max(maxHeight, h);
  }
  return { pos, height, maxHeight, n };
}

/** Groups in dendrogram order: the tree's groups first (those in `rows`), then
 *  any of `rows` the tree does not have, in their own order. */
export function dendrogramRowOrder(treeGroups: readonly string[], order: readonly number[], rows: readonly string[]): string[] {
  const inRows = new Set(rows);
  const ordered = order.map((i) => treeGroups[i]).filter((g): g is string => g !== undefined && inRows.has(g));
  const seen = new Set(ordered);
  return [...ordered, ...rows.filter((g) => !seen.has(g))];
}

/**
 * A Plotly trace and axes that draw the tree beside a plot whose rows are
 * category y values top to bottom (yaxis range [n - 0.5, -0.5]), on its own
 * axes x2 / y2: the leaves face the plot on the left, the root on the right.
 * `domain` is the horizontal strip of the figure it takes; `yDomain` must be
 * the plot's own y domain when that is not the whole height.
 */
export function rowDendrogram(
  lines: DendrogramLines,
  domain: [number, number],
  yDomain?: [number, number],
): { trace: Partial<Data>; layout: Partial<Layout> } {
  return {
    trace: {
      type: "scatter",
      mode: "lines",
      x: lines.height,
      y: lines.pos,
      xaxis: "x2",
      yaxis: "y2",
      line: { color: "#64748b", width: 1 },
      hoverinfo: "skip",
      showlegend: false,
      connectgaps: false,
    } as Partial<Data>,
    layout: {
      xaxis2: {
        domain,
        anchor: "y2",
        range: [0, (lines.maxHeight || 1) * 1.04],
        visible: false,
        fixedrange: true,
      },
      yaxis2: {
        anchor: "x2",
        ...(yDomain ? { domain: yDomain } : {}),
        range: [lines.n - 0.5, -0.5],
        visible: false,
        fixedrange: true,
      },
    } as Partial<Layout>,
  };
}

/**
 * A Plotly trace and axes that draw the tree above a plot whose columns are
 * category x values left to right (xaxis range [-0.5, n - 0.5]), on its own
 * axes x3 / y3: the leaves face the plot below, the root at the top.
 * `xDomain` must be the plot's own x domain; `yDomain` is the band it takes.
 */
export function columnDendrogram(
  lines: DendrogramLines,
  xDomain: [number, number],
  yDomain: [number, number],
): { trace: Partial<Data>; layout: Partial<Layout> } {
  return {
    trace: {
      type: "scatter",
      mode: "lines",
      x: lines.pos,
      y: lines.height,
      xaxis: "x3",
      yaxis: "y3",
      line: { color: "#64748b", width: 1 },
      hoverinfo: "skip",
      showlegend: false,
      connectgaps: false,
    } as Partial<Data>,
    layout: {
      xaxis3: {
        domain: xDomain,
        anchor: "y3",
        range: [-0.5, lines.n - 0.5],
        visible: false,
        fixedrange: true,
      },
      yaxis3: {
        domain: yDomain,
        anchor: "x3",
        range: [0, (lines.maxHeight || 1) * 1.04],
        visible: false,
        fixedrange: true,
      },
    } as Partial<Layout>,
  };
}
