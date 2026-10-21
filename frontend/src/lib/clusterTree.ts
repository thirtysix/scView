/** Layout and Plotly traces for the clustering tree (components/unified/ClusterTreePlot). */
import type { Data } from "plotly.js";
import type { ClusterTree, ClusterTreeEdge, ClusterTreeNode } from "@/api/clustering";

export interface PlacedNode extends ClusterTreeNode {
  /** Level index: 0 is the lowest resolution */
  level: number;
  x: number;
}

export interface TreeLayout {
  resolutions: number[];
  nodes: PlacedNode[];
  /** Node by `${resolution}|${cluster}` */
  byKey: Map<string, PlacedNode>;
  /** Half the widest level's width, for the x range */
  halfWidth: number;
}

export const nodeKey = (resolution: number, cluster: string) => `${resolution}|${cluster}`;

/** Numeric cluster names in numeric order, the rest after them alphabetically. */
function byName(a: ClusterTreeNode, b: ClusterTreeNode): number {
  const na = Number(a.cluster);
  const nb = Number(b.cluster);
  if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
  if (Number.isFinite(na)) return -1;
  if (Number.isFinite(nb)) return 1;
  return a.cluster.localeCompare(b.cluster);
}

/**
 * Place each level's clusters on a row, ordered to keep edges short: the first
 * level by name, then each level by the weighted mean position of its parents
 * (the barycentre heuristic), swept down, up and down again.
 */
export function layoutTree(tree: ClusterTree): TreeLayout {
  const resolutions = [...new Set(tree.nodes.map((n) => n.resolution))].sort((a, b) => a - b);
  const levels: ClusterTreeNode[][] = resolutions.map((r) =>
    tree.nodes.filter((n) => n.resolution === r).sort(byName),
  );
  const pos = new Map<string, number>();
  const place = (level: ClusterTreeNode[]) =>
    level.forEach((n, i) => pos.set(nodeKey(n.resolution, n.cluster), i - (level.length - 1) / 2));
  levels.forEach(place);

  const into = new Map<string, ClusterTreeEdge[]>(); // edges by child
  const out = new Map<string, ClusterTreeEdge[]>(); // edges by parent
  for (const e of tree.edges) {
    const c = nodeKey(e.to_resolution, e.to);
    const p = nodeKey(e.from_resolution, e.from);
    into.set(c, [...(into.get(c) ?? []), e]);
    out.set(p, [...(out.get(p) ?? []), e]);
  }
  const reorder = (i: number, down: boolean) => {
    const level = levels[i]!;
    const centre = new Map<string, number>();
    for (const n of level) {
      const k = nodeKey(n.resolution, n.cluster);
      const es = (down ? into.get(k) : out.get(k)) ?? [];
      let w = 0;
      let sum = 0;
      for (const e of es) {
        const other = down ? nodeKey(e.from_resolution, e.from) : nodeKey(e.to_resolution, e.to);
        sum += e.count * (pos.get(other) ?? 0);
        w += e.count;
      }
      centre.set(k, w > 0 ? sum / w : (pos.get(k) ?? 0));
    }
    level.sort(
      (a, b) =>
        centre.get(nodeKey(a.resolution, a.cluster))! - centre.get(nodeKey(b.resolution, b.cluster))! ||
        b.size - a.size,
    );
    place(level);
  };
  for (let i = 1; i < levels.length; i++) reorder(i, true);
  for (let i = levels.length - 2; i >= 0; i--) reorder(i, false);
  for (let i = 1; i < levels.length; i++) reorder(i, true);

  const nodes: PlacedNode[] = levels.flatMap((level, li) =>
    level.map((n) => ({ ...n, level: li, x: pos.get(nodeKey(n.resolution, n.cluster))! })),
  );
  const byKey = new Map(nodes.map((n) => [nodeKey(n.resolution, n.cluster), n]));
  const halfWidth = Math.max(0.5, ...levels.map((l) => (l.length - 1) / 2));
  return { resolutions, nodes, byKey, halfWidth };
}

const WIDTHS = [1, 2.5, 4.5, 7];
const ALPHAS = [0.25, 0.5, 0.85];

/** Width bin of an edge by its cells (sqrt scale) and opacity bin by the child's share. */
export function edgeBins(e: ClusterTreeEdge, maxCount: number): { width: number; alpha: number } {
  const w = Math.min(WIDTHS.length - 1, Math.floor(Math.sqrt(e.count / Math.max(maxCount, 1)) * WIDTHS.length));
  const a = e.in_prop >= 2 / 3 ? 2 : e.in_prop >= 1 / 3 ? 1 : 0;
  return { width: WIDTHS[w]!, alpha: ALPHAS[a]! };
}

/** Hover text for a node: its size, stability and where its cells came from. */
export function nodeHover(n: ClusterTreeNode, edges: readonly ClusterTreeEdge[]): string {
  const from = edges
    .filter((e) => e.to_resolution === n.resolution && e.to === n.cluster)
    .sort((a, b) => b.in_prop - a.in_prop)
    .map((e) => `${e.from} (${Math.round(e.in_prop * 100)}%)`);
  return [
    `Resolution ${n.resolution}, cluster ${n.cluster}`,
    `${n.size.toLocaleString("en-US")} cells`,
    `SC3 stability ${n.sc3_stability.toFixed(2)}`,
    ...(from.length ? [`From: ${from.join(", ")}`] : []),
    ...(n.mixed ? ["Draws on more than one parent"] : []),
  ].join("<br>");
}

/** The tree as Plotly traces: edges grouped by width and opacity, then the nodes. */
export function treeTraces(tree: ClusterTree, layout: TreeLayout): Data[] {
  const maxCount = Math.max(1, ...tree.edges.map((e) => e.count));
  const groups = new Map<string, { width: number; alpha: number; x: (number | null)[]; y: (number | null)[] }>();
  for (const e of tree.edges) {
    const a = layout.byKey.get(nodeKey(e.from_resolution, e.from));
    const b = layout.byKey.get(nodeKey(e.to_resolution, e.to));
    if (!a || !b) continue;
    const bin = edgeBins(e, maxCount);
    const key = `${bin.width}|${bin.alpha}`;
    const g = groups.get(key) ?? { ...bin, x: [], y: [] };
    g.x.push(a.x, b.x, null);
    g.y.push(a.level, b.level, null);
    groups.set(key, g);
  }
  const edgeTraces = [...groups.values()].map((g) => ({
    type: "scatter",
    mode: "lines",
    x: g.x,
    y: g.y,
    hoverinfo: "skip",
    line: { width: g.width, color: `rgba(71, 85, 105, ${g.alpha})` },
    showlegend: false,
  }));
  const maxSize = Math.max(1, ...layout.nodes.map((n) => n.size));
  const crowded = Math.max(...layout.resolutions.map((r) => layout.nodes.filter((n) => n.resolution === r).length)) > 25;
  const nodeTrace = {
    type: "scatter",
    mode: crowded ? "markers" : "markers+text",
    x: layout.nodes.map((n) => n.x),
    y: layout.nodes.map((n) => n.level),
    text: layout.nodes.map((n) => n.cluster),
    textposition: "middle center",
    textfont: { size: 8, color: "#0f172a" },
    customdata: layout.nodes.map((n) => n.resolution),
    hovertext: layout.nodes.map((n) => nodeHover(n, tree.edges)),
    hoverinfo: "text",
    marker: {
      size: layout.nodes.map((n) => 10 + 22 * Math.sqrt(n.size / maxSize)),
      color: layout.nodes.map((n) => n.sc3_stability),
      cmin: 0,
      cmax: 1,
      colorscale: "YlGnBu",
      reversescale: true,
      colorbar: { title: { text: "SC3<br>stability", font: { size: 10 } }, thickness: 10, len: 0.8, tickfont: { size: 9 } },
      line: {
        width: layout.nodes.map((n) => (n.mixed ? 2.5 : 1)),
        color: layout.nodes.map((n) => (n.mixed ? "#dc2626" : "#ffffff")),
      },
    },
    showlegend: false,
  } as Partial<Data>;
  return [...(edgeTraces as Data[]), nodeTrace as Data];
}
