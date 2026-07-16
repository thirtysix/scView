import { useRef, useEffect, useState, useCallback } from "react";
import type { NetworkNode, NetworkEdge } from "@/api/enrichmentNetwork";

/**
 * Term-similarity network on a plain 2D canvas.
 *
 * Deliberately not Plotly: `lib/plotly.ts` is a hand-rolled partial bundle that
 * registers only bar/scatter/scattergl/box/violin, and an unregistered trace type
 * fails at runtime rather than at build. Edges would need a line trace. A canvas
 * needs no bundle change and no new dependency.
 *
 * Only the largest clusters take a colour. The palette is validated for
 * colourblind separation across all pairs at four hues; past that, hues cannot be
 * told apart, so the tail stays grey rather than cycling. Colour is never the only
 * identity cue: every coloured cluster is also directly labelled.
 */

const SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"];
const OTHER = "#aab2bd";
const EDGE = "rgba(107,117,131,0.13)";
const MAX_COLOURED = SERIES.length;

// Edges below this similarity are drawn out; they add ink without structure.
const EDGE_DRAW_MIN = 0.5;

interface Props {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  /** Cluster ids in display order; the first four get a colour. */
  clusterOrder: number[];
  labels: Record<number, string>;
  onNodeClick?: (node: NetworkNode) => void;
  height?: number;
}

const shortLabel = (t: string) =>
  t.replace(/^GOBP_/, "").replace(/^GOMF_/, "").replace(/^GOCC_/, "")
    .replace(/^HALLMARK_/, "H:").replace(/_/g, " ").toLowerCase();

export function EnrichmentNetworkCanvas({
  nodes, edges, clusterOrder, labels, onNodeClick, height = 340,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layoutRef = useRef<{ px: (n: NetworkNode) => number; py: (n: NetworkNode) => number } | null>(null);
  const [hover, setHover] = useState<{ node: NetworkNode; x: number; y: number } | null>(null);

  const colourOf = useCallback(
    (cluster: number) => {
      const rank = clusterOrder.indexOf(cluster);
      return rank >= 0 && rank < MAX_COLOURED ? SERIES[rank]! : OTHER;
    },
    [clusterOrder],
  );

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv || nodes.length === 0) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = cv.clientWidth;
    const H = height;
    cv.width = W * dpr;
    cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // Clamp to percentiles: a few layout outliers otherwise squeeze the core
    // into an unreadable knot in the middle.
    const q = (arr: number[], f: number) => {
      const s = [...arr].sort((a, b) => a - b);
      return s[Math.min(s.length - 1, Math.max(0, Math.round(f * (s.length - 1))))]!;
    };
    const xs = nodes.map((n) => n.x);
    const ys = nodes.map((n) => n.y);
    const x0 = q(xs, 0.01), x1 = q(xs, 0.99), y0 = q(ys, 0.01), y1 = q(ys, 0.99);
    const pad = 18;
    const sc = Math.min((W - pad * 2) / Math.max(x1 - x0, 1e-6), (H - pad * 2) / Math.max(y1 - y0, 1e-6));
    const ox = (W - (x1 - x0) * sc) / 2;
    const oy = (H - (y1 - y0) * sc) / 2;
    const px = (n: NetworkNode) => (n.x - x0) * sc + ox;
    const py = (n: NetworkNode) => (n.y - y0) * sc + oy;
    layoutRef.current = { px, py };

    ctx.lineWidth = 1;
    ctx.strokeStyle = EDGE;
    ctx.beginPath();
    for (const e of edges) {
      if (e.weight < EDGE_DRAW_MIN) continue;
      const a = nodes[e.source], b = nodes[e.target];
      if (!a || !b) continue;
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
    }
    ctx.stroke();

    // Tail first so identified clusters sit on top, each with a surface ring.
    const ordered = [...nodes].sort(
      (a, b) => (colourOf(a.cluster) === OTHER ? 0 : 1) - (colourOf(b.cluster) === OTHER ? 0 : 1),
    );
    for (const n of ordered) {
      const c = colourOf(n.cluster);
      const isTail = c === OTHER;
      ctx.beginPath();
      ctx.arc(px(n), py(n), isTail ? 2.4 : 4, 0, Math.PI * 2);
      ctx.fillStyle = c;
      ctx.fill();
      if (!isTail) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      }
    }

    // Direct labels at each coloured cluster's centroid, pushed apart vertically
    // so they never stack. This is the secondary encoding the palette requires.
    ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const placed = clusterOrder.slice(0, MAX_COLOURED).map((cid) => {
      const pts = nodes.filter((n) => n.cluster === cid);
      if (!pts.length) return null;
      return {
        lab: shortLabel(labels[cid] ?? ""),
        x: pts.reduce((s, n) => s + px(n), 0) / pts.length,
        y: pts.reduce((s, n) => s + py(n), 0) / pts.length,
      };
    }).filter(Boolean) as { lab: string; x: number; y: number }[];
    placed.sort((a, b) => a.y - b.y);
    for (let i = 1; i < placed.length; i++) {
      if (placed[i]!.y - placed[i - 1]!.y < 18) placed[i]!.y = placed[i - 1]!.y + 18;
    }
    for (const { lab, x, y } of placed) {
      const w = ctx.measureText(lab).width;
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.beginPath();
      ctx.roundRect(x - w / 2 - 5, y - 8, w + 10, 16, 4);
      ctx.fill();
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x - w / 2 - 5, y - 8, w + 10, 16, 4);
      ctx.stroke();
      ctx.fillStyle = "#1e293b";
      ctx.fillText(lab, x, y);
    }
  }, [nodes, edges, clusterOrder, labels, colourOf, height]);

  useEffect(() => {
    draw();
    const ro = new ResizeObserver(draw);
    if (canvasRef.current) ro.observe(canvasRef.current);
    return () => ro.disconnect();
  }, [draw]);

  const hitTest = useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>) => {
      const lay = layoutRef.current;
      const cv = canvasRef.current;
      if (!lay || !cv) return null;
      const r = cv.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let best: NetworkNode | null = null;
      let bestD = 81; // 9px radius, a hit target larger than the mark
      for (const n of nodes) {
        const dx = lay.px(n) - mx, dy = lay.py(n) - my;
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = n; }
      }
      return best ? { node: best, x: mx, y: my } : null;
    },
    [nodes],
  );

  if (nodes.length === 0) return null;

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height, display: "block", cursor: hover ? "pointer" : "default" }}
        onMouseMove={(e) => setHover(hitTest(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const h = hitTest(e);
          if (h && onNodeClick) onNodeClick(h.node);
        }}
        role="img"
        aria-label={`Term similarity network: ${nodes.length} enriched terms as nodes, positioned by graph layout, with the largest clusters coloured and labelled.`}
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 max-w-[240px] rounded border border-slate-200 bg-white px-2 py-1 text-[10px] shadow-lg"
          style={{
            left: Math.min(hover.x + 10, 240),
            top: Math.max(hover.y - 34, 0),
          }}
        >
          <div className="font-medium text-slate-800">{hover.node.term}</div>
          <div className="tabular-nums text-slate-500">
            adj.P {hover.node.adjusted_pvalue.toExponential(1)} &middot; {hover.node.overlap_count}/
            {hover.node.gene_count} genes
          </div>
        </div>
      )}
    </div>
  );
}
