import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PpiLink, PpiNode } from "@/api/ppi";
import { useResolvedTheme } from "@/lib/theme";
import { communityColour, logfcColour, maxAbsLogfc, nodeRadius, type PpiColourBy } from "@/lib/ppi";

/**
 * Protein interaction subnetwork on a plain 2D canvas, the way the term network
 * (EnrichmentNetworkCanvas) is drawn: no Plotly trace type for edges, no new
 * dependency. Positions come from the backend (seeded spring layout); proteins
 * with no partner in the list have none and are not drawn. Mouse only: the
 * proteins table beside it holds the same information for keyboard users.
 */

interface Props {
  nodes: PpiNode[];
  links: PpiLink[];
  colourBy: PpiColourBy;
  height?: number;
  onNodeClick?: (node: PpiNode) => void;
}

export function PpiNetworkCanvas({ nodes, links, colourBy, height = 360, onNodeClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layoutRef = useRef<{ px: (n: PpiNode) => number; py: (n: PpiNode) => number } | null>(null);
  const [hover, setHover] = useState<{ node: PpiNode; x: number; y: number } | null>(null);
  const theme = useResolvedTheme();
  const drawn = useMemo(() => nodes.filter((n) => n.x !== null && n.y !== null), [nodes]);

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv || drawn.length === 0) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const dark = theme === "dark";
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = cv.clientWidth;
    const H = height;
    cv.width = W * dpr;
    cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const xs = drawn.map((n) => n.x!);
    const ys = drawn.map((n) => n.y!);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = 22;
    const sc = Math.min((W - pad * 2) / Math.max(x1 - x0, 1e-6), (H - pad * 2) / Math.max(y1 - y0, 1e-6));
    const ox = (W - (x1 - x0) * sc) / 2;
    const oy = (H - (y1 - y0) * sc) / 2;
    const px = (n: PpiNode) => (n.x! - x0) * sc + ox;
    const py = (n: PpiNode) => (n.y! - y0) * sc + oy;
    layoutRef.current = { px, py };

    // Edge opacity by STRING score: 400 faint, 1000 strong.
    for (const l of links) {
      const a = nodes[l.source], b = nodes[l.target];
      if (!a || !b || a.x === null || b.x === null) continue;
      const alpha = 0.12 + 0.4 * Math.max(0, Math.min(1, (l.score - 400) / 600));
      ctx.strokeStyle = dark ? `rgba(203,213,225,${alpha})` : `rgba(71,85,105,${alpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
      ctx.stroke();
    }

    const maxAbs = maxAbsLogfc(nodes);
    const colour = (n: PpiNode) => (colourBy === "logfc" ? logfcColour(n.logfc, maxAbs) : communityColour(n.community));
    const ordered = [...drawn].sort((a, b) => a.degree - b.degree); // hubs on top
    for (const n of ordered) {
      ctx.beginPath();
      ctx.arc(px(n), py(n), nodeRadius(n.degree), 0, Math.PI * 2);
      ctx.fillStyle = colour(n);
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = dark ? "#0f172a" : "#ffffff";
      ctx.stroke();
    }

    // Hub labels, on a backing so they stay readable over edges.
    ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const n of drawn.filter((d) => d.hub)) {
      const x = px(n);
      const y = py(n) - nodeRadius(n.degree) - 7;
      const w = ctx.measureText(n.gene).width;
      ctx.fillStyle = dark ? "rgba(15,23,42,0.85)" : "rgba(255,255,255,0.85)";
      ctx.beginPath();
      ctx.roundRect(x - w / 2 - 3, y - 6, w + 6, 12, 3);
      ctx.fill();
      ctx.fillStyle = dark ? "#e2e8f0" : "#1e293b";
      ctx.fillText(n.gene, x, y);
    }
  }, [drawn, nodes, links, colourBy, height, theme]);

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
      let best: PpiNode | null = null;
      let bestD = Infinity;
      for (const n of drawn) {
        const dx = lay.px(n) - mx, dy = lay.py(n) - my;
        const d = dx * dx + dy * dy;
        const reach = Math.max(9, nodeRadius(n.degree) + 2);
        if (d < reach * reach && d < bestD) { bestD = d; best = n; }
      }
      return best ? { node: best, x: mx, y: my } : null;
    },
    [drawn],
  );

  if (drawn.length === 0) return null;

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        data-testid="ppi-canvas"
        style={{ width: "100%", height, display: "block", cursor: hover ? "pointer" : "default" }}
        onMouseMove={(e) => setHover(hitTest(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const h = hitTest(e);
          if (h && onNodeClick) onNodeClick(h.node);
        }}
        role="img"
        aria-label={`Protein interaction network: ${drawn.length} proteins with at least one partner, sized by number of partners, the hubs labelled. The table lists every protein.`}
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 max-w-[220px] rounded border border-slate-200 bg-white px-2 py-1 text-[10px] shadow-lg"
          style={{ left: Math.min(hover.x + 10, 240), top: Math.max(hover.y - 34, 0) }}
        >
          <div className="font-medium text-slate-800">
            {hover.node.gene}
            {hover.node.name.toUpperCase() !== hover.node.gene.toUpperCase() ? ` (${hover.node.name})` : ""}
          </div>
          <div className="tabular-nums text-slate-500">
            {hover.node.degree} partner{hover.node.degree === 1 ? "" : "s"}
            {hover.node.community !== null ? ` · community ${hover.node.community}` : ""}
            {hover.node.logfc !== null ? ` · log2FC ${hover.node.logfc.toFixed(2)}` : ""}
          </div>
        </div>
      )}
    </div>
  );
}
