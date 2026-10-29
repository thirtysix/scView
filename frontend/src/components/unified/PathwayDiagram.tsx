import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Download, Maximize, Minus, Plus } from "lucide-react";
import type { DiagramNode, DiagramShape, PathwayDiagramModel, PathwayValues, Segment } from "@/api/pathways";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import type { ClipSetting, ResolvedColorScale } from "@/lib/colorScale";
import { ClipControl } from "@/components/plots/ClipControl";
import { colormapColor } from "@/lib/colors";
import { exportFilename } from "@/lib/exportFilename";
import { saveBlob, saveFigure } from "@/lib/pdfExport";
import {
  LABEL_FONT,
  diagramPalette,
  fitViewBox,
  inkOn,
  lfcScale,
  nodeShapeKind,
  nodeSummary,
  octagonPath,
  rgb,
  trianglePoints,
  valueColour,
  wrapLabel,
  zoomViewBox,
  type DiagramPalette,
} from "@/lib/pathwayDiagram";
import { useResolvedTheme } from "@/lib/theme";
import { useSettingsStore } from "@/stores/settingsStore";

interface ViewBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface SvgProps {
  model: PathwayDiagramModel;
  values: PathwayValues | null;
  scale: ResolvedColorScale | null;
  palette: DiagramPalette;
  viewBox: ViewBox;
  width?: number;
  height?: number;
  onHover?: (node: DiagramNode | null, e?: React.MouseEvent) => void;
  onOpenPathway?: (stId: string) => void;
  /** Download: a band below with the legend and the attribution (CC BY needs it on the figure). */
  footer?: { label: string };
}

function Shape({ s, stroke, fill }: { s: DiagramShape; stroke: string; fill: string }) {
  const f = s.empty ? fill : stroke;
  switch (s.type) {
    case "ARROW":
      return <polygon points={trianglePoints(s)} fill={f} stroke={stroke} strokeWidth={1} />;
    case "CIRCLE":
      return s.c ? <circle cx={s.c[0]} cy={s.c[1]} r={s.r ?? 4} fill={f} stroke={stroke} strokeWidth={1} /> : null;
    case "DOUBLE_CIRCLE":
      return s.c ? (
        <g>
          <circle cx={s.c[0]} cy={s.c[1]} r={s.r ?? 5} fill={fill} stroke={stroke} strokeWidth={1} />
          <circle cx={s.c[0]} cy={s.c[1]} r={s.r1 ?? 3} fill={f} stroke={stroke} strokeWidth={1} />
        </g>
      ) : null;
    case "STOP":
      return s.a && s.b ? <line x1={s.a[0]} y1={s.a[1]} x2={s.b[0]} y2={s.b[1]} stroke={stroke} strokeWidth={2} /> : null;
    case "BOX":
      return s.a && s.b ? (
        <g>
          <rect
            x={Math.min(s.a[0], s.b[0])}
            y={Math.min(s.a[1], s.b[1])}
            width={Math.abs(s.b[0] - s.a[0])}
            height={Math.abs(s.b[1] - s.a[1])}
            fill={f}
            stroke={stroke}
            strokeWidth={1}
          />
          {s.s && s.s !== "\\\\" && (
            <text
              x={(s.a[0] + s.b[0]) / 2}
              y={(s.a[1] + s.b[1]) / 2 + 3}
              fontSize={8}
              textAnchor="middle"
              fill={s.empty ? stroke : fill}
            >
              {s.s}
            </text>
          )}
        </g>
      ) : null;
    default:
      return null;
  }
}

function Lines({ segments, stroke, width = 1, dash }: { segments: Segment[]; stroke: string; width?: number; dash?: string }) {
  return (
    <>
      {segments.map((s, i) => (
        <line key={i} x1={s[0]} y1={s[1]} x2={s[2]} y2={s[3]} stroke={stroke} strokeWidth={width} strokeDasharray={dash} />
      ))}
    </>
  );
}

/** One node's outline, by class (lib/pathwayDiagram nodeShapeKind). */
function NodeOutline({ n, fill, stroke, palette }: { n: DiagramNode; fill: string; stroke: string; palette: DiagramPalette }) {
  const dash = n.drug ? "3 2" : undefined;
  const common = { fill, stroke, strokeWidth: 1.2, strokeDasharray: dash };
  switch (nodeShapeKind(n)) {
    case "rounded":
      return <rect data-shape="rounded" x={n.x} y={n.y} width={n.w} height={n.h} rx={6} {...common} />;
    case "octagon":
      return <path data-shape="octagon" d={octagonPath(n.x, n.y, n.w, n.h)} {...common} />;
    case "double":
      return (
        <g data-shape="double">
          <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={4} {...common} />
          <rect x={n.x + 3} y={n.y + 3} width={Math.max(n.w - 6, 1)} height={Math.max(n.h - 6, 1)} rx={3} fill="none" stroke={stroke} strokeWidth={0.8} />
        </g>
      );
    case "ellipse":
      return <ellipse data-shape="ellipse" cx={n.x + n.w / 2} cy={n.y + n.h / 2} rx={n.w / 2} ry={n.h / 2} {...common} />;
    case "gene":
      return (
        <g data-shape="gene">
          <rect x={n.x} y={n.y + n.h * 0.25} width={n.w} height={n.h * 0.75} {...common} />
          <path d={`M${n.x + n.w * 0.5},${n.y + n.h * 0.25}V${n.y + 2}H${n.x + n.w - 6}`} fill="none" stroke={stroke} strokeWidth={1.2} />
        </g>
      );
    case "rna":
      return <rect data-shape="rna" x={n.x} y={n.y} width={n.w} height={n.h} rx={n.h / 2} {...common} strokeDasharray="4 2" />;
    case "process":
      return (
        <g data-shape="process">
          <rect x={n.x} y={n.y} width={n.w} height={n.h} fill={palette.pathwayFill} stroke={stroke} strokeWidth={1.2} />
          <rect x={n.x + 3} y={n.y + 3} width={Math.max(n.w - 6, 1)} height={Math.max(n.h - 6, 1)} fill="none" stroke={stroke} strokeWidth={0.8} />
        </g>
      );
    case "encapsulated":
      return <rect data-shape="encapsulated" x={n.x} y={n.y} width={n.w} height={n.h} rx={2} fill={palette.pathwayFill} stroke={stroke} strokeWidth={1.2} strokeDasharray="5 2" />;
    default:
      return <rect data-shape="rect" x={n.x} y={n.y} width={n.w} height={n.h} {...common} />;
  }
}

/** A node's fill: its value on the scale; grey when none of its genes is measured; the surface without genes. */
export function nodeFill(
  n: DiagramNode,
  values: PathwayValues | null,
  scale: ResolvedColorScale | null,
  palette: DiagramPalette,
): string {
  if (!n.genes || n.genes.length === 0) return palette.noGenes;
  const v = values?.nodes[String(n.id)];
  if (!values || !scale) return palette.noGenes;
  if (!v || v.value === null) return palette.notMeasured;
  return rgb(valueColour(v.value, scale));
}

/** The diagram as plain SVG (no state): the page draws it, and so does a download. */
export function PathwayDiagramSvg({ model, values, scale, palette, viewBox, width, height, onHover, onOpenPathway, footer }: SvgProps) {
  const hl = model.highlight;
  const hlReactions = useMemo(() => new Set(hl?.reactions ?? []), [hl]);
  const band = footer ? 46 : 0;
  const vb = footer ? { ...viewBox, h: viewBox.h + band * (viewBox.w / (width ?? viewBox.w)) } : viewBox;
  const k = footer ? viewBox.w / (width ?? viewBox.w) : 1; // diagram units per px in the band
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
      width={width ?? "100%"}
      height={height ? height + band : "100%"}
      fontFamily="Helvetica, Arial, sans-serif"
      role="img"
      aria-label={`Reactome diagram: ${model.name}`}
      data-testid="pathway-svg"
    >
      <rect x={vb.x} y={vb.y} width={vb.w} height={vb.h} fill={palette.surface} />
      {model.compartments.map((c) => (
        <g key={c.id} data-testid="compartment">
          <rect x={c.x} y={c.y} width={c.w} height={c.h} rx={14} fill={palette.compartmentFill} stroke={palette.compartmentStroke} strokeWidth={2} />
          {c.inset && (
            <rect x={c.inset.x} y={c.inset.y} width={c.inset.w} height={c.inset.h} rx={10} fill="none" stroke={palette.compartmentStroke} strokeWidth={1} />
          )}
          <text x={c.text[0]} y={c.text[1] + 10} fontSize={10} fill={palette.compartmentStroke}>
            {c.label}
          </text>
        </g>
      ))}
      {hl && hl.outline.length > 2 && (
        <polygon
          data-testid="pathway-highlight"
          points={hl.outline.map((p) => `${p[0]},${p[1]}`).join(" ")}
          fill={palette.highlight}
          fillOpacity={0.06}
          stroke={palette.highlight}
          strokeWidth={3}
          strokeDasharray="10 5"
        />
      )}
      {model.links.map((l, i) => (
        <g key={`l${i}`}>
          <Lines segments={l.segments} stroke={palette.muted} dash={l.kind === "EntitySetAndMemberLink" ? "3 3" : undefined} />
          {l.end && <Shape s={l.end} stroke={palette.muted} fill={palette.surface} />}
        </g>
      ))}
      {model.connectors.map((c, i) => {
        const on = c.edge !== null && hlReactions.has(c.edge);
        const stroke = on ? palette.highlight : palette.line;
        return (
          <g key={`c${i}`} data-connector={c.type}>
            <Lines segments={c.segments} stroke={stroke} width={on ? 1.6 : 1} />
            {c.end && <Shape s={c.end} stroke={stroke} fill={palette.surface} />}
          </g>
        );
      })}
      {model.reactions.map((r) => {
        const on = hlReactions.has(r.id);
        const stroke = on ? palette.highlight : palette.line;
        return (
          <g key={`r${r.id}`} data-testid="reaction" data-highlight={on ? "true" : undefined}>
            <Lines segments={r.segments} stroke={stroke} width={on ? 1.8 : 1.1} />
            {r.end && <Shape s={r.end} stroke={stroke} fill={palette.surface} />}
            {r.shape && <Shape s={r.shape} stroke={stroke} fill={palette.surface} />}
          </g>
        );
      })}
      {model.nodes.map((n) => {
        const fill = nodeFill(n, values, scale, palette);
        const isPathway = n.class === "ProcessNode" || n.class === "EncapsulatedNode";
        const textFill = fill.startsWith("rgb(") ? inkOn(fill.slice(4, -1).split(",").map(Number) as [number, number, number]) : palette.ink;
        const lines = wrapLabel(n.label, n.w, n.h);
        const open = isPathway && n.st_id && onOpenPathway ? () => onOpenPathway(n.st_id!) : undefined;
        return (
          <g
            key={n.id}
            data-testid="pathway-node"
            data-class={n.class}
            data-label={n.label}
            data-fill={fill}
            onMouseEnter={onHover ? (e) => onHover(n, e) : undefined}
            onMouseMove={onHover ? (e) => onHover(n, e) : undefined}
            onMouseLeave={onHover ? () => onHover(null) : undefined}
            onClick={open}
            style={open ? { cursor: "pointer" } : undefined}
          >
            <NodeOutline n={n} fill={fill} stroke={palette.nodeStroke} palette={palette} />
            {n.end && <Shape s={n.end} stroke={palette.nodeStroke} fill={palette.nodeStroke} />}
            <text fontSize={LABEL_FONT} textAnchor="middle" fill={isPathway ? palette.ink : textFill}>
              {lines.map((line, i) => (
                <tspan key={i} x={n.x + n.w / 2} y={n.y + n.h / 2 + (i - (lines.length - 1) / 2) * LABEL_FONT * 1.15 + LABEL_FONT * 0.35}>
                  {line}
                </tspan>
              ))}
            </text>
          </g>
        );
      })}
      {footer && (
        <g transform={`translate(${viewBox.x}, ${viewBox.y + viewBox.h}) scale(${k})`} fontSize={11} fill={palette.ink}>
          {scale && (
            <>
              <defs>
                <linearGradient id="pathway-legend" x1="0" x2="1" y1="0" y2="0">
                  {Array.from({ length: 11 }, (_, i) => (
                    <stop key={i} offset={i / 10} stopColor={rgb(colormapColor(scale.colormap, i / 10))} />
                  ))}
                </linearGradient>
              </defs>
              <text x={10} y={16}>{footer.label}: log fold change</text>
              <text x={10} y={34}>{scale.min.toFixed(1)}</text>
              <rect x={40} y={24} width={120} height={12} fill="url(#pathway-legend)" />
              <text x={166} y={34}>{scale.max.toFixed(1)}</text>
              <rect x={200} y={24} width={12} height={12} fill={palette.notMeasured} />
              <text x={216} y={34}>not measured</text>
            </>
          )}
          <text x={10} y={scale ? 44 : 20} fontSize={9} fill={palette.muted}>
            {`Diagram: ${model.attribution} (${model.licence_url}); ${model.citation}; ${model.url}`}
          </text>
        </g>
      )}
    </svg>
  );
}

interface Props {
  model: PathwayDiagramModel;
  values: PathwayValues | null;
  /** Diagram height on the page, px. */
  height?: number;
  onOpenPathway?: (stId: string) => void;
  /** File name stem for downloads (the dataset's name). */
  datasetName?: string | null;
  /** Shown above the drawing (colour-by choice). */
  controls?: ReactNode;
}

/**
 * A Reactome pathway diagram in plain SVG, coloured by a DE result's log fold
 * changes: pan by dragging, zoom with the wheel or the buttons, hover a node for
 * its genes. A pathway drawn inside an ancestor's diagram is outlined.
 */
export function PathwayDiagram({ model, values, height = 460, onOpenPathway, datasetName, controls }: Props) {
  const theme = useResolvedTheme();
  const palette = useMemo(() => diagramPalette(theme), [theme]);
  const diverging = useSettingsStore((s) => s.divergingColormap);
  const defaultClip = useSettingsStore((s) => s.colorClip);
  // One extreme fold change washes out the rest: this view can clip on its own.
  const [clipOverride, setClipOverride] = useState<ClipSetting | null>(null);
  const clip = clipOverride ?? defaultClip;
  const scale = useMemo(() => lfcScale(values, diverging, clip), [values, diverging, clip]);
  const fit = useMemo(() => fitViewBox(model.bounds), [model.bounds]);
  const [vb, setVb] = useState<ViewBox>(fit);
  const [hover, setHover] = useState<{ node: DiagramNode; x: number; y: number } | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; vb: ViewBox } | null>(null);

  useEffect(() => setVb(fit), [fit]);

  const limits = useMemo(() => ({ min: 60, max: Math.max(fit.w * 2, 200) }), [fit]);

  // Diagram units of a client point (the viewBox keeps the aspect: "xMidYMid meet").
  const toDiagram = useCallback((clientX: number, clientY: number, box: ViewBox) => {
    const el = boxRef.current;
    if (!el) return { x: box.x + box.w / 2, y: box.y + box.h / 2 };
    const r = el.getBoundingClientRect();
    const s = Math.max(box.w / Math.max(r.width, 1), box.h / Math.max(r.height, 1));
    const ox = (r.width * s - box.w) / 2;
    const oy = (r.height * s - box.h) / 2;
    return { x: box.x - ox + (clientX - r.left) * s, y: box.y - oy + (clientY - r.top) * s };
  }, []);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setVb((cur) => {
        const p = toDiagram(e.clientX, e.clientY, cur);
        return zoomViewBox(cur, e.deltaY > 0 ? 1.15 : 1 / 1.15, p.x, p.y, limits);
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [limits, toDiagram]);

  const zoomBy = (f: number) => setVb((cur) => zoomViewBox(cur, f, cur.x + cur.w / 2, cur.y + cur.h / 2, limits));

  const onHover = useCallback((node: DiagramNode | null, e?: React.MouseEvent) => {
    if (!node || !e || !boxRef.current) return setHover(null);
    const r = boxRef.current.getBoundingClientRect();
    setHover({ node, x: e.clientX - r.left, y: e.clientY - r.top });
  }, []);

  const exportSvg = useCallback(async () => {
    const { renderToStaticMarkup } = await import("react-dom/server");
    const light = diagramPalette("light");
    const w = Math.round(Math.min(Math.max(fit.w, 600), 2400));
    const h = Math.round((w * fit.h) / fit.w);
    const svg = renderToStaticMarkup(
      <PathwayDiagramSvg model={model} values={values} scale={scale} palette={light} viewBox={fit} width={w} height={h}
        footer={{ label: values?.label ?? "Not coloured" }} />,
    );
    return { svg, width: w, height: h + 46 };
  }, [model, values, scale, fit]);

  const name = exportFilename(datasetName, `reactome_${model.requested}`);
  const downloadSvg = async () => saveFigure("svg", await exportSvg(), name);
  const downloadPng = async () => {
    const { svg, width, height: h } = await exportSvg();
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
    try {
      const img = new Image();
      await new Promise<void>((ok, fail) => {
        img.onload = () => ok();
        img.onerror = () => fail(new Error("Could not draw the diagram for the PNG."));
        img.src = url;
      });
      const canvas = document.createElement("canvas");
      canvas.width = width * 2;
      canvas.height = h * 2;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(2, 2);
      ctx.drawImage(img, 0, 0, width, h);
      canvas.toBlob((blob) => blob && saveBlob(blob, name, "png"), "image/png");
    } finally {
      URL.revokeObjectURL(url);
    }
  };

  const nv = hover ? values?.nodes[String(hover.node.id)] : undefined;
  const btn = "inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[11px] text-slate-600 hover:bg-slate-50";

  return (
    <div className="flex h-full min-h-0 flex-col gap-1.5 text-xs">
      <div className="flex flex-wrap items-center gap-1.5">
        {controls}
        <div className="ml-auto flex items-center gap-1">
          <button type="button" className={btn} onClick={() => zoomBy(1 / 1.3)} aria-label="Zoom in" title="Zoom in">
            <Plus className="h-3 w-3" />
          </button>
          <button type="button" className={btn} onClick={() => zoomBy(1.3)} aria-label="Zoom out" title="Zoom out">
            <Minus className="h-3 w-3" />
          </button>
          <button type="button" className={btn} onClick={() => setVb(fit)} aria-label="Fit the diagram" title="Fit the diagram">
            <Maximize className="h-3 w-3" />
          </button>
          <button type="button" className={btn} onClick={() => void downloadSvg()} title="Download as SVG (light background, legend and attribution included)">
            <Download className="h-3 w-3" /> SVG
          </button>
          <button type="button" className={btn} onClick={() => void downloadPng()} title="Download as PNG (2x)">
            <Download className="h-3 w-3" /> PNG
          </button>
        </div>
      </div>
      <div
        ref={boxRef}
        className="relative min-h-0 flex-1 cursor-grab touch-none overflow-hidden rounded border border-slate-200"
        style={{ minHeight: height }}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          drag.current = { x: e.clientX, y: e.clientY, vb };
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          const el = boxRef.current;
          if (!d || !el) return;
          const r = el.getBoundingClientRect();
          const s = Math.max(d.vb.w / Math.max(r.width, 1), d.vb.h / Math.max(r.height, 1));
          setVb({ ...d.vb, x: d.vb.x - (e.clientX - d.x) * s, y: d.vb.y - (e.clientY - d.y) * s });
        }}
        onPointerUp={() => (drag.current = null)}
        onPointerLeave={() => (drag.current = null)}
      >
        <div className="absolute inset-0">
          <PathwayDiagramSvg model={model} values={values} scale={scale} palette={palette} viewBox={vb}
            onHover={onHover} onOpenPathway={onOpenPathway} />
        </div>
        {hover && (
          <div
            role="tooltip"
            data-testid="pathway-tooltip"
            className="pointer-events-none absolute z-10 max-w-[280px] rounded border border-slate-200 bg-white p-2 text-[11px] text-slate-700 shadow-lg"
            style={{ left: Math.min(hover.x + 12, (boxRef.current?.clientWidth ?? 600) - 290), top: hover.y + 12 }}
          >
            <div className="font-semibold text-slate-800">{hover.node.label}</div>
            <div className="text-slate-500">
              {hover.node.class === "EntitySet" ? "Set (any one member)" : hover.node.class}
              {hover.node.drug ? " (drug)" : ""}
              {nv?.value != null ? ` · mean log FC ${nv.value.toFixed(2)}` : ""}
            </div>
            {(hover.node.class === "ProcessNode" || hover.node.class === "EncapsulatedNode") && hover.node.st_id && onOpenPathway && (
              <div className="text-slate-500">Click to open this pathway</div>
            )}
            {values && nv && <div className="mt-0.5">{nodeSummary(nv, values.padj_cutoff)}</div>}
            {hover.node.genes && hover.node.genes.length > 0 && (
              <ul className="mt-1 space-y-0.5">
                {hover.node.genes.slice(0, 12).map((g) => {
                  const gv = values?.genes[g];
                  return (
                    <li key={g} className="flex justify-between gap-3 tabular-nums">
                      <span className="font-medium">{g}{gv && gv.gene !== g ? ` (${gv.gene})` : ""}</span>
                      <span className={gv ? "" : "text-slate-400"}>
                        {gv
                          ? `${gv.lfc > 0 ? "+" : ""}${gv.lfc.toFixed(2)}${gv.padj !== null ? ` · p.adj ${gv.padj < 0.001 ? gv.padj.toExponential(1) : gv.padj.toFixed(3)}` : ""}`
                          : values ? "not measured" : ""}
                      </span>
                    </li>
                  );
                })}
                {hover.node.genes.length > 12 && <li className="text-slate-400">+ {hover.node.genes.length - 12} more</li>}
              </ul>
            )}
          </div>
        )}
      </div>
      <Legend scale={scale} values={values} palette={palette}
        clip={<ClipControl clip={clip} dataMin={scale?.dataMin} dataMax={scale?.dataMax} onChange={setClipOverride} overridden={clipOverride !== null} />} />
      <p className="text-[10px] leading-snug text-slate-500" data-testid="pathway-attribution">
        Diagram:{" "}
        <a href={model.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
          Reactome
        </a>{" "}
        (
        <a href={model.licence_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
          {model.licence}
        </a>
        ), release {model.release}; {model.citation}.
      </p>
    </div>
  );
}

function Legend({ scale, values, palette, clip }: { scale: ResolvedColorScale | null; values: PathwayValues | null; palette: DiagramPalette; clip?: ReactNode }) {
  if (!values) return <Caption>Not coloured: choose a DE result above.</Caption>;
  const stops = scale
    ? Array.from({ length: 11 }, (_, i) => rgb(colormapColor(scale.colormap, i / 10))).join(", ")
    : "";
  return (
    <div className="flex flex-col gap-0.5" data-testid="pathway-legend">
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
        <span className="inline-flex items-center gap-1">
          Log fold change
          <InfoTip>
            Each protein, gene or RNA box takes its gene&apos;s log fold change from the chosen result; a complex
            or set takes the mean of its measured genes. Grey: none of its genes is in the result. White: no
            gene (a small molecule). The colours say a gene&apos;s RNA changed, not that the step is more or
            less active.
          </InfoTip>
        </span>
        {scale && (
          <span className="inline-flex items-center gap-1 tabular-nums">
            {scale.min.toFixed(1)}
            <span className="inline-block h-2.5 w-28 rounded-sm" style={{ background: `linear-gradient(to right, ${stops})` }} />
            {scale.max.toFixed(1)}
          </span>
        )}
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm border border-slate-300" style={{ background: palette.notMeasured }} />
          not measured
        </span>
        {scale && clip}
      </div>
      <Caption
        n={[
          { value: values.n_measured, of: values.n_diagram_genes, unit: "diagram genes measured" },
          ...(values.n_changed !== null ? [{ value: values.n_changed, unit: `changed (adjusted p < ${values.padj_cutoff})` }] : []),
        ]}
      >
        {values.label}.{scale?.clip ? " Colour range clipped: boxes beyond it take the end colours." : ""}
      </Caption>
    </div>
  );
}
