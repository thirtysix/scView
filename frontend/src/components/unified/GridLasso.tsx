import { useCallback, useRef, useState } from "react";
import { cellsInPanelLasso, panelAt } from "@/lib/splitLasso";
import { useSelectionStore } from "@/stores/selectionStore";

/**
 * Lasso for a grid whose panels all draw the same cells (compare colourings,
 * gene grids): the cells under the outline, whichever panel it was drawn in.
 * `onLasso` is undefined without `onComplete`, which leaves the lasso off.
 */
export function useSameCellsGridLasso(
  positions: Float32Array,
  dimensions: 2 | 3,
  onComplete?: (cells: Set<number>) => void,
): {
  lassoActive: boolean;
  onLasso?: (e: { polygon: [number, number][]; viewState: Record<string, unknown>; width: number; height: number }) => void;
} {
  const lassoActive = useSelectionStore((s) => s.selectionMode === "lasso");
  const onLasso = useCallback(
    ({ polygon, viewState, width, height }: { polygon: [number, number][]; viewState: Record<string, unknown>; width: number; height: number }) =>
      onComplete?.(cellsInPanelLasso(positions, dimensions, polygon, viewState, width, height)),
    [positions, dimensions, onComplete],
  );
  return { lassoActive, onLasso: onComplete ? onLasso : undefined };
}

type Rect = { x: number; y: number; width: number; height: number };

/** A lasso closed in one panel: the outline in that panel view's own pixels. */
export interface GridLassoResult {
  panel: number;
  polygon: [number, number][];
  width: number;
  height: number;
}

/**
 * The lasso over a multi-panel grid (C039): drawn in the panel where the drag
 * starts and kept inside it, then handed back in that panel's own pixels, so
 * the caller maps it through the shared camera like the single plot's lasso.
 */
export function GridLasso({
  cells,
  onComplete,
}: {
  /** Each panel's view rectangle, in pixels of the grid area */
  cells: readonly { view: Rect }[];
  onComplete: (result: GridLassoResult) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [panel, setPanel] = useState(-1);
  const [points, setPoints] = useState<[number, number][]>([]);

  const local = (e: React.MouseEvent<SVGSVGElement>): [number, number] | null => {
    const rect = svgRef.current?.getBoundingClientRect();
    return rect ? [e.clientX - rect.left, e.clientY - rect.top] : null;
  };

  const onDown = (e: React.MouseEvent<SVGSVGElement>) => {
    const p = local(e);
    if (!p) return;
    const i = panelAt(cells, p[0], p[1]);
    if (i < 0) return;
    e.preventDefault();
    e.stopPropagation();
    setPanel(i);
    setPoints([p]);
  };
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (panel < 0) return;
    const p = local(e);
    const v = cells[panel]?.view;
    if (!p || !v) return;
    // Kept inside the panel the drag started in.
    const x = Math.min(Math.max(p[0], v.x), v.x + v.width);
    const y = Math.min(Math.max(p[1], v.y), v.y + v.height);
    setPoints((prev) => [...prev, [x, y]]);
  };
  const onUp = useCallback(() => {
    if (panel < 0) return;
    const v = cells[panel]?.view;
    if (v && points.length >= 3) {
      onComplete({
        panel,
        polygon: points.map(([x, y]) => [x - v.x, y - v.y]),
        width: v.width,
        height: v.height,
      });
    }
    setPanel(-1);
    setPoints([]);
  }, [panel, points, cells, onComplete]);

  const path = points.length > 1 ? `M ${points.map(([x, y]) => `${x},${y}`).join(" L ")} Z` : "";
  return (
    <svg
      ref={svgRef}
      data-testid="grid-lasso"
      className="absolute inset-0 z-10 h-full w-full"
      style={{ cursor: "crosshair" }}
      onMouseDown={onDown}
      onMouseMove={onMove}
      onMouseUp={onUp}
      onMouseLeave={onUp}
    >
      {path && (
        <path d={path} fill="rgba(255, 165, 0, 0.15)" stroke="rgba(255, 140, 0, 0.8)" strokeWidth={1.5} strokeDasharray="4 2" />
      )}
    </svg>
  );
}
