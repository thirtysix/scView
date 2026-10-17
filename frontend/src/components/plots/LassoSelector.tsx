import { useCallback, useRef, useState } from "react";

interface LassoSelectorProps {
  active: boolean;
  positions: Float32Array | null;
  /** The scatter's camera as shown when the lasso closes (the fit, a reset or
   *  the user's own), read then rather than passed on every camera move */
  getViewState: () => Record<string, unknown> | null;
  onSelectionComplete: (indices: Set<number>) => void;
}

/**
 * Test whether a 2-D point (px, py) lies inside the polygon defined by
 * `polygon` using the ray-casting algorithm.
 */
export function pointInPolygon(
  px: number,
  py: number,
  polygon: [number, number][],
): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]!;
    const [xj, yj] = polygon[j]!;
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Convert screen-space coordinates (pixels from the plot's top-left) to
 * data-space coordinates using the orthographic view state from deck.gl. The
 * scatter's OrthographicView has flipY false: data y grows upwards while
 * screen y grows downwards.
 */
export function screenToData(
  screenX: number,
  screenY: number,
  viewState: Record<string, unknown>,
  width: number,
  height: number,
): [number, number] {
  const target = (viewState.target as number[]) ?? [0, 0, 0];
  const zoom = (viewState.zoom as number) ?? 0;
  const scale = Math.pow(2, zoom);

  const dataX = target[0]! + (screenX - width / 2) / scale;
  const dataY = target[1]! - (screenY - height / 2) / scale;

  return [dataX, dataY];
}

export function LassoSelector({
  active,
  positions,
  getViewState,
  onSelectionComplete,
}: LassoSelectorProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [points, setPoints] = useState<[number, number][]>([]);
  const [isDrawing, setIsDrawing] = useState(false);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!active) return;
      e.preventDefault();
      e.stopPropagation();
      const svg = svgRef.current;
      if (!svg) return;

      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setPoints([[x, y]]);
      setIsDrawing(true);
    },
    [active],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!isDrawing || !active) return;
      const svg = svgRef.current;
      if (!svg) return;

      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setPoints((prev) => [...prev, [x, y]]);
    },
    [isDrawing, active],
  );

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !active) return;
    setIsDrawing(false);

    const viewState = getViewState();
    if (points.length < 3 || !positions || !viewState || !svgRef.current) {
      setPoints([]);
      return;
    }

    // Convert screen-space polygon to data-space
    const rect = svgRef.current.getBoundingClientRect();
    const dataPolygon: [number, number][] = points.map(([sx, sy]) =>
      screenToData(sx, sy, viewState, rect.width, rect.height),
    );

    // Find all cell indices inside the polygon
    const selected = new Set<number>();
    const numCells = positions.length / 2;
    for (let i = 0; i < numCells; i++) {
      const cx = positions[i * 2]!;
      const cy = positions[i * 2 + 1]!;
      if (pointInPolygon(cx, cy, dataPolygon)) {
        selected.add(i);
      }
    }

    onSelectionComplete(selected);
    setPoints([]);
  }, [isDrawing, active, points, positions, getViewState, onSelectionComplete]);

  if (!active) return null;

  const pathData =
    points.length > 1
      ? `M ${points.map(([x, y]) => `${x},${y}`).join(" L ")} Z`
      : "";

  return (
    <svg
      ref={svgRef}
      className="absolute inset-0 z-10 h-full w-full"
      style={{ cursor: "crosshair" }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {pathData && (
        <path
          d={pathData}
          fill="rgba(255, 165, 0, 0.15)"
          stroke="rgba(255, 140, 0, 0.8)"
          strokeWidth={1.5}
          strokeDasharray="4 2"
        />
      )}
    </svg>
  );
}
