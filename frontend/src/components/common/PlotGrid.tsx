import type { CSSProperties, ReactNode } from "react";

interface PlotGridProps {
  children: ReactNode;
  /** Minimum tile width in px before wrapping to fewer columns (default 360). */
  minTile?: number;
  /** Fixed row height in px; omit to let tiles size to content. */
  rowHeight?: number;
  className?: string;
}

/**
 * Responsive plot-tiling grid (wnt-hub `.bulk-charts-grid` style): each tile is
 * at least `minTile` px wide and grows to fill the row, wrapping to fewer
 * columns as space shrinks. With `rowHeight` set, tiles share a uniform height
 * so plots that fill their container render at a consistent size.
 */
export function PlotGrid({ children, minTile = 360, rowHeight, className }: PlotGridProps) {
  const style: CSSProperties = {
    gridTemplateColumns: `repeat(auto-fit, minmax(${minTile}px, 1fr))`,
    ...(rowHeight ? { gridAutoRows: `${rowHeight}px` } : {}),
  };
  return (
    <div className={`grid gap-3 ${className ?? ""}`} style={style}>
      {children}
    </div>
  );
}
