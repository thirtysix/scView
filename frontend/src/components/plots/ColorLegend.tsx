import { useMemo } from "react";
import { viridisColor } from "@/lib/colors";

interface ColorLegendProps {
  type: "categorical" | "continuous";
  /** Category labels (categorical mode) */
  categories?: string[];
  /** RGB triples per category (categorical mode) */
  categoryColors?: [number, number, number][];
  /** Minimum value (continuous mode) */
  min?: number;
  /** Maximum value (continuous mode) */
  max?: number;
  /** Axis / column label */
  label?: string;
  /** Fires when a category swatch is clicked */
  onCategoryClick?: (category: string) => void;
  /** Currently highlighted category (if any) */
  highlightedCategory?: string | null;
}

export function ColorLegend({
  type,
  categories,
  categoryColors,
  min,
  max,
  label,
  onCategoryClick,
  highlightedCategory,
}: ColorLegendProps) {
  // Build gradient CSS for continuous legend
  const gradientStyle = useMemo(() => {
    if (type !== "continuous") return undefined;
    const stops: string[] = [];
    const nStops = 64;
    for (let i = 0; i <= nStops; i++) {
      const t = i / nStops;
      const [r, g, b] = viridisColor(t);
      stops.push(`rgb(${r},${g},${b}) ${(t * 100).toFixed(1)}%`);
    }
    return {
      background: `linear-gradient(to right, ${stops.join(", ")})`,
    };
  }, [type]);

  if (type === "categorical" && categories && categories.length > 0) {
    return (
      <div className="flex flex-col gap-0.5">
        {label && (
          <div className="mb-1 text-xs font-medium text-slate-600">
            {label}
          </div>
        )}
        <div className="max-h-60 overflow-y-auto pr-1">
          {categories.map((cat, i) => {
            const color = categoryColors?.[i] ?? [127, 127, 127];
            const isHighlighted = highlightedCategory === cat;
            const isDimmed =
              highlightedCategory != null && !isHighlighted;

            return (
              <button
                key={cat}
                onClick={() => onCategoryClick?.(cat)}
                className={`flex w-full items-center gap-2 rounded px-1.5 py-0.5 text-left text-xs transition-opacity ${
                  isDimmed ? "opacity-40" : "opacity-100"
                } hover:bg-slate-100`}
              >
                <span
                  className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full"
                  style={{
                    backgroundColor: `rgb(${color[0]},${color[1]},${color[2]})`,
                  }}
                />
                <span className="truncate text-slate-700">{cat}</span>
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  if (type === "continuous") {
    return (
      <div className="flex flex-col gap-1">
        {label && (
          <div className="mb-0.5 text-xs font-medium text-slate-600">
            {label}
          </div>
        )}
        <div
          className="h-3 w-full rounded-sm"
          style={gradientStyle}
        />
        <div className="flex justify-between text-[10px] tabular-nums text-slate-500">
          <span>{min?.toFixed(2) ?? "0"}</span>
          <span>{max?.toFixed(2) ?? "1"}</span>
        </div>
      </div>
    );
  }

  return null;
}
