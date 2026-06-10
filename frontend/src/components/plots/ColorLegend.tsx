import { useMemo, useState } from "react";
import { Pencil, Sparkles } from "lucide-react";
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
  /** When provided, each category becomes inline-renamable (e.g. fix a cell-type label) */
  onRenameCategory?: (category: string, newName: string) => void;
  /** When provided, each category gets an "ask the co-pilot about this" button */
  onAskAbout?: (category: string) => void;
  /** When provided (continuous mode), an "ask about this" button sits by the label */
  onAsk?: () => void;
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
  onRenameCategory,
  onAskAbout,
  onAsk,
}: ColorLegendProps) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const commitEdit = (cat: string) => {
    const v = draft.trim();
    setEditing(null);
    if (v && v !== cat) onRenameCategory?.(cat, v);
  };
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

            const isEditing = editing === cat;
            return (
              <div
                key={cat}
                className={`group flex w-full items-center gap-2 rounded px-1.5 py-0.5 text-xs transition-opacity ${
                  isDimmed ? "opacity-40" : "opacity-100"
                } hover:bg-slate-100`}
              >
                <span
                  className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full"
                  style={{ backgroundColor: `rgb(${color[0]},${color[1]},${color[2]})` }}
                />
                {isEditing ? (
                  <input
                    autoFocus
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitEdit(cat);
                      else if (e.key === "Escape") setEditing(null);
                    }}
                    onBlur={() => commitEdit(cat)}
                    className="min-w-0 flex-1 rounded border border-primary px-1 py-0 text-xs focus:outline-none"
                  />
                ) : (
                  <>
                    <button
                      onClick={() => onCategoryClick?.(cat)}
                      className="min-w-0 flex-1 truncate text-left text-slate-700"
                    >
                      {cat}
                    </button>
                    {onAskAbout && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onAskAbout(cat);
                        }}
                        title="Ask the co-pilot about this"
                        className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary group-hover:opacity-100"
                      >
                        <Sparkles className="h-3 w-3" />
                      </button>
                    )}
                    {onRenameCategory && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDraft(cat);
                          setEditing(cat);
                        }}
                        title="Rename label"
                        className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary group-hover:opacity-100"
                      >
                        <Pencil className="h-3 w-3" />
                      </button>
                    )}
                  </>
                )}
              </div>
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
          <div className="mb-0.5 flex items-center justify-between gap-1">
            <span className="text-xs font-medium text-slate-600">{label}</span>
            {onAsk && (
              <button
                onClick={onAsk}
                title="Ask the co-pilot about this"
                className="flex-shrink-0 text-slate-300 transition-colors hover:text-primary"
              >
                <Sparkles className="h-3 w-3" />
              </button>
            )}
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
