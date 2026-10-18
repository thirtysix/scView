import { useMemo, useState, type MouseEvent } from "react";
import { ArrowUpDown, ChevronDown, ChevronUp, GripVertical, Loader2, Pencil, Sparkles, Split } from "lucide-react";
import { colormapColor } from "@/lib/colors";
import { moveCategory } from "@/lib/categoryOrder";
import { useSettingsStore } from "@/stores/settingsStore";

/** Ctrl, Cmd or Shift held: a click adds to (or takes from) the picked categories. */
function isToggleClick(e: MouseEvent): boolean {
  return e.ctrlKey || e.metaKey || e.shiftKey;
}

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
  /** When provided, each category gets a "split into sub-clusters" button */
  onSubcluster?: (category: string) => void;
  /** The category being split right now (its button shows a spinner) */
  splittingCategory?: string | null;
  /** Ctrl/Cmd/Shift-click adds or removes a category from the picked set
   *  (without it, such a click is a plain click) */
  onCategoryToggle?: (category: string) => void;
  /** Categories picked together; the others are dimmed */
  pickedCategories?: string[];
  /** With picked categories: make their cells the selection */
  onSelectPicked?: () => void;
  /** With picked categories: drop the picks */
  onClearPicked?: () => void;
  /** Cells in the picked categories, shown beside the actions */
  pickedCellCount?: number;
  /** When provided, the legend offers an "order" mode: drag or move categories,
   *  then this gets the full new order */
  onReorder?: (order: string[]) => void;
  /** In order mode: a quick sort, "natural" (A-Z, numbers by value) or "size" */
  onSortCategories?: (sort: "natural" | "size") => void;
  /** An order change is being saved */
  orderBusy?: boolean;
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
  onSubcluster,
  splittingCategory,
  onCategoryToggle,
  pickedCategories,
  onSelectPicked,
  onClearPicked,
  pickedCellCount,
  onReorder,
  onSortCategories,
  orderBusy = false,
}: ColorLegendProps) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [ordering, setOrdering] = useState(false);
  const [dragFrom, setDragFrom] = useState<number | null>(null);
  const picked = pickedCategories ?? [];
  const commitEdit = (cat: string) => {
    const v = draft.trim();
    setEditing(null);
    if (v && v !== cat) onRenameCategory?.(cat, v);
  };
  const colormap = useSettingsStore((s) => s.continuousColormap);
  // Build gradient CSS for continuous legend
  const gradientStyle = useMemo(() => {
    if (type !== "continuous") return undefined;
    const stops: string[] = [];
    const nStops = 64;
    for (let i = 0; i <= nStops; i++) {
      const t = i / nStops;
      const [r, g, b] = colormapColor(colormap, t);
      stops.push(`rgb(${r},${g},${b}) ${(t * 100).toFixed(1)}%`);
    }
    return {
      background: `linear-gradient(to right, ${stops.join(", ")})`,
    };
  }, [type, colormap]);

  if (type === "categorical" && categories && categories.length > 0) {
    const move = (from: number, to: number) => {
      const next = moveCategory(categories, from, to);
      if (next !== categories) onReorder?.(next);
    };
    const orderButton = "rounded px-1 text-[10px] font-medium text-slate-500 hover:bg-slate-100 hover:text-primary disabled:opacity-40";
    return (
      <div className="flex flex-col gap-0.5">
        {(label || onReorder) && (
          <div className="mb-1 flex items-center justify-between gap-1 text-xs font-medium text-slate-600">
            <span className="min-w-0 truncate">{label}</span>
            {onReorder && !ordering && (
              <button
                type="button"
                onClick={() => setOrdering(true)}
                title="Reorder categories (saved for every plot)"
                aria-label="Reorder categories"
                className="flex-shrink-0 text-slate-300 transition-colors hover:text-primary"
              >
                <ArrowUpDown className="h-3 w-3" />
              </button>
            )}
          </div>
        )}
        {ordering && (
          <div className="mb-1 flex flex-wrap items-center gap-1" role="group" aria-label="Category order">
            {orderBusy && <Loader2 className="h-3 w-3 animate-spin text-primary" />}
            {onSortCategories && (
              <>
                <button type="button" disabled={orderBusy} onClick={() => onSortCategories("natural")}
                  title="Sort A-Z, numbers by value (2 before 10)" className={orderButton}>
                  A–Z
                </button>
                <button type="button" disabled={orderBusy} onClick={() => onSortCategories("size")}
                  title="Sort by number of cells, largest first" className={orderButton}>
                  By size
                </button>
              </>
            )}
            <button type="button" onClick={() => setOrdering(false)} className={`${orderButton} ml-auto text-primary`}>
              Done
            </button>
            <p className="w-full text-[10px] font-normal text-slate-400">
              Drag or use the arrows. Colours follow the order.
            </p>
          </div>
        )}
        <div className="max-h-60 overflow-y-auto pr-1">
          {categories.map((cat, i) => {
            const color = categoryColors?.[i] ?? [127, 127, 127];
            const isHighlighted = highlightedCategory === cat;
            const isPicked = picked.includes(cat);
            const isDimmed =
              picked.length > 0 ? !isPicked : highlightedCategory != null && !isHighlighted;

            const isEditing = editing === cat;
            if (ordering) {
              return (
                <div
                  key={cat}
                  draggable={!orderBusy}
                  onDragStart={(e) => {
                    setDragFrom(i);
                    e.dataTransfer.effectAllowed = "move";
                  }}
                  onDragOver={(e) => {
                    if (dragFrom !== null) e.preventDefault();
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (dragFrom !== null) move(dragFrom, i);
                    setDragFrom(null);
                  }}
                  onDragEnd={() => setDragFrom(null)}
                  data-testid="legend-order-row"
                  className={`flex w-full cursor-grab items-center gap-1 rounded px-1 py-0.5 text-xs hover:bg-slate-100 ${
                    dragFrom === i ? "opacity-40" : ""
                  }`}
                >
                  <GripVertical className="h-3 w-3 flex-shrink-0 text-slate-300" />
                  <span
                    className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full"
                    style={{ backgroundColor: `rgb(${color[0]},${color[1]},${color[2]})` }}
                  />
                  <span className="min-w-0 flex-1 truncate text-slate-700">{cat}</span>
                  <button type="button" disabled={orderBusy || i === 0} onClick={() => move(i, i - 1)}
                    aria-label={`Move ${cat} up`} className="flex-shrink-0 text-slate-400 hover:text-primary disabled:opacity-30">
                    <ChevronUp className="h-3 w-3" />
                  </button>
                  <button type="button" disabled={orderBusy || i === categories.length - 1} onClick={() => move(i, i + 1)}
                    aria-label={`Move ${cat} down`} className="flex-shrink-0 text-slate-400 hover:text-primary disabled:opacity-30">
                    <ChevronDown className="h-3 w-3" />
                  </button>
                </div>
              );
            }
            return (
              <div
                key={cat}
                className={`group flex w-full items-center gap-2 rounded px-1.5 py-0.5 text-xs transition-opacity ${
                  isDimmed ? "opacity-40" : "opacity-100"
                } ${isPicked ? "bg-slate-100 font-medium" : ""} hover:bg-slate-100`}
              >
                <span
                  className={`inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full ${isPicked ? "ring-2 ring-primary/40" : ""}`}
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
                      onMouseDown={(e) => {
                        if (e.shiftKey) e.preventDefault(); // no text selection on shift-click
                      }}
                      onClick={(e) => {
                        if (onCategoryToggle && isToggleClick(e)) onCategoryToggle(cat);
                        else onCategoryClick?.(cat);
                      }}
                      aria-pressed={pickedCategories ? isPicked : undefined}
                      title={onCategoryToggle ? "Click: select this group. Ctrl/Cmd/Shift-click: pick several" : undefined}
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
                    {onSubcluster && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSubcluster(cat);
                        }}
                        disabled={!!splittingCategory}
                        title="Split into sub-clusters (on this group's own graph)"
                        aria-label={`Split ${cat} into sub-clusters`}
                        className={`flex-shrink-0 transition-opacity hover:text-primary ${
                          splittingCategory === cat
                            ? "text-primary opacity-100"
                            : "text-slate-300 opacity-0 group-hover:opacity-100"
                        }`}
                      >
                        {splittingCategory === cat ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          <Split className="h-3 w-3" />
                        )}
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
        {picked.length > 0 && !ordering && (
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 border-t border-slate-100 pt-1 text-[11px]">
            <span className="text-slate-500">
              {picked.length} group{picked.length === 1 ? "" : "s"}
              {pickedCellCount != null && ` · ${pickedCellCount.toLocaleString()} cells`}
            </span>
            {onSelectPicked && (
              <button type="button" onClick={onSelectPicked} className="font-medium text-primary hover:underline">
                Select these cells
              </button>
            )}
            {onClearPicked && (
              <button type="button" onClick={onClearPicked} title="Clear (Esc)" className="text-slate-400 hover:text-slate-700">
                Clear
              </button>
            )}
          </div>
        )}
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
