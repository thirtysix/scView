import { waffleSquares } from "@/lib/compositionBars";

export interface WaffleCategory {
  name: string;
  count: number;
  /** CSS colour (the category's stable colour). */
  color: string;
}

interface Props {
  categories: readonly WaffleCategory[];
  title?: string;
}

/**
 * One bar's composition as a 10 x 10 grid: each square is 1% of the cells,
 * filled row by row in the categories' order (lib/compositionBars waffleSquares).
 */
export function WaffleChart({ categories, title }: Props) {
  const squares = waffleSquares(categories.map((c) => c.count));
  const total = categories.reduce((a, c) => a + c.count, 0);
  const cells: { color: string; name: string }[] = [];
  categories.forEach((c, i) => {
    for (let k = 0; k < (squares[i] ?? 0); k++) cells.push({ color: c.color, name: c.name });
  });
  const shown = categories
    .map((c, i) => ({ ...c, squares: squares[i] ?? 0 }))
    .filter((c) => c.count > 0);

  return (
    <div className="flex flex-wrap items-start gap-4" data-testid="waffle">
      <div>
        {title && <div className="mb-1 text-xs font-semibold text-slate-700">{title}</div>}
        <div
          role="img"
          aria-label={`${title ?? "Composition"}: ${shown.map((c) => `${c.name} ${c.squares}%`).join(", ")}`}
          className="grid grid-cols-10 gap-0.5"
          style={{ width: 180 }}
        >
          {cells.map((c, i) => (
            <span
              key={i}
              data-testid="waffle-square"
              title={c.name}
              className="aspect-square rounded-[2px]"
              style={{ backgroundColor: c.color }}
            />
          ))}
        </div>
        <div className="mt-1 text-[11px] text-slate-400">1 square = 1% of {total.toLocaleString()} cells</div>
      </div>
      <ul className="min-w-0 space-y-0.5 text-xs text-slate-600">
        {shown.map((c) => (
          <li key={c.name} className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-[2px]" style={{ backgroundColor: c.color }} />
            <span className="truncate">{c.name}</span>
            <span className="ml-auto pl-2 tabular-nums text-slate-500">
              {total > 0 ? ((c.count / total) * 100).toFixed(1) : "0"}%
              {c.squares === 0 && " (under half a square)"}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
