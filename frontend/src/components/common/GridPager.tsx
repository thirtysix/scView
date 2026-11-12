import { ChevronLeft, ChevronRight } from "lucide-react";
import { Help } from "@/components/common/Help";
import {
  PANELS_PER_PAGE_OPTIONS,
  clampPage,
  isPanelsPerPage,
  pageCount,
  pageLabel,
  repage,
  type PanelsPerPage,
} from "@/lib/gridPaging";

interface GridPagerProps {
  /** Panels in the grid, on every page */
  n: number;
  perPage: PanelsPerPage;
  page: number;
  /** A new page size, with the page that keeps the current panels in view */
  onPerPageChange: (perPage: PanelsPerPage, page: number) => void;
  onPageChange: (page: number) => void;
}

const btn =
  "rounded p-0.5 text-slate-500 hover:bg-slate-100 hover:text-slate-700 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent";

/**
 * Header control of a small-multiples grid: panels per page (4 / 6 / 9 / 12 /
 * all) and, when there is more than one page, Prev / Next with "Page 2 of 4".
 * Hidden for a grid of four panels or fewer, which always fits one page.
 */
export function GridPager({ n, perPage, page, onPerPageChange, onPageChange }: GridPagerProps) {
  if (n <= (PANELS_PER_PAGE_OPTIONS[0] as number)) return null;
  const pages = pageCount(n, perPage);
  const current = clampPage(page, n, perPage);
  return (
    <div className="flex flex-shrink-0 items-center gap-1" data-testid="grid-pager">
      <label className="flex items-center gap-1">
        <Help id="plot.grid-pages" side="below" align="right">
          Per page
        </Help>
        <select
          aria-label="Panels per page"
          value={String(perPage)}
          onChange={(e) => {
            const v = e.target.value === "all" ? "all" : Number(e.target.value);
            if (isPanelsPerPage(v)) onPerPageChange(v, repage(current, perPage, v, n));
          }}
          className="rounded border border-slate-300 bg-white px-1 py-0 text-[11px] text-slate-700"
        >
          {PANELS_PER_PAGE_OPTIONS.map((o) => (
            <option key={o} value={String(o)}>
              {o === "all" ? `All (${n})` : o}
            </option>
          ))}
        </select>
      </label>
      {pages > 1 && (
        <>
          <button
            type="button"
            onClick={() => onPageChange(current - 1)}
            disabled={current === 0}
            aria-label="Previous page"
            title="Previous page"
            className={btn}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
          <span className="whitespace-nowrap tabular-nums" aria-live="polite">
            {pageLabel(current, n, perPage)}
          </span>
          <button
            type="button"
            onClick={() => onPageChange(current + 1)}
            disabled={current >= pages - 1}
            aria-label="Next page"
            title="Next page"
            className={btn}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </>
      )}
    </div>
  );
}
