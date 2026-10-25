import type { ReactNode } from "react";

export type AriaSort = "ascending" | "descending" | "none";

/** `aria-sort` of a column: its direction when the table is sorted by it, else "none". */
export function ariaSort<F extends string>(field: F, sortField: F, sortDir: "asc" | "desc"): AriaSort {
  if (field !== sortField) return "none";
  return sortDir === "asc" ? "ascending" : "descending";
}

interface SortableThProps {
  onSort: () => void;
  sort: AriaSort;
  /** Header tooltip (what the column means) */
  title?: string;
  /** Classes of the <th> (padding, alignment, colour) */
  className?: string;
  align?: "left" | "right";
  children: ReactNode;
}

/**
 * A sortable column header: the click target is a real button (Tab, Enter,
 * Space) and the <th> carries `aria-sort`, so a screen reader says which
 * column the table is sorted by, and which way.
 */
export function SortableTh({ onSort, sort, title, className = "", align = "left", children }: SortableThProps) {
  return (
    <th aria-sort={sort} title={title} className={className}>
      <button
        type="button"
        onClick={onSort}
        className={`flex w-full items-center gap-1 font-medium hover:text-slate-800 ${
          align === "right" ? "justify-end text-right" : "text-left"
        }`}
      >
        {children}
      </button>
    </th>
  );
}
