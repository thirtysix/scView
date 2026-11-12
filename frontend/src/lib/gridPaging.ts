/**
 * Paging for the small-multiples grids (split by, compare colourings, the gene
 * feature grid): how many panels a page shows, which panels are on page k, and
 * which page keeps the same panels in view when the page size changes. Pure, so
 * the grids and their tests share one set of rules.
 *
 * Every page is laid out for the same number of panels (`layoutCount`), so a
 * short last page keeps the panel size, and one camera fits every page.
 */

export type PanelsPerPage = 4 | 6 | 9 | 12 | "all";

export const PANELS_PER_PAGE_OPTIONS: readonly PanelsPerPage[] = [4, 6, 9, 12, "all"];
/** Six panels a page: the grids' look before paging. */
export const DEFAULT_PANELS_PER_PAGE: PanelsPerPage = 6;

export function isPanelsPerPage(v: unknown): v is PanelsPerPage {
  return PANELS_PER_PAGE_OPTIONS.includes(v as PanelsPerPage);
}

/** Panels a page holds: `perPage`, or all `n` of them. */
function size(n: number, perPage: PanelsPerPage): number {
  return perPage === "all" ? Math.max(1, n) : perPage;
}

/** Pages needed for `n` panels (at least one). */
export function pageCount(n: number, perPage: PanelsPerPage): number {
  return Math.max(1, Math.ceil(Math.max(0, n) / size(n, perPage)));
}

/** `page` moved into 0 .. pageCount - 1 (a page past the end shows the last one). */
export function clampPage(page: number, n: number, perPage: PanelsPerPage): number {
  const last = pageCount(n, perPage) - 1;
  return Number.isFinite(page) ? Math.min(Math.max(0, Math.floor(page)), last) : 0;
}

/** The panels on `page` (clamped): indices start .. end - 1 of all `n`. */
export function pageRange(n: number, perPage: PanelsPerPage, page: number): { start: number; end: number } {
  const k = size(n, perPage);
  const start = clampPage(page, n, perPage) * k;
  return { start, end: Math.min(Math.max(0, n), start + k) };
}

/** Panels each page is laid out for: a full page, or all `n` when fewer. */
export function layoutCount(n: number, perPage: PanelsPerPage): number {
  return Math.min(Math.max(0, n), size(n, perPage));
}

/**
 * The page that shows the first panel of `page` after the page size changes
 * from `from` to `to`, so a new size keeps the panels in view.
 */
export function repage(page: number, from: PanelsPerPage, to: PanelsPerPage, n: number): number {
  const first = pageRange(n, from, page).start;
  return clampPage(Math.floor(first / size(n, to)), n, to);
}

/** "Page 2 of 4" (1-based for people). */
export function pageLabel(page: number, n: number, perPage: PanelsPerPage): string {
  return `Page ${clampPage(page, n, perPage) + 1} of ${pageCount(n, perPage)}`;
}

/** "Panels 7–12 of 24", or "" when every panel is on one page. */
export function pagePanelsText(n: number, perPage: PanelsPerPage, page: number): string {
  if (pageCount(n, perPage) <= 1) return "";
  const { start, end } = pageRange(n, perPage, page);
  return end - start === 1 ? `Panel ${start + 1} of ${n}` : `Panels ${start + 1}–${end} of ${n}`;
}
