import type { ReactNode } from "react";

/** One "n" a caption states: `{ value: 1234, unit: "cells" }` reads "1,234 cells";
 *  with `of`, "1,234 of 13,836 cells". */
export interface CaptionN {
  value: number;
  unit: string;
  of?: number;
}

/** "1,234 cells", "1,234 of 13,836 cells"; several joined with "; ". */
export function nText(n: CaptionN | CaptionN[]): string {
  const parts = Array.isArray(n) ? n : [n];
  return parts
    .map((p) => `${p.value.toLocaleString()}${p.of !== undefined ? ` of ${p.of.toLocaleString()}` : ""} ${p.unit}`)
    .join("; ");
}

/**
 * Honest label for a capped table or list: names the ordering and the total, so a
 * reader never takes "the first 200" for "all". Returns null when nothing was cut.
 *
 *   capText({ shown: 200, total: 2311, by: "adjusted p" })
 *   -> "Top 200 of 2,311 by adjusted p."
 *   capText({ shown: 200, total: 2311, by: "adjusted p", rest: "the CSV has all" })
 *   -> "Top 200 of 2,311 by adjusted p; the CSV has all."
 */
export function capText({
  shown,
  total,
  by,
  rest,
}: {
  shown: number;
  total: number;
  /** The ordering that chose the rows ("adjusted p", "|log2FC|"). */
  by: string;
  /** Where the others are ("the CSV has all"). */
  rest?: string;
}): string | null {
  if (total <= shown) return null;
  return `Top ${shown.toLocaleString()} of ${total.toLocaleString()} by ${by}${rest ? `; ${rest}` : ""}.`;
}

/**
 * A figure or table caption: one plain sentence saying what is shown, with the n.
 * 11px slate-500, like the captions already under the app's plots.
 */
export function Caption({
  children,
  n,
  className = "",
  testId,
}: {
  children?: ReactNode;
  /** Appended as "n = 1,234 cells" (several: "n = 1,234 cells; 20 genes"). */
  n?: CaptionN | CaptionN[];
  className?: string;
  testId?: string;
}) {
  return (
    <p className={`text-[11px] leading-snug text-slate-500 ${className}`} data-testid={testId}>
      {children}
      {n !== undefined && (
        <>
          {children ? " " : ""}
          <span className="whitespace-nowrap">n = {nText(n)}.</span>
        </>
      )}
    </p>
  );
}

/** The cap line under a capped table; renders nothing when every row is shown. */
export function CapNote(props: Parameters<typeof capText>[0] & { className?: string }) {
  const text = capText(props);
  if (!text) return null;
  return (
    <p className={`text-[11px] leading-snug text-slate-500 ${props.className ?? ""}`} data-testid="cap-note">
      {text}
    </p>
  );
}
