/**
 * The first line of a scatter's hover tooltip: what the plot is coloured by
 * and the hovered cell's value of it (its category with the category's colour,
 * or the number).
 */
import { effectiveColorType } from "@/lib/splitView";

type RGB = readonly [number, number, number];

export interface HoverLine {
  /** What the plot is coloured by: a column label, a gene or a score */
  label: string;
  /** The cell's value, formatted */
  value: string;
  /** The category's colour (categorical colouring only) */
  color: RGB | null;
}

/** A number as the tooltip shows it; NaN (no value) as "N/A". */
export function formatHoverNumber(v: number): string {
  if (!Number.isFinite(v)) return "N/A";
  if (Number.isInteger(v)) return v.toLocaleString();
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (a < 0.001) return v.toExponential(2);
  return v.toFixed(3);
}

export interface ColorByLineOptions {
  label: string;
  /** Per-cell colour values: category codes, or numbers */
  values: ArrayLike<number> | null;
  /** Original cell index */
  index: number;
  type: "categorical" | "continuous";
  /** Category names in code order */
  categories?: readonly string[] | null;
  /** Colour of a category, as the plot draws it */
  colorOf?: ((code: number, category: string) => RGB) | null;
  /** The cell's value from elsewhere (its metadata), for a code the
   *  categories do not name (a column over the backend's 100-value cap) */
  fallback?: unknown;
}

/** The colour-by line for one cell, or null when there is nothing to show. */
export function colorByLine(opts: ColorByLineOptions): HoverLine | null {
  const { label, values, index, categories, colorOf, fallback } = opts;
  if (!values || index < 0 || index >= values.length) return null;
  const v = values[index]!;
  const type = effectiveColorType(values as Float32Array | Int32Array, opts.type);
  if (type === "continuous") return { label, value: formatHoverNumber(v), color: null };

  if (v < 0 || !Number.isInteger(v)) return { label, value: "N/A", color: null };
  const category = categories?.[v];
  if (category !== undefined) {
    return { label, value: category, color: colorOf ? colorOf(v, category) : null };
  }
  if (fallback !== undefined && fallback !== null) return { label, value: String(fallback), color: null };
  return null;
}
