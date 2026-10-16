import { getPalette, type CategoricalPaletteName } from "@/lib/colors";

type RGB = readonly [number, number, number];

// One category -> colour mapping per (dataset, column), shared by every panel,
// so a cluster or cell type has the same colour in the scatter, its legend,
// the cluster map, violins and the Observations charts.
//
// A category's slot is its position in the column's category order from the
// backend (`obs_columns[].values`). That is also the code the embedding
// endpoint colours the scatter by, so the scatter and its legend (which colour
// by index into that list) are the reference and need no lookup. A category
// the order does not list (a column over the backend's 100-value cap, or a
// label only another endpoint returns) takes the next free slot the first time
// any panel asks for it, and keeps it while the order stays the same.

interface Entry {
  order: readonly string[];
  slots: Map<string, number>;
  next: number;
}

const registry = new Map<string, Entry>();

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function entryFor(datasetId: string, column: string, order: readonly string[]): Entry {
  const key = `${datasetId}\u0000${column}`;
  let entry = registry.get(key);
  // A changed order (e.g. a renamed category) starts the column afresh.
  if (!entry || !sameOrder(entry.order, order)) {
    const slots = new Map<string, number>();
    order.forEach((c, i) => {
      if (!slots.has(c)) slots.set(c, i);
    });
    entry = { order, slots, next: order.length };
    registry.set(key, entry);
  }
  return entry;
}

/** The palette slot of `category` in (datasetId, column); see the note above. */
export function categorySlot(
  datasetId: string,
  column: string,
  order: readonly string[],
  category: string,
): number {
  const entry = entryFor(datasetId, column, order);
  let slot = entry.slots.get(category);
  if (slot === undefined) {
    slot = entry.next++;
    entry.slots.set(category, slot);
  }
  return slot;
}

/** The colour of `category` in (datasetId, column) on the named palette. */
export function categoryColor(
  datasetId: string,
  column: string,
  order: readonly string[],
  category: string,
  paletteName?: CategoricalPaletteName,
): RGB {
  const palette = getPalette(paletteName);
  return palette[categorySlot(datasetId, column, order, category) % palette.length]!;
}

/** `names` sorted into the column's category order; names it does not list keep their order, last. */
export function sortByCategoryOrder(names: readonly string[], order: readonly string[]): string[] {
  const pos = new Map(order.map((c, i) => [c, i]));
  return names
    .map((name, i) => ({ name, key: pos.get(name) ?? order.length + i }))
    .sort((a, b) => a.key - b.key)
    .map((x) => x.name);
}

/** CSS colour string for an RGB triple, with optional alpha. */
export function rgbCss(c: RGB, alpha?: number): string {
  return alpha == null ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

/** Forget every mapping (tests). */
export function resetCategoryColors(): void {
  registry.clear();
}
