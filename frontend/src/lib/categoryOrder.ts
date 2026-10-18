/** `order` with the item at `from` moved to `to`; the same array when nothing moves. */
export function moveCategory(order: readonly string[], from: number, to: number): string[] {
  if (from === to || from < 0 || to < 0 || from >= order.length || to >= order.length) {
    return order as string[];
  }
  const next = [...order];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item!);
  return next;
}

/** The cells in any of the `picked` categories (`codes` index into `categories`), or
 *  null when none of `picked` is a category. */
export function cellsInCategories(
  codes: ArrayLike<number>,
  categories: readonly string[],
  picked: readonly string[],
): Set<number> | null {
  const wanted = new Set<number>();
  for (const p of picked) {
    const i = categories.indexOf(p);
    if (i >= 0) wanted.add(i);
  }
  if (wanted.size === 0) return null;
  const out = new Set<number>();
  for (let i = 0; i < codes.length; i++) if (wanted.has(codes[i]!)) out.add(i);
  return out;
}

/** `picked` with `category` added, or taken out when it is already there. */
export function togglePicked(picked: readonly string[], category: string): string[] {
  return picked.includes(category) ? picked.filter((c) => c !== category) : [...picked, category];
}
