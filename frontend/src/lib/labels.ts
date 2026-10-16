/**
 * Hand labels for selected cells: the same rules the backend applies
 * (core/user_labels.py), checked as the user types so a bad name never needs a
 * round trip. The backend checks again.
 */

export const UNASSIGNED = "Unassigned";
export const LABEL_MAX = 80;
export const COLUMN_MAX = 64;

const COLUMN_RE = /^[A-Za-z][A-Za-z0-9 _.-]*$/;
// Control, invisible format (bidi, zero-width) and line/paragraph separators.
const INVISIBLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Zl}\p{Zp}]/u;

/** Why `label` can't be used, or null when it can. */
export function labelError(label: string): string | null {
  const v = label.trim();
  if (!v) return "Type a label for the selected cells.";
  if (v.length > LABEL_MAX) return `Label is too long (max ${LABEL_MAX} characters).`;
  if (INVISIBLE.test(v)) return "Label must not contain control or invisible characters.";
  return null;
}

/** Why `name` can't be a new column of a dataset with `existing` columns, or null. */
export function columnNameError(name: string, existing: readonly string[]): string | null {
  const v = name.trim();
  if (!v) return "Name the new column.";
  if (v.length > COLUMN_MAX) return `Column name is too long (max ${COLUMN_MAX} characters).`;
  if (!COLUMN_RE.test(v))
    return "Column name must start with a letter and use only letters, digits, spaces, '_', '-' or '.'.";
  if (v.toLowerCase().startsWith("scview_")) return "Names starting with 'scview_' are kept for scView's own columns.";
  const lower = v.toLowerCase();
  if (existing.some((c) => c.toLowerCase() === lower)) return `A column named '${v}' already exists.`;
  return null;
}

/** A free name for a new label column: manual_labels, manual_labels_2, ... */
export function suggestColumnName(existing: readonly string[]): string {
  const taken = new Set(existing.map((c) => c.toLowerCase()));
  for (let i = 1; ; i++) {
    const name = i === 1 ? "manual_labels" : `manual_labels_${i}`;
    if (!taken.has(name)) return name;
  }
}
