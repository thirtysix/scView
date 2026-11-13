/**
 * The line under a drawn score that says where it was saved (lib/savedScores.ts), or why
 * it was not: as the Gene Sets tab says it for a scored set.
 */
export function SavedScoreNote({ column, note }: { column?: string | null; note?: string | null }) {
  if (column) {
    return (
      <p className="text-[10px] leading-snug text-slate-500" data-testid="score-saved">
        Saved as column {column}: Compare and Color offer it.
      </p>
    );
  }
  if (note) {
    return (
      <p className="text-[10px] leading-snug text-amber-600" data-testid="score-saved">
        {note}
      </p>
    );
  }
  return null;
}
