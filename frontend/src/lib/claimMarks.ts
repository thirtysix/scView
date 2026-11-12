import type { Claim } from "@/api/assistant";

/** One checked sentence found in a block of an answer: [start, end) of the block text. */
export interface ClaimMarkSpan {
  start: number;
  end: number;
  claim: Claim;
}

/** Markdown emphasis and code marks in `s` all close: cutting there leaves no stray `**`. */
export function balancedInline(s: string): boolean {
  const count = (t: string, mark: string) => t.split(mark).length - 1;
  return (
    count(s, "**") % 2 === 0 &&
    count(s.replace(/\*\*/g, ""), "*") % 2 === 0 &&
    count(s, "`") % 2 === 0
  );
}

/**
 * Where the claims' sentences sit in one block of the answer (a paragraph, a list item).
 * Each claim carries its sentence as written (`span`, from the server's split), so this is
 * a plain search. `used` holds the ids already marked in earlier blocks: a sentence written
 * twice is marked once. A sentence whose cut would split bold, italic or code is left
 * unmarked (the ledger still lists it).
 */
export function findClaimMarks(text: string, claims: Claim[], used: Set<number>): ClaimMarkSpan[] {
  const found: ClaimMarkSpan[] = [];
  for (const c of claims) {
    if (!c.span || used.has(c.id)) continue;
    for (let i = text.indexOf(c.span); i >= 0; i = text.indexOf(c.span, i + 1)) {
      const end = i + c.span.length;
      if (found.some((f) => i < f.end && end > f.start)) continue;
      if (!balancedInline(text.slice(0, i)) || !balancedInline(c.span)) break;
      found.push({ start: i, end, claim: c });
      used.add(c.id);
      break;
    }
  }
  return found.sort((a, b) => a.start - b.start);
}
