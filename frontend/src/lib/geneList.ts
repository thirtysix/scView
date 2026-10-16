/** Most genes `GET /expression/by-group` accepts in one request. */
export const MAX_PLOT_GENES = 300;

/**
 * Gene names from typed or pasted text: separated by commas, semicolons,
 * whitespace or newlines (a column copied from a spreadsheet works), with
 * quotes stripped, each name once, in the order given.
 */
export function parseGeneList(text: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of text.split(/[\s,;]+/)) {
    const gene = raw.replace(/^["']+|["']+$/g, "").trim();
    if (!gene || seen.has(gene)) continue;
    seen.add(gene);
    out.push(gene);
  }
  return out;
}

/** `current` followed by the new genes of `added`, each once, capped at `max`. */
export function mergeGeneLists(current: string[], added: string[], max = MAX_PLOT_GENES): string[] {
  return parseGeneList([...current, ...added].join("\n")).slice(0, max);
}

/** Requested genes the response left out, in request order. */
export function missingGenes(requested: string[], returned: string[]): string[] {
  const found = new Set(returned);
  return requested.filter((g) => !found.has(g));
}
