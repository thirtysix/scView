/** A reference's labels as a spot composition run takes them (backend spot_deconvolution.label_edits +
 *  sd_edit_labels): the panel's live count, its reasons and the coarsen helper. */

export interface LabelCount {
  type: string;
  n_cells: number;
}

export interface LabelEdits {
  exclude: string[];
  /** label -> broader name, as typed */
  merge: Record<string, string>;
}

export const NO_EDITS: LabelEdits = { exclude: [], merge: {} };

/** The merges that do something: names trimmed, empty fields and a label merged into itself dropped. */
export function liveMerges(merge: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [label, name] of Object.entries(merge)) {
    const n = name.trim();
    if (n && n !== label) out[label] = n;
  }
  return out;
}

export function hasEdits(e: LabelEdits): boolean {
  return e.exclude.length > 0 || Object.keys(liveMerges(e.merge)).length > 0;
}

/** As the API takes them; a label left out is not also sent as merged (excluding wins). */
export function editsBody(e: LabelEdits): { exclude_labels: string[]; merge_labels: Record<string, string> } {
  const merge = liveMerges(e.merge);
  for (const label of e.exclude) delete merge[label];
  return { exclude_labels: [...e.exclude], merge_labels: merge };
}

/** Choices naming labels this column does not have dropped (a saved run's, after the reference changed). */
export function pruneEdits(e: LabelEdits, labels: LabelCount[]): LabelEdits {
  const known = new Set(labels.map((l) => l.type));
  const merge = Object.fromEntries(Object.entries(e.merge).filter(([label]) => known.has(label)));
  const names = new Set(Object.values(merge).map((n) => n.trim()));
  const exclude = e.exclude.filter((label) => known.has(label) || names.has(label));
  return exclude.length === e.exclude.length && Object.keys(merge).length === Object.keys(e.merge).length
    ? e
    : { exclude, merge };
}

export interface LabelPlan {
  /** Types with enough cells, largest first */
  types: LabelCount[];
  /** Under the minimum, so not used */
  small: LabelCount[];
  nOut: number;
  nMerged: number;
}

/** What the choices leave: the backend's rule (merge, then anything left out or merged into a left-out name goes). */
export function planLabels(labels: LabelCount[], e: LabelEdits, minCells: number): LabelPlan {
  const merge = liveMerges(e.merge);
  const gone = new Set(e.exclude);
  const counts = new Map<string, number>();
  let nOut = 0;
  let nMerged = 0;
  for (const { type, n_cells } of labels) {
    const name = merge[type] ?? type;
    if (gone.has(type) || gone.has(name)) {
      nOut += 1;
      continue;
    }
    if (name !== type) nMerged += 1;
    counts.set(name, (counts.get(name) ?? 0) + n_cells);
  }
  const all = [...counts].map(([type, n_cells]) => ({ type, n_cells })).sort((a, b) => b.n_cells - a.n_cells);
  return { types: all.filter((t) => t.n_cells >= minCells), small: all.filter((t) => t.n_cells < minCells), nOut, nMerged };
}

/** Why these choices cannot run, in a sentence; null when they can. */
export function planProblem(plan: LabelPlan, e: LabelEdits, maxTypes: number, minCells: number): string | null {
  const merge = liveMerges(e.merge);
  for (const [label, name] of Object.entries(merge)) {
    if (e.exclude.includes(label)) continue;
    const next = merge[name];
    if (next && !e.exclude.includes(name)) {
      return `${label} merges into ${name}, which itself merges into ${next}: merge both into the final name.`;
    }
  }
  const n = plan.types.length;
  if (n > maxTypes) return `${n} types, at most ${maxTypes}: merge or leave out at least ${n - maxTypes} more.`;
  if (n < 2) return `${n} ${n === 1 ? "type" : "types"} with ${minCells}+ cells: a run needs at least 2.`;
  return null;
}

export type CutAt = "number" | "last" | "first";

export const CUT_LABELS: Record<CutAt, string> = {
  number: "a trailing number only",
  last: "the last separator",
  first: "the first separator",
};

export const SEPARATORS = ["_", "-", ".", ":", "/", " "] as const;

export const separatorLabel = (s: string) => (s === " " ? "space" : s);

/** The separator most labels contain ("_" when none does). */
export function likelySeparator(labels: LabelCount[]): string {
  let best = "_";
  let most = 0;
  for (const s of SEPARATORS) {
    const n = labels.filter((l) => l.type.includes(s)).length;
    if (n > most) {
      best = s;
      most = n;
    }
  }
  return best;
}

/** A label's shared name: the part before the separator, at the chosen place; the label itself when it has none. */
export function sharedName(label: string, sep: string, cut: CutAt): string {
  if (cut === "number") {
    const at = label.lastIndexOf(sep);
    return at > 0 && /^\d+$/.test(label.slice(at + sep.length)) ? label.slice(0, at) : label;
  }
  const at = cut === "first" ? label.indexOf(sep) : label.lastIndexOf(sep);
  return at > 0 ? label.slice(0, at) : label;
}

export interface PrefixGroup {
  name: string;
  labels: string[];
  n_cells: number;
}

/** Labels (those not left out) sharing a name, the groups of two or more, largest first. */
export function prefixGroups(labels: LabelCount[], exclude: string[], sep: string, cut: CutAt): PrefixGroup[] {
  const groups = new Map<string, PrefixGroup>();
  for (const { type, n_cells } of labels) {
    if (exclude.includes(type)) continue;
    const name = sharedName(type, sep, cut);
    const g = groups.get(name) ?? { name, labels: [], n_cells: 0 };
    g.labels.push(type);
    g.n_cells += n_cells;
    groups.set(name, g);
  }
  return [...groups.values()].filter((g) => g.labels.length >= 2).sort((a, b) => b.n_cells - a.n_cells);
}

/** The merges of the ticked groups added to `merge`. */
export function withGroups(merge: Record<string, string>, groups: PrefixGroup[]): Record<string, string> {
  const out = { ...merge };
  for (const g of groups) {
    for (const label of g.labels) {
      if (label === g.name) delete out[label];
      else out[label] = g.name;
    }
  }
  return out;
}

/** The labels a run's type stands for: those merged into it, or itself. */
export function sourcesOf(type: string, merged: Record<string, string>, labels?: LabelCount[]): string[] {
  const from = Object.entries(merged)
    .filter(([, name]) => name === type)
    .map(([label]) => label);
  // A merged name that is also a label (it joined it) stands for that label's cells too.
  const isLabel = !labels || labels.some((l) => l.type === type);
  return from.length === 0 || isLabel ? [...from, type] : from;
}

/** The choices of a saved run with one of its types left out as well. */
export function withTypeOut(
  s: { excluded_labels?: string[]; merged_labels?: Record<string, string> },
  type: string,
  labels?: LabelCount[],
): LabelEdits {
  const merge = { ...(s.merged_labels ?? {}) };
  const sources = sourcesOf(type, merge, labels);
  for (const label of sources) delete merge[label];
  return { exclude: [...new Set([...(s.excluded_labels ?? []), ...sources])], merge };
}

/** A saved run's or self-check's left-out and merged labels in words, or null when it used them as they are
 *  (an older result records neither). `detail`: each merged name with its labels, for the tooltip. */
export function labelsUsed(s: {
  excluded_labels?: string[];
  merged_labels?: Record<string, string>;
  n_excluded_cells?: number;
}): { text: string; detail: string[] } | null {
  const out = s.excluded_labels ?? [];
  const merged = s.merged_labels ?? {};
  const groups = new Map<string, string[]>();
  for (const [label, name] of Object.entries(merged)) groups.set(name, [...(groups.get(name) ?? []), label]);
  const parts: string[] = [];
  if (out.length) {
    const cells = s.n_excluded_cells ? ` (${s.n_excluded_cells.toLocaleString()} cells)` : "";
    const names = out.length <= 8 ? out.join(", ") : `${out.slice(0, 8).join(", ")} and ${out.length - 8} more`;
    parts.push(`left out ${names}${cells}`);
  }
  if (groups.size) {
    const n = Object.keys(merged).length;
    parts.push(`${n} ${n === 1 ? "label" : "labels"} merged into ${groups.size}`);
  }
  if (!parts.length) return null;
  return { text: parts.join("; "), detail: [...groups].map(([name, ls]) => `${name} = ${ls.join(" + ")}`) };
}
