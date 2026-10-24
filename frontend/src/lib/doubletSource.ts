import {
  DOUBLET_SOURCES,
  type DoubletAgreement,
  type DoubletChoices,
  type DoubletSource,
  type DoubletSourceInfo,
} from "@/api/doubletSource";

const pct = (f: number) => `${(100 * f).toFixed(1)}%`;

/** A step param value as a doublet source; anything else is "auto". */
export function asDoubletSource(v: unknown): DoubletSource {
  return DOUBLET_SOURCES.includes(v as DoubletSource) ? (v as DoubletSource) : "auto";
}

/** The source choice as the filter control lists it; auto names what it stands for here. */
export function sourceOptionLabel(source: DoubletSource, choices: DoubletChoices): string {
  const n = choices.n_by_source[source];
  const count = n == null ? "" : ` (${n.toLocaleString()})`;
  switch (source) {
    case "auto":
      return `Auto: ${choices.auto.label}${count}`;
    case "hashing":
      return `Cell hashing${count}`;
    case "scrublet":
      return `Scrublet${count}`;
    case "either":
      return `Either: hashing or Scrublet${count}`;
  }
}

/** "12.0% doublets by cell hashing; Scrublet flags 7.0%, 4.0% by both." */
export function sourceText(info: DoubletSourceInfo | null | undefined): string {
  if (!info) return "";
  let text = `${pct(info.fraction)} doublets by ${info.label}`;
  if (info.both && info.scrublet && info.source === "hashing") {
    text += `; Scrublet flags ${pct(info.scrublet.fraction)}, ${pct(info.both.fraction)} by both`;
  }
  return `${text}.`;
}

/** Under the filter's doublet control: what the choice drops here. */
export function choiceText(c: DoubletChoices): string {
  if (!c.used) {
    return c.source === "hashing"
      ? "No cell hashing calls in this dataset: nothing to drop by hashing."
      : "No doublet calls of that kind in this dataset: nothing to drop.";
  }
  const n = c.used.n.toLocaleString();
  const both = c.both?.toLocaleString();
  const kept = c.unscored
    ? ` ${c.unscored.toLocaleString()} cells Scrublet did not score (channels too small for it) are kept.`
    : "";
  if (c.used.source === "either") {
    return `Drops ${n} cells that cell hashing or Scrublet calls doublets${both ? ` (${both} by both)` : ""}.${kept}`;
  }
  const other = c.used.source === "hashing" ? "Scrublet" : c.used.source === "scrublet" ? "cell hashing" : null;
  return `Drops ${n} doublets by ${c.used.label}${other && both ? `; ${other} calls ${both} of them too` : ""}.${kept}`;
}

/** Under the doublet card's agreement table: what the two calls share and why they differ. */
export function agreementText(a: DoubletAgreement, scrubletAt: string): string {
  const share = a.hashing ? ` (${pct(a.both / a.hashing)})` : "";
  const neg = a.negatives_scrublet
    ? `; ${a.negatives_scrublet.toLocaleString()} of those are hashing negatives (no tag)`
    : "";
  return (
    `Of ${a.hashing.toLocaleString()} hashing doublets, ${scrubletAt} also calls ${a.both.toLocaleString()}${share}. ` +
    `It calls ${a.scrublet_only.toLocaleString()} cells that hashing does not: doublets of two cells from one ` +
    `sample, which hashing cannot see, or false calls${neg}.`
  );
}
