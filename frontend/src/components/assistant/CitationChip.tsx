/** Renders a co-pilot citation tag (e.g. "lit:PMID:123", "result:markers:B")
 *  as a compact chip. Literature chips link to PubMed; result chips are
 *  clickable (handled by the parent, which jumps to the cluster/gene); others
 *  are subtle, non-interactive chips that de-noise the inline brackets. */

const base =
  "mx-0.5 inline-flex items-center rounded px-1 py-0 align-baseline text-[10px] font-medium";

function shortLabel(tag: string): string {
  const parts = tag.split(":");
  const kind = parts[0];
  if (kind === "lit" && parts[1] === "PMID") return `PMID ${parts[2] ?? ""}`;
  if (kind === "result") return parts.slice(2).join(":") || parts[1] || "result";
  if (kind === "doc") return "docs";
  if (kind === "app") return parts[1] ?? "app";
  if (kind === "dataset") return parts[1] ?? "dataset";
  if (kind === "preprocessing") return "QC state";
  if (kind === "provenance") return "steps";
  return tag;
}

export function CitationChip({
  tag,
  onClick,
}: {
  tag: string;
  onClick?: (tag: string) => void;
}) {
  const label = shortLabel(tag);

  // Literature → external PubMed link.
  const pmid = /^lit:PMID:(\d+)/.exec(tag);
  if (pmid) {
    return (
      <a
        href={`https://pubmed.ncbi.nlm.nih.gov/${pmid[1]}/`}
        target="_blank"
        rel="noopener noreferrer"
        title={`Open ${tag} on PubMed`}
        className={`${base} bg-emerald-50 text-emerald-700 hover:bg-emerald-100`}
      >
        {label}
      </a>
    );
  }

  // Result → clickable (parent navigates to the cluster/gene).
  if (tag.startsWith("result:") && onClick) {
    return (
      <button
        type="button"
        onClick={() => onClick(tag)}
        title={`Show ${tag} in the Unified View`}
        className={`${base} cursor-pointer bg-primary/10 text-primary hover:bg-primary/20`}
      >
        {label}
      </button>
    );
  }

  // Everything else → subtle, non-interactive.
  return (
    <span title={tag} className={`${base} bg-muted text-muted-foreground`}>
      {label}
    </span>
  );
}
