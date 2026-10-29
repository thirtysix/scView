import { ExternalLink, Network } from "lucide-react";
import { isReactomeId, outboundLabel, sourceKind } from "@/lib/pathwayDiagram";

/**
 * The pathway action of an enrichment row: a "Diagram" button for a Reactome set
 * (opens its diagram coloured by the DE result), an outbound link for a KEGG or
 * WikiPathways set (KEGG's licence rules out drawing its maps here). Nothing for
 * other collections.
 */
export function PathwaySourceAction({
  sourceId,
  sourceUrl,
  onDiagram,
}: {
  sourceId?: string;
  sourceUrl?: string;
  onDiagram?: () => void;
}) {
  if (isReactomeId(sourceId) && onDiagram) {
    return (
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDiagram();
        }}
        title={`Open the Reactome diagram of ${sourceId}, coloured by fold change`}
        className="inline-flex flex-shrink-0 items-center gap-0.5 rounded border border-slate-200 px-1 text-[10px] text-slate-500 hover:border-blue-300 hover:text-blue-600"
      >
        <Network className="h-2.5 w-2.5" />
        Diagram
      </button>
    );
  }
  const kind = sourceKind(sourceId);
  const label = kind === "kegg" || kind === "wikipathways" ? outboundLabel(sourceId, sourceUrl) : null;
  if (!label || !sourceUrl) return null;
  return (
    <a
      href={sourceUrl}
      target="_blank"
      rel="noopener noreferrer"
      onClick={(e) => e.stopPropagation()}
      title={`${label} (${sourceId})`}
      aria-label={`${label} (${sourceId})`}
      className="flex-shrink-0 text-slate-300 hover:text-blue-600"
    >
      <ExternalLink className="h-3 w-3" />
    </a>
  );
}
