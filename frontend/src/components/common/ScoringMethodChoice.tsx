import { SCORE_METHODS, type ScoreMethod } from "@/api/genesetScore";
import { InfoTip } from "@/components/common/InfoTip";
import { useScoringMethodStore } from "@/stores/scoringMethodStore";

/** What the two methods compute: the tooltip of every scoring control. */
export function ScoringMethodHelp() {
  return (
    <>
      <span className="block">
        <b>score_genes</b> (scanpy): the set&apos;s mean log expression in the cell minus that of
        control genes of similar average expression. A mean of expression levels, so it can move
        with how deeply the cell was sequenced.
      </span>
      <span className="mt-1 block">
        <b>AUCell</b> (Aibar et al. 2017): genes are ranked within each cell, and the score is how
        many of the set&apos;s genes sit among the cell&apos;s top-ranked genes (5% of the detected
        genes), weighted toward the top: the area under the recovery curve, 0 to 1. Only the order
        of genes inside the cell counts. Computed as decoupler does; ties, mostly between
        undetected genes in a shallow cell, are broken in one fixed random order.
      </span>
      <span className="mt-1 block">
        &quot;Check depth&quot; in the Gene Sets tab shows which of the two follows depth on this
        dataset.
      </span>
    </>
  );
}

/**
 * The gene-set scoring method, beside every control that scores a set (Gene Sets,
 * Enrichment, DE enrichment, Gene Lists). One choice for all of them, remembered
 * (stores/scoringMethodStore.ts).
 */
export function ScoringMethodChoice({ label = "Score with", className = "" }: { label?: string; className?: string }) {
  const method = useScoringMethodStore((s) => s.method);
  const setMethod = useScoringMethodStore((s) => s.setMethod);
  return (
    <span className={`inline-flex items-center gap-1 text-[11px] text-slate-500 ${className}`}>
      {label && <span>{label}</span>}
      <select
        value={method}
        onChange={(e) => setMethod(e.target.value as ScoreMethod)}
        aria-label="Gene-set scoring method"
        className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700"
      >
        {SCORE_METHODS.map((m) => (
          <option key={m.id} value={m.id}>{m.label}</option>
        ))}
      </select>
      <InfoTip width={300} label="How gene sets are scored">
        <ScoringMethodHelp />
      </InfoTip>
    </span>
  );
}
