import { useState } from "react";
import { Loader2 } from "lucide-react";
import {
  checkScoreDepth,
  methodLabel,
  type DepthCheck,
  type DepthCorrelation,
  type DepthMetric,
  type ScoreMethod,
} from "@/api/genesetScore";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { errorMessage } from "@/lib/errorMessage";

const METRICS: { id: DepthMetric; label: string; title: string }[] = [
  { id: "total_counts", label: "Total counts", title: "Spearman's rho of the score with the cell's total counts (UMIs)" },
  { id: "n_genes", label: "Genes detected", title: "Spearman's rho of the score with the number of genes the cell expresses" },
];
const METHODS: ScoreMethod[] = ["score_genes", "aucell"];

function rho(v: number | null | undefined): string {
  return v == null ? "n/a" : v.toFixed(2);
}

function Cell({ c }: { c: DepthCorrelation | undefined }) {
  return (
    <td className="px-1.5 py-0.5 text-right tabular-nums">
      {rho(c?.within)} <span className="text-slate-400">/ {rho(c?.overall)}</span>
    </td>
  );
}

/** The depth check's numbers: each method's rho with each depth measure. */
export function DepthCheckTable({ check }: { check: DepthCheck }) {
  const metrics = METRICS.filter((m) => METHODS.some((x) => check.methods[x]?.[m.id]));
  const within = check.n_groups > 0;
  return (
    <table className="mt-1 text-[10px] text-slate-600" data-testid="depth-check-table">
      <thead>
        <tr className="text-slate-400">
          <th className="px-1.5 text-left font-normal" />
          {metrics.map((m) => (
            <th key={m.id} className="px-1.5 text-right font-normal" title={`${m.title}: ${within ? "median within groups / over all cells" : "over all cells"}`}>
              {m.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {METHODS.map((x) => (
          <tr key={x}>
            <td className="px-1.5 py-0.5 font-medium">{methodLabel(x)}</td>
            {metrics.map((m) => <Cell key={m.id} c={check.methods[x]?.[m.id]} />)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/**
 * "Which method follows depth on this data?": scores the set both ways and
 * correlates each score with the cells' depth (POST /genesets/score/compare). On
 * request, since it computes both scores.
 */
export function ScoreDepthCheck({ datasetId, genes }: { datasetId: string; genes: string[] }) {
  const [check, setCheck] = useState<DepthCheck | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setCheck(await checkScoreDepth(datasetId, genes));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const groups = check && check.n_groups > 0
    ? `the median of the rhos within each of the ${check.n_groups} groups of ${check.groupby} with ${check.min_group_cells}+ cells, so groups that differ in both depth and score do not pass for depth sensitivity`
    : "over all cells (no grouping to check within)";

  return (
    <div className="mt-2 border-t border-blue-100 pt-1.5" data-testid="score-depth-check">
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={run}
          disabled={busy || genes.length === 0}
          className="flex items-center gap-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[11px] text-slate-600 hover:bg-slate-50 disabled:opacity-50"
          title="Score the set with both methods and see which one follows sequencing depth on this dataset"
        >
          {busy && <Loader2 className="h-3 w-3 animate-spin" />}
          Check depth
        </button>
        <InfoTip width={300} label="How the depth check works">
          Scores the set with score_genes and with AUCell, then Spearman&apos;s rho of each score
          with the cell&apos;s total counts and with its number of detected genes: {groups}. The
          sentence quotes the measure either method follows most; |rho| of 0.3 or more reads
          &quot;follows depth&quot;, 0.1 to 0.3 &quot;weakly&quot;. A biological signal can also
          go with depth (activated cells make more RNA), so a correlation says the score and depth
          move together, not which causes which.
        </InfoTip>
      </div>
      {error && <p className="mt-1 text-[11px] text-red-600">{error}</p>}
      {check && (
        <div className="mt-1">
          <Caption n={{ value: check.n_cells, unit: "cells" }} testId="depth-check-sentence">
            {check.sentence}
          </Caption>
          <DepthCheckTable check={check} />
          <p className="mt-1 text-[10px] leading-snug text-slate-400">
            rho {check.n_groups > 0 ? `within ${check.groupby} groups / over all cells` : "over all cells"};
            the two methods agree at rho {rho(check.between?.within ?? check.between?.overall)}.
            {check.share_below_max_rank > 0 &&
              ` AUCell counts each cell's top ${check.max_rank.toLocaleString()} genes; ${Math.round(check.share_below_max_rank * 100)}% of cells detect fewer, so their lowest counted ranks are ties among undetected genes.`}
          </p>
        </div>
      )}
    </div>
  );
}
