import { useState, useCallback, useMemo } from "react";
import { Loader2, ChevronRight, Sparkles, Target } from "lucide-react";
import { InfoTip } from "@/components/common/InfoTip";
import { formatPValue } from "@/lib/formatting";
import { EnrichmentNetworkCanvas } from "@/components/unified/EnrichmentNetworkCanvas";
import type { EnrichmentNetworkResponse, NetworkNode } from "@/api/enrichmentNetwork";

/**
 * The collapsed view of an enrichment result: one row per program (a cluster of
 * redundant terms) rather than one row per term.
 *
 * A ranked term list is a poor summary of a redundant collection, because the
 * largest, vaguest parent terms carry the most statistical power and so crowd the
 * top with rephrasings of one signal. This shows what those rephrasings collapse to.
 */

// CPM resolution: a density threshold on similarity weights, so strictly between
// 0 and 1. Higher splits more. Not a modularity resolution; 2 or 4 are invalid.
const RESOLUTIONS = [0.2, 0.3, 0.4, 0.5, 0.7];

interface Props {
  data: EnrichmentNetworkResponse | null;
  isLoading: boolean;
  error: string | null;
  resolution: number;
  onResolutionChange: (r: number) => void;
  onScoreGenes: (genes: string[], name: string) => void;
  onAsk: (label: string, terms: string[], genes: string[]) => void;
  scoringName: string | null;
  showGraph: boolean;
  onToggleGraph: (v: boolean) => void;
}

export function EnrichmentPrograms({
  data, isLoading, error, resolution, onResolutionChange,
  onScoreGenes, onAsk, scoringName, showGraph, onToggleGraph,
}: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  const nodesByTerm = useMemo(() => {
    const m = new Map<string, NetworkNode>();
    for (const n of data?.nodes ?? []) m.set(n.term, n);
    return m;
  }, [data]);

  const clusterOrder = useMemo(() => (data?.clusters ?? []).map((c) => c.cluster), [data]);
  const labels = useMemo(() => {
    const m: Record<number, string> = {};
    for (const c of data?.clusters ?? []) m[c.cluster] = c.label;
    return m;
  }, [data]);

  // Clusters arrive ranked by significance, not size, so the widest bar is not
  // necessarily the first row.
  const maxTerms = Math.max(1, ...(data?.clusters ?? []).map((c) => c.n_terms));

  const handleNodeClick = useCallback(
    (n: NetworkNode) => onScoreGenes(n.genes, n.term),
    [onScoreGenes],
  );

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 p-3 text-[11px] text-slate-400">
        <Loader2 className="h-3 w-3 animate-spin" /> Building network...
      </div>
    );
  }
  if (error) {
    return (
      <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">{error}</div>
    );
  }
  if (!data) {
    return (
      <p className="p-2 text-[10px] text-slate-400">
        Compute enrichment to group the redundant terms into programs.
      </p>
    );
  }
  if (data.n_terms === 0) {
    return (
      <p className="p-2 text-[10px] text-slate-400">
        No terms passed FDR {data.params.fdr} and the {data.params.min_term_size}&ndash;
        {data.params.max_term_size} gene size filter
        {data.n_size_filtered > 0 && ` (${data.n_size_filtered} filtered by size)`}.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {/* Summary + the knobs that actually change the answer */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-md border border-slate-200 bg-slate-50/70 px-2 py-1.5">
        <span className="text-[11px] tabular-nums text-slate-600">
          <span className="font-semibold text-slate-800">{data.n_terms}</span> terms &rarr;{" "}
          <span className="font-semibold text-slate-800">{data.n_clusters}</span> programs
          <InfoTip width={300}>
            <strong className="text-slate-700">What you are looking at.</strong> Each program is a
            group of enriched terms that overlap so heavily in their genes that they are describing
            one signal. Terms are linked when their genes overlap (combined coefficient &ge;{" "}
            {data.params.min_similarity}), and Leiden community detection groups the linked ones.
            The program takes its name from its most significant member term.
          </InfoTip>
        </span>

        {data.n_size_filtered > 0 && (
          <span className="text-[10px] tabular-nums text-slate-400">
            {data.n_size_filtered} size-filtered
            <InfoTip width={300}>
              Terms outside {data.params.min_term_size}&ndash;{data.params.max_term_size} genes are
              excluded, the conventional bounds for over-representation analysis. GO terms range
              from 1 to nearly 2,000 genes, and the enormous vague ones ("homeostatic process")
              overlap almost everything, so leaving them in makes every program merge into a blob.
            </InfoTip>
          </span>
        )}

        {data.truncated > 0 && (
          <span className="text-[10px] tabular-nums text-amber-600">
            {data.truncated} terms not shown
            <InfoTip width={260}>
              Only the most significant terms are graphed. {data.truncated} further significant
              terms were dropped by the cap, so this view is not complete.
            </InfoTip>
          </span>
        )}

        <div className="ml-auto flex items-center gap-1.5">
          <label className="text-[10px] font-medium text-slate-500">Granularity</label>
          <InfoTip width={320}>
            <strong className="text-slate-700">This is a dial, not a discovery.</strong> Roughly,
            the minimum gene-set overlap a program has to hold together at. Higher splits the same
            terms into more, smaller programs; lower merges them. On our sample data the very same
            terms give 34 programs at 0.2 and 144 at 0.9. There is no correct value, so pick what
            reads well and report what you picked.
          </InfoTip>
          <select
            value={resolution}
            onChange={(e) => onResolutionChange(parseFloat(e.target.value))}
            className="rounded border border-slate-300 bg-white px-1.5 py-1 text-[11px] tabular-nums text-slate-700"
          >
            {RESOLUTIONS.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Graph */}
      <div className="rounded-md border border-slate-200 bg-white">
        <div className="flex items-center gap-1.5 border-b border-slate-100 px-2 py-1">
          <label className="flex cursor-pointer items-center gap-1.5 text-[10px] font-medium text-slate-500">
            <input
              type="checkbox"
              checked={showGraph}
              onChange={(e) => onToggleGraph(e.target.checked)}
              className="h-3 w-3 rounded border-slate-300"
            />
            Similarity graph
          </label>
          <InfoTip width={320}>
            <strong className="text-slate-700">This is the evidence, not the answer.</strong> Each
            dot is one enriched term, placed so that terms sharing genes sit near each other. The
            visible clumps are what the program list on the left is derived from. It is here to show
            the redundancy is real and structured; the list is the readable version. Hover a dot for
            its term, click to score its genes on the scatter. Only the four strongest programs take
            a colour, because more than four colours cannot be reliably told apart; the rest stay grey.
          </InfoTip>
          <span className="ml-auto text-[10px] tabular-nums text-slate-400">
            {data.n_edges.toLocaleString()} links
          </span>
        </div>
        {showGraph && (
          <div className="p-1">
            <EnrichmentNetworkCanvas
              nodes={data.nodes}
              edges={data.edges}
              clusterOrder={clusterOrder}
              labels={labels}
              onNodeClick={handleNodeClick}
            />
          </div>
        )}
      </div>

      {/* Programs */}
      <div className="max-h-[calc(100vh-560px)] min-h-[160px] overflow-auto rounded-md border border-slate-200">
        <div className="sticky top-0 z-10 flex items-center gap-1 border-b border-slate-200 bg-slate-50 px-2 py-1.5 text-[10px] font-medium text-slate-600">
          <span className="w-8 text-right">Terms</span>
          <InfoTip width={300}>
            <strong className="text-slate-700">Do not read this as importance.</strong> It counts
            how many redundant terms collapsed into the program, which mostly reflects how finely
            the ontology's curators subdivided that branch. Immune biology is annotated far more
            densely than most metabolism, so immune programs look big by default. That is why rows
            are ranked by adj.P rather than by this number.
          </InfoTip>
          <span className="flex-1 pl-1">Program</span>
          <span className="w-14 text-right">Adj.P</span>
        </div>

        <ul className="divide-y divide-slate-50">
          {data.clusters.map((c) => {
            const isOpen = expanded === c.cluster;
            const isScoring = scoringName === c.label;
            return (
              <li key={c.cluster}>
                <div
                  onClick={() => setExpanded(isOpen ? null : c.cluster)}
                  className={`group flex cursor-pointer items-center gap-1 px-2 py-1.5 transition-colors hover:bg-blue-50 ${isScoring ? "bg-blue-50/70" : ""}`}
                  title={`${c.n_terms} redundant term${c.n_terms === 1 ? "" : "s"} in this program. Click to list them.`}
                >
                  <span className="w-8 text-right text-[11px] tabular-nums text-slate-500">
                    {c.n_terms}
                  </span>
                  <ChevronRight
                    className={`h-3 w-3 flex-shrink-0 text-slate-300 transition-transform ${isOpen ? "rotate-90" : ""}`}
                  />
                  <span className="flex min-w-0 flex-1 items-center gap-1.5">
                    <span
                      className="h-2 flex-shrink-0 rounded-r-sm bg-blue-500"
                      style={{ width: `${Math.max(4, (c.n_terms / maxTerms) * 46)}px` }}
                      aria-hidden="true"
                    />
                    <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-slate-800">
                      {c.label}
                    </span>
                    <button
                      onClick={(e) => { e.stopPropagation(); onScoreGenes(c.genes, c.label); }}
                      title={`Score all ${c.genes.length} genes across this program on the scatter`}
                      className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary focus:opacity-100 group-hover:opacity-100"
                    >
                      {isScoring ? <Loader2 className="h-3 w-3 animate-spin text-blue-500" /> : <Target className="h-3 w-3" />}
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); onAsk(c.label, c.terms, c.genes); }}
                      title="Ask the co-pilot about this program"
                      className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary focus:opacity-100 group-hover:opacity-100"
                    >
                      <Sparkles className="h-3 w-3" />
                    </button>
                  </span>
                  <span className="w-14 text-right text-[10px] tabular-nums text-blue-600">
                    {formatPValue(c.best_adjusted_pvalue)}
                  </span>
                </div>

                {isOpen && (
                  <ul className="bg-slate-50/60 px-2 pb-1.5 pl-11">
                    <li className="py-1 text-[9px] uppercase tracking-wide text-slate-400">
                      {c.n_terms} term{c.n_terms === 1 ? "" : "s"} collapsed here &middot; click one to score it
                    </li>
                    {c.terms.map((t) => {
                      const n = nodesByTerm.get(t);
                      return (
                        <li
                          key={t}
                          onClick={() => n && onScoreGenes(n.genes, n.term)}
                          className="cursor-pointer truncate py-0.5 text-[10px] text-slate-600 hover:text-blue-600"
                          title={n ? `${t}\nadj.P ${n.adjusted_pvalue.toExponential(2)} · ${n.overlap_count}/${n.gene_count} genes` : t}
                        >
                          {t}
                        </li>
                      );
                    })}
                  </ul>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
