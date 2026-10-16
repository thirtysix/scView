import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Download, Loader2, Network } from "lucide-react";
import type { DEResponse } from "@/api/de";
import {
  enrichGeneLists,
  type EnrichmentResult,
  type GeneListEnrichmentResponse,
} from "@/api/enrichment";
import { collectionLabel } from "@/api/genesets";
import { InfoTip } from "@/components/common/InfoTip";
import { MSigDBCollectionTree, DEFAULT_MSIGDB_COLLECTIONS } from "@/components/panels/MSigDBCollectionTree";
import { EnrichmentTermsTable } from "@/components/unified/EnrichmentTermsTable";
import { useEnrichmentSort } from "@/hooks/useEnrichmentSort";
import { useGeneSetScoring } from "@/hooks/useGeneSetScoring";
import { downloadCsv } from "@/lib/csv";
import { deEnrichmentQuery } from "@/lib/de";
import { useViewStore } from "@/stores/viewStore";

type Side = "up" | "down";

/** The cut-offs a run used, to tell when the table has moved on since. */
interface RanWith {
  pThreshold: number;
  fcThreshold: number;
}

interface Props {
  datasetId: string;
  result: DEResponse;
  /** Display names of the two sides: up = higher in `labelA`. */
  labelA: string;
  labelB: string;
  /** The DE table's significance cut-offs; the gene lists use exactly these. */
  pThreshold: number;
  fcThreshold: number;
  onScoreGeneSet: (scores: Float32Array, name: string) => void;
  onGeneClick: (gene: string) => void;
}

/**
 * Over-representation of a DE result's significant genes: the up genes and,
 * separately, the down genes against the chosen collections, with every tested
 * gene as the background. Lives under the DE table so the cut-offs it uses are
 * the ones on screen.
 */
export function DEEnrichment({
  datasetId,
  result,
  labelA,
  labelB,
  pThreshold,
  fcThreshold,
  onScoreGeneSet,
  onGeneClick,
}: Props) {
  const askCopilot = useViewStore((s) => s.askCopilot);
  const [collections, setCollections] = useState<Set<string>>(
    new Set(DEFAULT_MSIGDB_COLLECTIONS),
  );
  const [excludeRiboMito, setExcludeRiboMito] = useState(true);
  const [showSets, setShowSets] = useState(false);
  const [data, setData] = useState<GeneListEnrichmentResponse | null>(null);
  const [ranWith, setRanWith] = useState<RanWith | null>(null);
  const [side, setSide] = useState<Side>("up");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const query = useMemo(
    () => deEnrichmentQuery(result, pThreshold, fcThreshold),
    [result, pThreshold, fcThreshold],
  );

  // A new DE result makes the enrichment of the old one meaningless.
  useEffect(() => {
    setData(null);
    setRanWith(null);
    setError(null);
  }, [result]);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await enrichGeneLists(datasetId, {
        ...query,
        collections: Array.from(collections),
        exclude_ribo_mito: excludeRiboMito,
      });
      setData(res);
      setRanWith({ pThreshold, fcThreshold });
      setSide(res.up.n_genes_used > 0 || res.down.n_genes_used === 0 ? "up" : "down");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [datasetId, query, collections, excludeRiboMito, pThreshold, fcThreshold]);

  const { scoreGenes, scoringName } = useGeneSetScoring(datasetId, onScoreGeneSet, onGeneClick);
  const shown = data ? data[side].results : [];
  const { sorted, sortField, sortDir, onSort } = useEnrichmentSort(shown);

  const higherIn = useCallback((s: Side) => (s === "up" ? labelA : labelB), [labelA, labelB]);

  const askAboutTerm = useCallback(
    (r: EnrichmentResult) => {
      askCopilot(
        `In a differential-expression comparison of ${labelA} vs ${labelB}, the genes higher in ` +
          `${higherIn(side)} are enriched for "${r.term}" (${collectionLabel(r)}, adjusted p ` +
          `${r.adjusted_pvalue.toExponential(2)}, ${r.overlap_count} of its ${r.gene_count} genes). ` +
          `Genes driving it: ${r.genes.slice(0, 12).join(", ")}. What does this suggest biologically?`,
      );
    },
    [askCopilot, labelA, labelB, side, higherIn],
  );

  const exportCsv = useCallback(() => {
    if (!data) return;
    const rows = (["up", "down"] as const).flatMap((s) =>
      data[s].results.map((r) => [
        `higher in ${higherIn(s)}`,
        r.term,
        collectionLabel(r),
        r.pvalue.toExponential(4),
        r.adjusted_pvalue.toExponential(4),
        `${r.overlap_count}/${r.gene_count}`,
        r.genes.join(";"),
      ]),
    );
    downloadCsv(
      `de_enrichment_${result.label}_${datasetId}.csv`,
      ["direction", "term", "collection", "p_value", "adj_pvalue", "overlap", "genes"],
      rows,
    );
  }, [data, result.label, datasetId, higherIn]);

  const stale =
    ranWith !== null &&
    (ranWith.pThreshold !== pThreshold || ranWith.fcThreshold !== fcThreshold);
  const nothingToTest = query.up.length === 0 && query.down.length === 0;
  const fmt = (n: number) => n.toLocaleString();

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-slate-200 p-2.5 text-xs">
      <div className="flex items-center gap-2">
        <Network className="h-3.5 w-3.5 flex-shrink-0 text-slate-400" />
        <span className="font-semibold text-slate-700">Pathway enrichment</span>
        <InfoTip width={300}>
          Over-representation (hypergeometric test, Benjamini-Hochberg adjusted) of the significant
          genes in the table above: the genes higher in {labelA} and, separately, those higher in{" "}
          {labelB}, so two opposing programmes are not blurred into one. The background is every
          gene this DE tested, not every gene the collections name.
        </InfoTip>
        <span className="flex-1 truncate text-slate-500">
          {fmt(query.up.length)} up · {fmt(query.down.length)} down
        </span>
        <button
          onClick={run}
          disabled={loading || nothingToTest || collections.size === 0}
          title={nothingToTest ? "No genes pass the thresholds" : "Enrich the up and down genes"}
          className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
        >
          {loading && <Loader2 className="h-3 w-3 animate-spin" />}
          Enrich
        </button>
      </div>

      <button
        onClick={() => setShowSets((v) => !v)}
        aria-expanded={showSets}
        className="flex items-center gap-1 self-start text-[11px] text-slate-500 hover:text-slate-700"
      >
        {showSets ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        Gene sets: {collections.size} collection{collections.size === 1 ? "" : "s"}
        {excludeRiboMito ? ", ribosomal / mito genes left out" : ""}
      </button>
      {showSets && (
        <div className="flex flex-col gap-2">
          <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
            <input
              type="checkbox"
              checked={excludeRiboMito}
              onChange={(e) => setExcludeRiboMito(e.target.checked)}
              className="h-3 w-3 rounded border-slate-300"
            />
            Exclude ribosomal / mitochondrial genes (from the gene lists, not the background)
          </label>
          <MSigDBCollectionTree
            selected={collections}
            onChange={setCollections}
            datasetId={datasetId}
            compact
          />
        </div>
      )}

      {error && (
        <div className="rounded border border-red-200 bg-red-50 p-2 text-red-700">{error}</div>
      )}

      {data && ranWith && (
        <div className={`flex flex-col gap-2 ${stale ? "opacity-60" : ""}`}>
          <p className="text-[10px] leading-snug text-slate-500">
            Genes with adj. p &le; {ranWith.pThreshold} and |log2FC| &ge;{" "}
            {ranWith.fcThreshold.toFixed(2)}: {fmt(data.up.n_genes_used)} higher in {labelA},{" "}
            {fmt(data.down.n_genes_used)} higher in {labelB}
            {data.up.n_excluded_ribo_mito + data.down.n_excluded_ribo_mito > 0 &&
              ` (${fmt(data.up.n_excluded_ribo_mito + data.down.n_excluded_ribo_mito)} ribosomal / mito left out)`}
            . Background: the {fmt(data.background_n)} genes tested. {data.collections.length}{" "}
            collection{data.collections.length === 1 ? "" : "s"}.
          </p>
          {data.warning && (
            <p role="note" className="rounded border border-amber-200 bg-amber-50 px-1.5 py-1 text-amber-800">
              {data.warning}
            </p>
          )}
          {stale && (
            <p role="status" className="rounded border border-amber-200 bg-amber-50 px-1.5 py-1 text-amber-800">
              The table&apos;s cut-off is now |log2FC| &ge; {fcThreshold.toFixed(2)}; this enrichment
              used {ranWith.fcThreshold.toFixed(2)}. Enrich again to match.
            </p>
          )}

          <div className="flex items-center gap-2">
            <div className="flex min-w-0 flex-1 rounded-md border border-slate-200 p-0.5">
              {(["up", "down"] as const).map((s) => {
                const n = data[s].n_significant;
                return (
                  <button
                    key={s}
                    onClick={() => setSide(s)}
                    title="Terms with adjusted p < 0.05"
                    className={`flex-1 truncate rounded px-2 py-1 transition ${
                      side === s ? "bg-primary font-medium text-white" : "text-slate-500 hover:text-slate-700"
                    }`}
                  >
                    Higher in {higherIn(s)} · {n} term{n === 1 ? "" : "s"}
                  </button>
                );
              })}
            </div>
            <button
              onClick={exportCsv}
              title="Export both directions as CSV"
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50"
            >
              <Download className="h-3 w-3" />
              CSV
            </button>
          </div>

          {data[side].n_genes_used === 0 ? (
            <p className="py-2 text-center text-slate-400">
              No significant genes higher in {higherIn(side)} to test.
            </p>
          ) : sorted.length === 0 ? (
            <p className="py-2 text-center text-slate-400">
              No gene set shares a gene with these {fmt(data[side].n_genes_used)} genes.
            </p>
          ) : (
            <>
              {data[side].n_terms > data[side].results.length && (
                <p className="text-[10px] text-slate-400">
                  The {fmt(data[side].results.length)} most significant of{" "}
                  {fmt(data[side].n_terms)} gene sets sharing a gene with these genes.
                </p>
              )}
              <EnrichmentTermsTable
                results={sorted}
                sortField={sortField}
                sortDir={sortDir}
                onSort={onSort}
                onTermClick={(r) => scoreGenes(r.genes, r.term)}
                onAsk={askAboutTerm}
                scoringTerm={scoringName}
                className="max-h-72"
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}
