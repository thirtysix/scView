import { useState, useCallback, useMemo, useEffect } from "react";
import { Loader2, Download, ArrowUpDown, ArrowUp, ArrowDown, Sparkles } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { apiFetch } from "@/api/client";
import { MSigDBCollectionTree, DEFAULT_MSIGDB_COLLECTIONS } from "@/components/panels/MSigDBCollectionTree";
import { formatPValue } from "@/lib/formatting";
import { downloadCsv } from "@/lib/csv";
import { InfoTip } from "@/components/common/InfoTip";
import { EnrichmentPrograms } from "@/components/unified/EnrichmentPrograms";
import {
  fetchEnrichmentNetwork,
  type EnrichmentNetworkResponse,
} from "@/api/enrichmentNetwork";

interface EnrichmentResult {
  term: string;
  pvalue: number;
  adjusted_pvalue: number;
  overlap_count: number;
  gene_count: number;
  genes: string[];
  collection: string;
}

interface EnrichmentResponse {
  group: string;
  groupby: string;
  n_genes_used: number;
  results: EnrichmentResult[];
  source: string;
}

type SortField = "term" | "adj_pval" | "overlap";
type SortDir = "asc" | "desc";

interface UnifiedEnrichmentSubtabProps {
  onTermClick: (gene: string) => void;
  onScoreGeneSet: (scores: Float32Array, name: string) => void;
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
}

export function UnifiedEnrichmentSubtab({
  onTermClick,
  onScoreGeneSet,
  groupByColumn,
  setGroupByColumn,
}: UnifiedEnrichmentSubtabProps) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const askCopilot = useViewStore((s) => s.askCopilot);

  const [selectedGroup, setSelectedGroup] = useState("");
  const [groups, setGroups] = useState<string[]>([]);
  const [isLoadingGroups, setIsLoadingGroups] = useState(false);
  const [topN, setTopN] = useState(100);
  const [selectedCollections, setSelectedCollections] = useState<Set<string>>(
    new Set(DEFAULT_MSIGDB_COLLECTIONS),
  );

  const [results, setResults] = useState<EnrichmentResult[]>([]);
  const [isComputing, setIsComputing] = useState(false);
  const [computeError, setComputeError] = useState<string | null>(null);
  const [scoringTerm, setScoringTerm] = useState<string | null>(null);

  const [sortField, setSortField] = useState<SortField>("adj_pval");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  // Programs: the same enrichment, collapsed by gene overlap into clusters.
  const [network, setNetwork] = useState<EnrichmentNetworkResponse | null>(null);
  const [isBuildingNetwork, setIsBuildingNetwork] = useState(false);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [resolution, setResolution] = useState(0.4);
  const [showGraph, setShowGraph] = useState(true);

  const categoricalColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter((c) => {
      if (c.dtype !== "category" && c.dtype !== "object" && c.dtype !== "bool") return false;
      if (c.n_unique > 100) return false;
      if (nCells > 0 && c.n_unique / nCells >= 0.9) return false;
      return true;
    });
  }, [dataset]);

  // Fetch groups when column changes
  useEffect(() => {
    if (!datasetId || !groupByColumn) return;
    let cancelled = false;
    setIsLoadingGroups(true);
    setSelectedGroup("");
    setGroups([]);
    setResults([]);
    setNetwork(null);
    setNetworkError(null);

    type GroupsResp = { groups: string[] };
    const setGroupsFromData = (data: GroupsResp) => {
      if (cancelled) return;
      const grps = data.groups ?? [];
      setGroups(grps);
      if (grps.length > 0) setSelectedGroup(grps[0]!);
    };

    // Try with column first, fall back to default markers, then to obs categories
    (async () => {
      try {
        const data = await apiFetch<GroupsResp>(
          `/datasets/${datasetId}/enrichment/groups?column=${encodeURIComponent(groupByColumn)}`,
        );
        setGroupsFromData(data);
      } catch {
        try {
          // Fallback: try without column (use default markers)
          const data = await apiFetch<GroupsResp>(
            `/datasets/${datasetId}/enrichment/groups`,
          );
          setGroupsFromData(data);
        } catch {
          if (cancelled) return;
          // Final fallback: use obs column categories from dataset metadata
          const col = dataset?.obs_columns.find((c) => c.name === groupByColumn);
          if (col?.values && col.values.length > 0) {
            setGroups(col.values);
            setSelectedGroup(col.values[0]!);
          } else {
            setGroups([]);
          }
        }
      } finally {
        if (!cancelled) setIsLoadingGroups(false);
      }
    })();

    return () => { cancelled = true; };
  }, [datasetId, groupByColumn, dataset]);

  // Build the collapsed program view. Separate from the term table so changing
  // granularity does not re-run the enrichment itself (the backend serves it from
  // cache when the parameters match).
  const buildNetwork = useCallback(
    async (res: number) => {
      if (!datasetId || !groupByColumn || !selectedGroup) return;
      setIsBuildingNetwork(true);
      setNetworkError(null);
      try {
        setNetwork(
          await fetchEnrichmentNetwork(datasetId, {
            column: groupByColumn,
            group: selectedGroup,
            n_genes: topN,
            collections: Array.from(selectedCollections),
            resolution: res,
          }),
        );
      } catch (err) {
        setNetworkError(err instanceof Error ? err.message : String(err));
        setNetwork(null);
      } finally {
        setIsBuildingNetwork(false);
      }
    },
    [datasetId, groupByColumn, selectedGroup, topN, selectedCollections],
  );

  // Compute enrichment
  const handleCompute = useCallback(async () => {
    if (!datasetId || !groupByColumn || !selectedGroup) return;
    setIsComputing(true);
    setComputeError(null);
    try {
      const data = await apiFetch<EnrichmentResponse>(
        `/datasets/${datasetId}/enrichment/compute-local`,
        {
          method: "POST",
          body: JSON.stringify({
            column: groupByColumn,
            group: selectedGroup,
            n_genes: topN,
            collections: Array.from(selectedCollections),
          }),
        },
      );
      setResults(data.results ?? []);
      void buildNetwork(resolution);
    } catch (err) {
      setComputeError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsComputing(false);
    }
  }, [datasetId, groupByColumn, selectedGroup, topN, selectedCollections, buildNetwork, resolution]);

  const handleResolutionChange = useCallback(
    (r: number) => {
      setResolution(r);
      void buildNetwork(r);
    },
    [buildNetwork],
  );

  // Score an arbitrary gene set on the scatter. Shared by the term table, the
  // program rows, and the graph nodes so all three behave identically.
  const scoreGenes = useCallback(
    async (genes: string[], name: string) => {
      if (!datasetId || scoringTerm || genes.length === 0) return;
      setScoringTerm(name);
      try {
        const scoreResponse = await apiFetch<{
          score_name: string;
          scores: number[];
          genes_found: string[];
          genes_missing: string[];
        }>(`/datasets/${datasetId}/genesets/score`, {
          method: "POST",
          body: JSON.stringify({ gene_set: genes, score_name: name }),
        });
        onScoreGeneSet(new Float32Array(scoreResponse.scores), name);
      } catch {
        // Fallback: just signal the gene for expression view
        onTermClick(genes[0]!);
      } finally {
        setScoringTerm(null);
      }
    },
    [datasetId, scoringTerm, onScoreGeneSet, onTermClick],
  );

  const handleTermClick = useCallback(
    (term: EnrichmentResult) => scoreGenes(term.genes, term.term),
    [scoreGenes],
  );

  const askAboutProgram = useCallback(
    (label: string, terms: string[], genes: string[]) => {
      const where = selectedGroup ? ` in the ${selectedGroup} group` : "";
      askCopilot(
        `An enrichment network grouped ${terms.length} redundant gene sets${where} into one program, ` +
          `named after its most significant member "${label}". The member terms are: ${terms.slice(0, 12).join(", ")}` +
          `${terms.length > 12 ? ", ..." : ""}. Genes driving it: ${genes.slice(0, 15).join(", ")}. ` +
          `What single biological programme does this describe, and what does it suggest about these cells?`,
      );
    },
    [askCopilot, selectedGroup],
  );

  const askAboutTerm = useCallback(
    (r: EnrichmentResult) => {
      const where = selectedGroup ? ` enriched in the ${selectedGroup} group` : "";
      askCopilot(
        `What does the pathway "${r.term}" (${r.collection})${where} tell me biologically? Top genes: ${r.genes.slice(0, 8).join(", ")}.`,
      );
    },
    [askCopilot, selectedGroup],
  );

  // Sort results
  const sortedResults = useMemo(() => {
    const sorted = [...results];
    sorted.sort((a, b) => {
      if (sortField === "term")
        return sortDir === "asc" ? a.term.localeCompare(b.term) : b.term.localeCompare(a.term);
      if (sortField === "adj_pval")
        return sortDir === "asc" ? a.adjusted_pvalue - b.adjusted_pvalue : b.adjusted_pvalue - a.adjusted_pvalue;
      return sortDir === "asc" ? a.overlap_count - b.overlap_count : b.overlap_count - a.overlap_count;
    });
    return sorted;
  }, [results, sortField, sortDir]);

  const handleSort = useCallback(
    (field: SortField) => {
      if (sortField === field) setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
      else { setSortField(field); setSortDir(field === "adj_pval" ? "asc" : "desc"); }
    },
    [sortField],
  );

  const sortIcon = useCallback(
    (field: SortField) => {
      if (sortField !== field) return <ArrowUpDown className="h-2.5 w-2.5 text-slate-400" />;
      return sortDir === "asc"
        ? <ArrowUp className="h-2.5 w-2.5 text-blue-500" />
        : <ArrowDown className="h-2.5 w-2.5 text-blue-500" />;
    },
    [sortField, sortDir],
  );

  // Export CSV
  const handleExportCSV = useCallback(() => {
    if (sortedResults.length === 0) return;
    downloadCsv(
      `enrichment_${groupByColumn}_${selectedGroup}_${datasetId}.csv`,
      ["term", "collection", "adj_pvalue", "overlap", "genes"],
      sortedResults.map((r) => [
        r.term,
        r.collection,
        r.adjusted_pvalue.toExponential(4),
        `${r.overlap_count}/${r.gene_count}`,
        r.genes.join(";"),
      ]),
    );
  }, [sortedResults, groupByColumn, selectedGroup, datasetId]);

  if (!dataset || !datasetId) {
    return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  }

  return (
    <div className="space-y-3">
      {/* Column + Group selectors */}
      <div className="flex gap-2">
        <div className="flex-1">
          <label className="mb-1 block text-[11px] font-medium text-slate-500">Column</label>
          <select
            value={groupByColumn}
            onChange={(e) => setGroupByColumn(e.target.value)}
            className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
          >
            {categoricalColumns.map((col) => (
              <option key={col.name} value={col.name}>
                {col.name} ({col.n_unique})
              </option>
            ))}
          </select>
        </div>
        <div className="flex-1">
          <label className="mb-1 block text-[11px] font-medium text-slate-500">Group</label>
          <select
            value={selectedGroup}
            onChange={(e) => setSelectedGroup(e.target.value)}
            disabled={isLoadingGroups || groups.length === 0}
            className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700 disabled:opacity-50"
          >
            {groups.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
            {groups.length === 0 && !isLoadingGroups && (
              <option value="">No groups available</option>
            )}
          </select>
          {isLoadingGroups && (
            <div className="mt-0.5 flex items-center gap-1 text-[10px] text-slate-400">
              <Loader2 className="h-2.5 w-2.5 animate-spin" /> Loading groups...
            </div>
          )}
        </div>
      </div>

      {/* Top N + compute */}
      <div className="flex items-end gap-2">
        <div className="w-20">
          <label className="mb-1 block text-[11px] font-medium text-slate-500">Top N</label>
          <input
            type="number"
            value={topN}
            onChange={(e) => setTopN(parseInt(e.target.value) || 100)}
            min={10}
            max={500}
            className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
          />
        </div>
        <button
          onClick={handleCompute}
          disabled={isComputing || !selectedGroup}
          className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
        >
          {isComputing && <Loader2 className="h-3 w-3 animate-spin" />}
          Compute
        </button>
        {sortedResults.length > 0 && (
          <button
            onClick={handleExportCSV}
            className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-600 hover:bg-slate-50"
          >
            <Download className="h-3 w-3" />
            CSV
          </button>
        )}
      </div>

      {/* MSigDB collection tree */}
      <MSigDBCollectionTree
        selected={selectedCollections}
        onChange={setSelectedCollections}
        compact
      />

      {/* Error */}
      {computeError && (
        <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
          {computeError}
        </div>
      )}

      {/* Terms and programs, side by side: the same result read two ways.
          Container query, not a viewport breakpoint: this subtab lives in a ~650px
          side panel regardless of how wide the window is, so `2xl:` would split
          that panel into two unreadable ~300px columns. Side by side only once the
          panel itself is wide enough, e.g. when expanded to full page. */}
      {sortedResults.length > 0 && (
        <div className="@container">
        <div className="grid grid-cols-1 gap-3 @4xl:grid-cols-2">
        <div className="min-w-0">
        <div className="mb-1 flex items-center gap-1 px-0.5">
          <h4 className="text-[11px] font-semibold text-slate-700">Terms</h4>
          <InfoTip width={300}>
            Every enriched gene set, ranked by adjusted p-value. Gene-set collections are highly
            redundant, so this list repeats itself: the biggest, vaguest parent terms have the most
            statistical power and crowd the top with rephrasings of one signal. The programs view
            groups those rephrasings.
          </InfoTip>
          <span className="ml-auto text-[10px] tabular-nums text-slate-400">
            {sortedResults.length}
          </span>
        </div>
        <div className="max-h-[calc(100vh-500px)] overflow-auto rounded-lg border border-slate-200">
          <table className="w-full text-xs">
            <thead className="sticky top-0 z-10">
              <tr className="border-b border-slate-200 bg-slate-50">
                <th
                  onClick={() => handleSort("term")}
                  className="cursor-pointer px-2 py-1.5 text-left font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center gap-1">Term {sortIcon("term")}</span>
                </th>
                <th
                  onClick={() => handleSort("adj_pval")}
                  className="cursor-pointer px-2 py-1.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">Adj.P {sortIcon("adj_pval")}</span>
                </th>
                <th
                  onClick={() => handleSort("overlap")}
                  className="cursor-pointer px-2 py-1.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">Overlap {sortIcon("overlap")}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedResults.map((r) => {
                const isScoring = scoringTerm === r.term;
                return (
                <tr
                  key={`${r.collection}-${r.term}`}
                  onClick={() => handleTermClick(r)}
                  className={`group cursor-pointer border-b border-slate-50 transition-colors hover:bg-blue-50 ${isScoring ? "bg-blue-50/70" : ""} ${scoringTerm && !isScoring ? "opacity-50" : ""}`}
                  title={`Click to score on scatter\nCollection: ${r.collection}\nGenes: ${r.genes.slice(0, 10).join(", ")}${r.genes.length > 10 ? "..." : ""}`}
                >
                  <td className="max-w-[200px] px-2 py-1.5">
                    <div className="flex items-center gap-1.5">
                      {isScoring && <Loader2 className="h-3 w-3 flex-shrink-0 animate-spin text-blue-500" />}
                      <div className="min-w-0 flex-1 truncate font-medium text-slate-800">{r.term}</div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          askAboutTerm(r);
                        }}
                        title="Ask the co-pilot about this pathway"
                        className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary group-hover:opacity-100"
                      >
                        <Sparkles className="h-3 w-3" />
                      </button>
                    </div>
                    <div className="text-[10px] text-slate-400">{isScoring ? "Scoring..." : r.collection}</div>
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums">
                    <span className={r.adjusted_pvalue < 0.05 ? "font-medium text-blue-600" : "text-slate-600"}>
                      {formatPValue(r.adjusted_pvalue)}
                    </span>
                  </td>
                  <td className="whitespace-nowrap px-2 py-1.5 text-right tabular-nums text-slate-600">
                    {r.overlap_count}/{r.gene_count}
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        </div>

        <div className="min-w-0">
          <div className="mb-1 flex items-center gap-1 px-0.5">
            <h4 className="text-[11px] font-semibold text-slate-700">Programs</h4>
            <InfoTip width={300}>
              The same terms, grouped by how much their genes overlap. Terms that describe one
              signal collapse into a single row named after the strongest of them. This is the
              readable version of the list on the left.
            </InfoTip>
          </div>
          <EnrichmentPrograms
            data={network}
            isLoading={isBuildingNetwork}
            error={networkError}
            resolution={resolution}
            onResolutionChange={handleResolutionChange}
            onScoreGenes={scoreGenes}
            onAsk={askAboutProgram}
            scoringName={scoringTerm}
            showGraph={showGraph}
            onToggleGraph={setShowGraph}
          />
        </div>
        </div>
        </div>
      )}

      {results.length === 0 && !isComputing && !computeError && (
        <p className="text-[10px] text-slate-400">
          Select a group and compute enrichment. Click a result term to score its genes on the scatter plot.
        </p>
      )}
    </div>
  );
}
