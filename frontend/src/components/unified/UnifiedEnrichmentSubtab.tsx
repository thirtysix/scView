import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { Loader2, Download } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { useOnScreenStore } from "@/stores/onScreenStore";
import { apiFetch } from "@/api/client";
import { runPipelineSteps } from "@/api/assistant";
import { getDataset } from "@/api/datasets";
import { collectionLabel } from "@/api/genesets";
import { runGsea, type EnrichmentResult, type GseaResponse, type GseaTerm } from "@/api/enrichment";
import { MSigDBCollectionTree, DEFAULT_MSIGDB_COLLECTIONS } from "@/components/panels/MSigDBCollectionTree";
import { downloadCsv } from "@/lib/csv";
import { InfoTip } from "@/components/common/InfoTip";
import { EnrichmentPrograms } from "@/components/unified/EnrichmentPrograms";
import { EnrichmentTermsTable } from "@/components/unified/EnrichmentTermsTable";
import { GseaResults } from "@/components/unified/GseaResults";
import { PathwayDiagramPanel, type OpenPathway } from "@/components/unified/PathwayDiagramPanel";
import { useEnrichmentSort } from "@/hooks/useEnrichmentSort";
import { useGeneSetScoring, type OnScoreGeneSet } from "@/hooks/useGeneSetScoring";
import { ScoringMethodChoice } from "@/components/common/ScoringMethodChoice";
import {
  fetchEnrichmentNetwork,
  type EnrichmentNetworkResponse,
} from "@/api/enrichmentNetwork";

interface EnrichmentResponse {
  group: string;
  groupby: string;
  n_genes_used: number;
  results: EnrichmentResult[];
  source: string;
  /** Set when the (human) gene sets may not fit the dataset, e.g. a mouse dataset. */
  warning?: string;
}

interface UnifiedEnrichmentSubtabProps {
  onTermClick: (gene: string) => void;
  onScoreGeneSet: OnScoreGeneSet;
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
  const setCurrentDataset = useDatasetStore((s) => s.setCurrentDataset);
  const askCopilot = useViewStore((s) => s.askCopilot);

  const [selectedGroup, setSelectedGroup] = useState("");
  const [groups, setGroups] = useState<string[]>([]);
  const [isLoadingGroups, setIsLoadingGroups] = useState(false);
  // The column has no marker genes yet, so there is nothing to enrich.
  const [noMarkers, setNoMarkers] = useState(false);
  const [isComputingMarkers, setIsComputingMarkers] = useState(false);
  const [markersError, setMarkersError] = useState<string | null>(null);
  const [topN, setTopN] = useState(100);
  const [selectedCollections, setSelectedCollections] = useState<Set<string>>(
    new Set(DEFAULT_MSIGDB_COLLECTIONS),
  );

  const [results, setResults] = useState<EnrichmentResult[]>([]);
  const [isComputing, setIsComputing] = useState(false);
  const [computeError, setComputeError] = useState<string | null>(null);
  const [warning, setWarning] = useState("");

  // Programs: the same enrichment, collapsed by gene overlap into clusters.
  const [network, setNetwork] = useState<EnrichmentNetworkResponse | null>(null);
  const [isBuildingNetwork, setIsBuildingNetwork] = useState(false);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [resolution, setResolution] = useState(0.4);
  const [showGraph, setShowGraph] = useState(true);
  const [excludeRiboMito, setExcludeRiboMito] = useState(true);

  // Preranked GSEA of the group's whole marker ranking (scanpy's score vs the rest).
  const [gsea, setGsea] = useState<GseaResponse | null>(null);
  const [isRunningGsea, setIsRunningGsea] = useState(false);
  const [gseaError, setGseaError] = useState<string | null>(null);

  // A Reactome row's diagram, coloured by the group's markers (what both tables came from).
  const [openPathway, setOpenPathway] = useState<OpenPathway | null>(null);
  const openDiagram = useCallback(
    (r: { term: string; source_id?: string }) => {
      if (!r.source_id) return;
      setOpenPathway({
        stId: r.source_id,
        term: r.term,
        source: groupByColumn && selectedGroup ? { kind: "markers", column: groupByColumn, group: selectedGroup } : null,
      });
    },
    [groupByColumn, selectedGroup],
  );

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

  // The co-pilot's "this result" (stores/onScreenStore.ts)
  const setOnScreen = useOnScreenStore((s) => s.setEnrichment);
  useEffect(() => {
    setOnScreen(results.length && selectedGroup ? { column: groupByColumn, group: selectedGroup } : null);
  }, [results, groupByColumn, selectedGroup, setOnScreen]);
  useEffect(() => () => setOnScreen(null), [setOnScreen]);

  // Fetch groups when column changes
  useEffect(() => {
    if (!datasetId || !groupByColumn) return;
    let cancelled = false;
    setIsLoadingGroups(true);
    setSelectedGroup("");
    setGroups([]);
    setNoMarkers(false);
    setMarkersError(null);
    setResults([]);
    setWarning("");
    setNetwork(null);
    setNetworkError(null);
    setGsea(null);
    setGseaError(null);

    type GroupsResp = { groups: string[] };
    const setGroupsFromData = (data: GroupsResp) => {
      if (cancelled) return;
      const grps = data.groups ?? [];
      setGroups(grps);
      if (grps.length > 0) setSelectedGroup(grps[0]!);
    };

    // Only the column's own marker genes will do: another column's groups (the
    // clusters, say) can't be enriched under this column's name.
    (async () => {
      try {
        const data = await apiFetch<GroupsResp>(
          `/datasets/${datasetId}/enrichment/groups?column=${encodeURIComponent(groupByColumn)}`,
        );
        setGroupsFromData(data);
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("404")) setNoMarkers(true);
        else setMarkersError(msg);
      } finally {
        if (!cancelled) setIsLoadingGroups(false);
      }
    })();

    return () => { cancelled = true; };
    // `dataset` is a dependency so groups are fetched again after markers are computed.
  }, [datasetId, groupByColumn, dataset]);

  // Build the collapsed program view. Separate from the term table so changing
  // granularity does not re-run the enrichment itself (the backend serves it from
  // cache when the parameters match).
  // Results belong to one group: clear them when the group changes, and drop a
  // response that arrives after the user has moved on to another group.
  const groupRef = useRef(selectedGroup);
  groupRef.current = selectedGroup;
  useEffect(() => {
    setResults([]);
    setWarning("");
    setNetwork(null);
    setNetworkError(null);
    setComputeError(null);
    setGsea(null);
    setGseaError(null);
    setOpenPathway(null);
  }, [selectedGroup]);

  const buildNetwork = useCallback(
    async (res: number) => {
      if (!datasetId || !groupByColumn || !selectedGroup) return;
      setIsBuildingNetwork(true);
      setNetworkError(null);
      try {
        const net = await fetchEnrichmentNetwork(datasetId, {
          column: groupByColumn,
          group: selectedGroup,
          n_genes: topN,
          collections: Array.from(selectedCollections),
          resolution: res,
          excludeRiboMito,
        });
        if (groupRef.current === selectedGroup) setNetwork(net);
      } catch (err) {
        setNetworkError(err instanceof Error ? err.message : String(err));
        setNetwork(null);
      } finally {
        setIsBuildingNetwork(false);
      }
    },
    [datasetId, groupByColumn, selectedGroup, topN, selectedCollections, excludeRiboMito],
  );

  // Compute marker genes for a column that has none, then reload the dataset so
  // its groups are fetched again.
  const handleComputeMarkers = useCallback(async () => {
    if (!datasetId || !groupByColumn) return;
    setIsComputingMarkers(true);
    setMarkersError(null);
    try {
      const res = await runPipelineSteps(datasetId, ["marker_genes"], {
        marker_columns: [groupByColumn],
      });
      const err = res.errors?.marker_genes;
      if (err) throw new Error(err);
      setCurrentDataset(await getDataset(datasetId));
    } catch (e) {
      setMarkersError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsComputingMarkers(false);
    }
  }, [datasetId, groupByColumn, setCurrentDataset]);

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
            exclude_ribo_mito: excludeRiboMito,
          }),
        },
      );
      if (groupRef.current !== selectedGroup) return;
      setResults(data.results ?? []);
      setWarning(data.warning ?? "");
      void buildNetwork(resolution);
    } catch (err) {
      setComputeError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsComputing(false);
    }
  }, [datasetId, groupByColumn, selectedGroup, topN, selectedCollections, excludeRiboMito, buildNetwork, resolution]);

  const handleGsea = useCallback(async () => {
    if (!datasetId || !groupByColumn || !selectedGroup) return;
    setIsRunningGsea(true);
    setGseaError(null);
    try {
      const data = await runGsea(datasetId, {
        column: groupByColumn,
        group: selectedGroup,
        collections: Array.from(selectedCollections),
        exclude_ribo_mito: excludeRiboMito,
      });
      if (groupRef.current === selectedGroup) setGsea(data);
    } catch (err) {
      setGseaError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsRunningGsea(false);
    }
  }, [datasetId, groupByColumn, selectedGroup, selectedCollections, excludeRiboMito]);

  const handleResolutionChange = useCallback(
    (r: number) => {
      setResolution(r);
      void buildNetwork(r);
    },
    [buildNetwork],
  );

  // Score an arbitrary gene set on the scatter. Shared by the term table, the
  // program rows, and the graph nodes so all three behave identically.
  const { scoreGenes, scoringName: scoringTerm } = useGeneSetScoring(
    datasetId,
    onScoreGeneSet,
    onTermClick,
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

  const gseaLabels = useMemo(
    () => ({ up: `Up in ${selectedGroup}`, down: `Down in ${selectedGroup}` }),
    [selectedGroup],
  );

  const askAboutGseaTerm = useCallback(
    (r: GseaTerm) => {
      askCopilot(
        `In a GSEA of the ${selectedGroup} group's marker ranking (every gene ranked by its Wilcoxon ` +
          `score against the other cells), "${r.term}" (${collectionLabel(r)}) has NES ${r.nes.toFixed(2)} ` +
          `(${r.nes > 0 ? "higher" : "lower"} in ${selectedGroup}), FDR ${r.fdr.toExponential(2)}. ` +
          `Leading-edge genes: ${r.leading_edge.slice(0, 15).join(", ")}. What does this tell me about these cells?`,
      );
    },
    [askCopilot, selectedGroup],
  );

  const askAboutTerm = useCallback(
    (r: EnrichmentResult) => {
      const where = selectedGroup ? ` enriched in the ${selectedGroup} group` : "";
      askCopilot(
        `What does the pathway "${r.term}" (${collectionLabel(r)})${where} tell me biologically? Top genes: ${r.genes.slice(0, 8).join(", ")}.`,
      );
    },
    [askCopilot, selectedGroup],
  );

  // Sort results
  const { sorted: sortedResults, sortField, sortDir, onSort } = useEnrichmentSort(results);

  // Export CSV
  const handleExportCSV = useCallback(() => {
    if (sortedResults.length === 0) return;
    downloadCsv(
      `enrichment_${groupByColumn}_${selectedGroup}_${datasetId}.csv`,
      ["term", "collection", "adj_pvalue", "overlap", "genes"],
      sortedResults.map((r) => [
        r.term,
        collectionLabel(r),
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
            aria-label="Column"
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
            aria-label="Group"
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

      {noMarkers && (
        <div className="rounded border border-amber-200 bg-amber-50 px-2 py-1.5 text-[11px] text-amber-800">
          <p>
            <span className="font-semibold">{groupByColumn}</span> has no marker genes yet, and
            enrichment starts from them.
          </p>
          <button
            onClick={handleComputeMarkers}
            disabled={isComputingMarkers}
            className="mt-1 flex items-center gap-1.5 rounded-md bg-primary px-2.5 py-1 text-[11px] font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
          >
            {isComputingMarkers && <Loader2 className="h-3 w-3 animate-spin" />}
            {isComputingMarkers ? "Computing markers..." : "Compute markers for this column"}
          </button>
        </div>
      )}
      {markersError && (
        <div className="rounded border border-red-200 bg-red-50 px-2 py-1.5 text-[11px] text-red-700">
          {markersError}
        </div>
      )}

      {/* Top N + compute */}
      <div className="flex items-end gap-2">
        <div className="w-20">
          <label className="mb-1 block text-[11px] font-medium text-slate-500">Top N</label>
          <input
            aria-label="Top N"
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
        <button
          onClick={handleGsea}
          disabled={isRunningGsea || !selectedGroup || selectedCollections.size === 0}
          title="GSEA on every gene, ranked by its marker score: no top-N cut-off"
          className="flex items-center gap-1.5 rounded-md border border-primary px-3 py-1.5 text-xs font-medium text-primary transition hover:bg-primary/5 disabled:opacity-50"
        >
          {isRunningGsea && <Loader2 className="h-3 w-3 animate-spin" />}
          GSEA
        </button>
        <InfoTip width={300}>
          Compute tests up to N markers as a set (over-representation): genes in at least 20% of
          the group&apos;s cells with adjusted p &lt; 0.05, highest fold change first, ribosomal and mitochondrial genes left out,
          against every gene the marker test saw. GSEA ranks every gene by
          its marker score (Wilcoxon, this group against the other cells) and asks whether a gene
          set bunches at the top or bottom of that ranking, so it needs no top-N cut-off. The
          leading edge is the genes that drive the score; click a row to score them on the scatter.
        </InfoTip>
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

      {/* Ribo/mito filter */}
      <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
        <input
          type="checkbox"
          checked={excludeRiboMito}
          onChange={(e) => setExcludeRiboMito(e.target.checked)}
          className="h-3 w-3 rounded border-slate-300"
        />
        Exclude ribosomal / mitochondrial genes
        <InfoTip width={300}>
          Ribosomal and mitochondrial genes dominate single-cell markers as an artifact and are
          shared across nearly every pathway and cell-type signature, so they inflate a generic
          "translation" program and mislabel cell types (an ovary cluster tagged as pancreatic).
          Dropping them from the query — not the background — fixes both. Turn off only to study
          ribosome biology directly.
        </InfoTip>
      </label>

      {/* MSigDB collection tree */}
      <MSigDBCollectionTree
        selected={selectedCollections}
        onChange={setSelectedCollections}
        datasetId={datasetId}
        compact
      />

      {/* Error */}
      {computeError && (
        <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
          {computeError}
        </div>
      )}
      {warning && (
        <div role="note" className="rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
          {warning}
        </div>
      )}

      {/* Terms and programs, side by side: the same result read two ways.
          Container query, not a viewport breakpoint: this subtab lives in a ~650px
          side panel regardless of how wide the window is, so `2xl:` would split
          that panel into two unreadable ~300px columns. Side by side only once the
          panel itself is wide enough, e.g. when expanded to full page. */}
      {(sortedResults.length > 0 || gsea) && <ScoringMethodChoice label="Click a term to score its genes with" />}

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
        <EnrichmentTermsTable
          results={sortedResults}
          sortField={sortField}
          sortDir={sortDir}
          onSort={onSort}
          onTermClick={handleTermClick}
          onAsk={askAboutTerm}
          scoringTerm={scoringTerm}
          onDiagram={openDiagram}
        />
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

      {gseaError && (
        <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">GSEA: {gseaError}</div>
      )}
      {isRunningGsea && !gsea && (
        <p className="text-[10px] text-slate-400">
          Running GSEA: a few seconds for Hallmark, up to ~30 s for many collections.
        </p>
      )}
      {gsea && (
        <div className="text-xs">
          <div className="mb-1 flex items-center gap-1 px-0.5">
            <h4 className="text-[11px] font-semibold text-slate-700">GSEA: {selectedGroup}, every gene ranked</h4>
          </div>
          <GseaResults
            data={gsea}
            labels={gseaLabels}
            csvName={`gsea_${groupByColumn}_${selectedGroup}_${datasetId}`}
            onTermClick={(r) => scoreGenes(r.leading_edge, r.term)}
            onAsk={askAboutGseaTerm}
            scoringTerm={scoringTerm}
            onDiagram={openDiagram}
          />
        </div>
      )}

      {openPathway && datasetId && (
        <PathwayDiagramPanel
          datasetId={datasetId}
          datasetName={dataset?.name}
          open={openPathway}
          onClose={() => setOpenPathway(null)}
        />
      )}

      {results.length === 0 && !isComputing && !computeError && (
        <p className="text-[10px] text-slate-400">
          Select a group and compute enrichment. Click a result term to score its genes on the scatter plot.
        </p>
      )}
    </div>
  );
}
