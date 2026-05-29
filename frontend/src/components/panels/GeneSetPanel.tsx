import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import {
  Library,
  Loader2,
  AlertCircle,
  Check,
  X,
  Search,
  ArrowRight,
  PlayCircle,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Download,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import Plot from "react-plotly.js";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useEmbedding } from "@/hooks/useEmbedding";
import { apiFetch } from "@/api/client";
import { EmbeddingScatter } from "@/components/plots/EmbeddingScatter";
import { ViolinPlot } from "@/components/plots/ViolinPlot";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { formatNumber, formatPValue } from "@/lib/formatting";
import {
  MSigDBCollectionTree,
  DEFAULT_MSIGDB_COLLECTIONS,
} from "./MSigDBCollectionTree";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface GeneSetCollection {
  id: string;
  name: string;
  description: string;
  n_sets?: number;
}

interface GeneSetSearchResult {
  name: string;
  collection: string;
  n_genes: number;
  genes: string[];
}

interface GeneSetScoreResponse {
  score_name: string;
  n_cells: number;
  scores: number[];
  min_score: number;
  max_score: number;
  genes_found: string[];
  genes_missing: string[];
}

interface EnrichmentResult {
  term: string;
  pvalue: number;
  adjusted_pvalue: number;
  overlap_count: number;
  gene_count: number;
  genes: string[];
  collection?: string;
}

interface EnrichmentResponse {
  group: string;
  groupby: string;
  n_genes_used: number;
  results: EnrichmentResult[];
  source: string;
}

type SortField = "term" | "pvalue" | "adjusted_pvalue" | "overlap_count";
type SortDir = "asc" | "desc";

/* ------------------------------------------------------------------ */
/* Main component                                                      */
/* ------------------------------------------------------------------ */

export function GeneSetPanel() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const plotBackground = useSettingsStore((s) => s.plotBackground);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const selectedIndices = useSelectionStore((s) => s.selectedCellIndices);

  const { positions, numCells, dimensions, isLoading: embeddingLoading } =
    useEmbedding();

  // ---- Section collapse ----
  const [scoringSectionOpen, setScoringSectionOpen] = useState(true);
  const [enrichmentSectionOpen, setEnrichmentSectionOpen] = useState(true);

  // ---- Gene Set Scoring state ----
  const [collections, setCollections] = useState<GeneSetCollection[]>([]);
  const [selectedCollection, setSelectedCollection] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<GeneSetSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const [geneInput, setGeneInput] = useState("");
  const [scoreName, setScoreName] = useState("gene_set_score");
  const [isScoring, setIsScoring] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [scoreResult, setScoreResult] = useState<GeneSetScoreResponse | null>(
    null,
  );
  const [scoreValues, setScoreValues] = useState<Float32Array | null>(null);
  const [violinData, setViolinData] = useState<Record<string, number[]>>({});
  const [currentGroupBy, setCurrentGroupBy] = useState("");

  // ---- Enrichment state ----
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [selectedColumn, setSelectedColumn] = useState("");
  const [groups, setGroups] = useState<string[]>([]);
  const [enrichmentStatus, setEnrichmentStatus] = useState<
    Record<string, boolean>
  >({});
  const [selectedGroup, setSelectedGroup] = useState("");
  const [nGenes, setNGenes] = useState(100);
  const [selectedCollections, setSelectedCollections] = useState<Set<string>>(
    new Set(DEFAULT_MSIGDB_COLLECTIONS),
  );

  const [enrichment, setEnrichment] = useState<EnrichmentResponse | null>(
    null,
  );
  const [isEnrichLoading, setIsEnrichLoading] = useState(false);
  const [isLoadingColumns, setIsLoadingColumns] = useState(false);
  const [isLoadingGroups, setIsLoadingGroups] = useState(false);
  const [enrichError, setEnrichError] = useState<string | null>(null);

  // Batch enrichment state
  const [isBatchComputing, setIsBatchComputing] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });

  const [sortField, setSortField] = useState<SortField>("pvalue");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [topN, setTopN] = useState(20);

  // ---- Derived ----
  const categoricalColumns = useMemo(() => {
    if (!dataset) return [];
    return dataset.obs_columns.filter(
      (c) =>
        c.dtype === "category" || c.dtype === "object" || c.dtype === "bool",
    );
  }, [dataset]);

  // ---- Initialize group-by ----
  useEffect(() => {
    if (!currentGroupBy) {
      if (dataset?.active_clustering) {
        const found = categoricalColumns.find(
          (c) => c.name === dataset.active_clustering,
        );
        if (found) {
          setCurrentGroupBy(found.name);
          return;
        }
      }
      if (categoricalColumns.length > 0) {
        setCurrentGroupBy(categoricalColumns[0]!.name);
      }
    }
  }, [categoricalColumns, currentGroupBy, dataset]);

  // ---- Fetch gene set collections on mount ----
  useEffect(() => {
    if (!datasetId) return;
    apiFetch<{ collections: GeneSetCollection[] }>(
      `/datasets/${datasetId}/genesets/collections`,
    )
      .then((data) => {
        setCollections(data.collections);
        if (data.collections.length > 0) {
          setSelectedCollection(data.collections[0]!.id);
        }
      })
      .catch(() => {});
  }, [datasetId]);

  // ---- Fetch enrichment columns on mount ----
  useEffect(() => {
    if (!datasetId) return;
    setIsLoadingColumns(true);
    apiFetch<{ columns: string[]; categorical_columns?: string[] }>(
      `/datasets/${datasetId}/enrichment/columns`,
    )
      .then((data) => {
        setAvailableColumns(data.columns);
        if (data.columns.length > 0 && !selectedColumn) {
          setSelectedColumn(data.columns[0]!);
        }
      })
      .catch(() => setAvailableColumns([]))
      .finally(() => setIsLoadingColumns(false));
  }, [datasetId]);

  // ---- Fetch groups when enrichment column changes ----
  useEffect(() => {
    if (!datasetId || !selectedColumn) return;
    setIsLoadingGroups(true);
    setEnrichment(null);
    setEnrichError(null);
    apiFetch<{
      groups: string[];
      enrichment_computed: Record<string, boolean>;
    }>(
      `/datasets/${datasetId}/enrichment/groups?column=${encodeURIComponent(selectedColumn)}`,
    )
      .then((data) => {
        setGroups(data.groups);
        setEnrichmentStatus(data.enrichment_computed ?? {});
        if (data.groups.length > 0) {
          setSelectedGroup(data.groups[0]!);
        }
      })
      .catch(() => {
        setGroups([]);
        setEnrichmentStatus({});
      })
      .finally(() => setIsLoadingGroups(false));
  }, [datasetId, selectedColumn]);

  // ---- Gene set search (debounced) ----
  const doSearch = useCallback(
    (q: string) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (!datasetId) {
        setSearchResults([]);
        setDropdownOpen(false);
        setIsSearching(false);
        return;
      }
      setIsSearching(true);
      const delay = q.trim().length > 0 ? 300 : 0;
      debounceRef.current = setTimeout(async () => {
        try {
          const data = await apiFetch<{
            results: GeneSetSearchResult[];
            total: number;
          }>(
            `/datasets/${datasetId}/genesets/search?q=${encodeURIComponent(q.trim())}&collection=${encodeURIComponent(selectedCollection)}&limit=200`,
          );
          setSearchResults(data.results);
          setDropdownOpen(true);
          setActiveIndex(-1);
        } catch {
          setSearchResults([]);
          setDropdownOpen(false);
        } finally {
          setIsSearching(false);
        }
      }, delay);
    },
    [datasetId, selectedCollection],
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  useEffect(() => {
    setSearchResults([]);
    if (dropdownOpen) doSearch(searchQuery);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCollection]);

  // Click-outside dropdown
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        searchInputRef.current &&
        !searchInputRef.current.contains(e.target as Node)
      ) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleSearchInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSearchQuery(e.target.value);
      doSearch(e.target.value);
    },
    [doSearch],
  );

  const handleUseGeneSet = useCallback((result: GeneSetSearchResult) => {
    setGeneInput(result.genes.join(", "));
    setScoreName(result.name.toLowerCase().replace(/[^a-z0-9]+/g, "_"));
    setSearchResults([]);
    setDropdownOpen(false);
    setSearchQuery(result.name);
  }, []);

  const handleSearchKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!dropdownOpen || searchResults.length === 0) return;
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setActiveIndex((prev) =>
            prev < searchResults.length - 1 ? prev + 1 : 0,
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setActiveIndex((prev) =>
            prev > 0 ? prev - 1 : searchResults.length - 1,
          );
          break;
        case "Enter":
          e.preventDefault();
          if (activeIndex >= 0 && activeIndex < searchResults.length) {
            handleUseGeneSet(searchResults[activeIndex]!);
          }
          break;
        case "Escape":
          e.preventDefault();
          setDropdownOpen(false);
          setActiveIndex(-1);
          break;
      }
    },
    [dropdownOpen, searchResults, activeIndex, handleUseGeneSet],
  );

  // ---- Gene parsing & scoring ----
  const parsedGenes = useMemo(() => {
    if (!geneInput.trim()) return [];
    return geneInput
      .split(/[,\n\r\t;]+/)
      .map((g) => g.trim())
      .filter((g) => g.length > 0);
  }, [geneInput]);

  const handleScore = useCallback(async () => {
    if (!datasetId || parsedGenes.length === 0) return;
    setIsScoring(true);
    setScoreError(null);
    setScoreResult(null);
    setScoreValues(null);

    try {
      const data = await apiFetch<GeneSetScoreResponse>(
        `/datasets/${datasetId}/genesets/score`,
        {
          method: "POST",
          body: JSON.stringify({
            gene_set: parsedGenes,
            score_name: scoreName,
          }),
        },
      );
      setScoreResult(data);
      const f32 = new Float32Array(data.scores.length);
      for (let i = 0; i < data.scores.length; i++) f32[i] = data.scores[i]!;
      setScoreValues(f32);

      if (currentGroupBy && dataset) {
        buildViolinFromScores(data.scores);
      }
    } catch (err) {
      setScoreError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsScoring(false);
    }
  }, [datasetId, parsedGenes, scoreName, currentGroupBy, dataset]);

  const buildViolinFromScores = useCallback(
    async (scores: number[]) => {
      if (!datasetId || !currentGroupBy) return;
      try {
        const data = await apiFetch<{
          column: string;
          values: (string | number)[];
        }>(
          `/datasets/${datasetId}/metadata/${encodeURIComponent(currentGroupBy)}?format=json`,
        );
        const groups: Record<string, number[]> = {};
        for (
          let i = 0;
          i < Math.min(scores.length, data.values.length);
          i++
        ) {
          const g = String(data.values[i] ?? "unknown");
          if (!groups[g]) groups[g] = [];
          groups[g]!.push(scores[i]!);
        }
        setViolinData(groups);
      } catch {
        setViolinData({});
      }
    },
    [datasetId, currentGroupBy],
  );

  const handleGroupByChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setCurrentGroupBy(e.target.value);
      if (scoreResult) buildViolinFromScores(scoreResult.scores);
    },
    [scoreResult, buildViolinFromScores],
  );

  // ---- Enrichment compute ----
  const handleComputeEnrichment = useCallback(async () => {
    if (!datasetId || !selectedGroup || selectedCollections.size === 0) return;
    setIsEnrichLoading(true);
    setEnrichError(null);

    try {
      const data = await apiFetch<EnrichmentResponse>(
        `/datasets/${datasetId}/enrichment/compute-local`,
        {
          method: "POST",
          body: JSON.stringify({
            column: selectedColumn,
            group: selectedGroup,
            n_genes: nGenes,
            collections: Array.from(selectedCollections),
          }),
        },
      );
      setEnrichment(data);
      setEnrichmentStatus((prev) => ({ ...prev, [selectedGroup]: true }));
    } catch (err) {
      setEnrichError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsEnrichLoading(false);
    }
  }, [datasetId, selectedColumn, selectedGroup, nGenes, selectedCollections]);

  // ---- Batch compute all groups ----
  const handleComputeAllGroups = useCallback(async () => {
    if (!datasetId || groups.length === 0 || selectedCollections.size === 0)
      return;
    setIsBatchComputing(true);
    setBatchProgress({ current: 0, total: groups.length });

    for (let i = 0; i < groups.length; i++) {
      const g = groups[i]!;
      setBatchProgress({ current: i + 1, total: groups.length });
      if (enrichmentStatus[g]) continue; // skip already computed

      try {
        const data = await apiFetch<EnrichmentResponse>(
          `/datasets/${datasetId}/enrichment/compute-local`,
          {
            method: "POST",
            body: JSON.stringify({
              column: selectedColumn,
              group: g,
              n_genes: nGenes,
              collections: Array.from(selectedCollections),
            }),
          },
        );
        setEnrichmentStatus((prev) => ({ ...prev, [g]: true }));
        // Show last result
        if (i === groups.length - 1 || g === selectedGroup) {
          setEnrichment(data);
        }
      } catch {
        // Continue batch even if one group fails
      }
    }
    setIsBatchComputing(false);
  }, [
    datasetId,
    groups,
    selectedColumn,
    nGenes,
    selectedCollections,
    enrichmentStatus,
    selectedGroup,
  ]);

  // ---- Term click: auto-score on UMAP ----
  const handleTermClick = useCallback(
    async (term: string) => {
      if (!datasetId) return;
      // Look up the term in MSigDB to get its genes
      try {
        const data = await apiFetch<{
          results: GeneSetSearchResult[];
        }>(`/datasets/${datasetId}/genesets/search?q=${encodeURIComponent(term)}&limit=1`);
        if (data.results.length > 0) {
          const result = data.results[0]!;
          setGeneInput(result.genes.join(", "));
          setScoreName(result.name.toLowerCase().replace(/[^a-z0-9]+/g, "_"));
          setSearchQuery(result.name);
          // Auto-score
          setScoringSectionOpen(true);
          // Trigger score after state update
          setTimeout(() => {
            const scoreBtn = document.getElementById("score-btn");
            scoreBtn?.click();
          }, 100);
        }
      } catch {
        // Ignore
      }
    },
    [datasetId],
  );

  // ---- Sorting ----
  const sortedResults = useMemo(() => {
    if (!enrichment) return [];
    const sorted = [...enrichment.results];
    sorted.sort((a, b) => {
      switch (sortField) {
        case "term":
          return sortDir === "asc"
            ? a.term.localeCompare(b.term)
            : b.term.localeCompare(a.term);
        case "pvalue":
          return sortDir === "asc"
            ? a.pvalue - b.pvalue
            : b.pvalue - a.pvalue;
        case "adjusted_pvalue":
          return sortDir === "asc"
            ? a.adjusted_pvalue - b.adjusted_pvalue
            : b.adjusted_pvalue - a.adjusted_pvalue;
        case "overlap_count":
          return sortDir === "asc"
            ? a.overlap_count - b.overlap_count
            : b.overlap_count - a.overlap_count;
        default:
          return 0;
      }
    });
    return sorted;
  }, [enrichment, sortField, sortDir]);

  const topResults = useMemo(
    () => sortedResults.slice(0, topN),
    [sortedResults, topN],
  );

  const handleSort = useCallback(
    (field: SortField) => {
      if (sortField === field) {
        setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setSortField(field);
        setSortDir(
          field === "pvalue" || field === "adjusted_pvalue" ? "asc" : "desc",
        );
      }
    },
    [sortField],
  );

  const sortIcon = useCallback(
    (field: SortField) => {
      if (sortField !== field)
        return <ArrowUpDown className="h-3 w-3 text-slate-400" />;
      return sortDir === "asc" ? (
        <ArrowUp className="h-3 w-3 text-blue-500" />
      ) : (
        <ArrowDown className="h-3 w-3 text-blue-500" />
      );
    },
    [sortField, sortDir],
  );

  // ---- Export CSV ----
  const handleExportCSV = useCallback(() => {
    if (sortedResults.length === 0) return;
    const headers = [
      "term",
      "collection",
      "p_value",
      "adjusted_p_value",
      "overlap_count",
      "gene_count",
      "genes",
    ];
    const rows = sortedResults.map((r) =>
      [
        `"${r.term}"`,
        `"${r.collection ?? ""}"`,
        r.pvalue.toExponential(4),
        r.adjusted_pvalue.toExponential(4),
        r.overlap_count,
        r.gene_count,
        `"${r.genes.join(";")}"`,
      ].join(","),
    );
    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `enrichment_${selectedGroup}_${datasetId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [sortedResults, selectedGroup, datasetId]);

  // ---- Score range for legend ----
  const scoreMinMax = useMemo(() => {
    if (!scoreResult) return { min: 0, max: 1 };
    return { min: scoreResult.min_score, max: scoreResult.max_score };
  }, [scoreResult]);

  // ---- No dataset ----
  if (!dataset || !datasetId) {
    return (
      <div className="mx-auto max-w-5xl space-y-6">
        <div className="flex items-center gap-3">
          <Library className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">
            Gene Sets & Enrichment
          </h2>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
          Load a dataset to explore gene sets and pathway enrichment.
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center justify-between">
        <div className="flex items-center gap-3">
          <Library className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">
            Gene Sets & Enrichment
          </h2>
        </div>
        {numCells > 0 && (
          <span className="text-sm text-slate-500">
            {formatNumber(numCells)} cells
          </span>
        )}
      </div>

      {/* ================================================================ */}
      {/*  SECTION 1: Gene Set Scoring                                     */}
      {/* ================================================================ */}
      <div className="flex-shrink-0 rounded-xl border border-slate-200 bg-white shadow-sm">
        <button
          onClick={() => setScoringSectionOpen(!scoringSectionOpen)}
          className="flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold text-slate-700 hover:bg-slate-50"
        >
          {scoringSectionOpen ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          Gene Set Scoring
          {scoreResult && (
            <span className="ml-2 rounded-full bg-purple-100 px-2 py-0.5 text-xs font-semibold text-purple-700">
              {scoreResult.genes_found.length} genes scored
            </span>
          )}
        </button>

        {scoringSectionOpen && (
          <div className="space-y-3 border-t border-slate-100 px-4 pb-4 pt-3">
            {/* Collection selector + search */}
            <div className="flex items-center gap-2">
              <select
                value={selectedCollection}
                onChange={(e) => setSelectedCollection(e.target.value)}
                className="flex-shrink-0 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
              >
                {collections.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                    {c.n_sets ? ` (${c.n_sets})` : ""}
                  </option>
                ))}
              </select>
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
                <input
                  ref={searchInputRef}
                  type="text"
                  value={searchQuery}
                  onChange={handleSearchInput}
                  onKeyDown={handleSearchKeyDown}
                  onFocus={() => {
                    if (searchResults.length > 0) setDropdownOpen(true);
                    else doSearch(searchQuery);
                  }}
                  placeholder="Search gene sets (e.g. HALLMARK, APOPTOSIS)..."
                  className="w-full rounded-lg border border-slate-300 bg-white py-2 pl-8 pr-8 text-sm text-slate-700 placeholder-slate-400 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
                  autoComplete="off"
                  spellCheck={false}
                />
                {searchQuery && (
                  <button
                    onClick={() => {
                      setSearchQuery("");
                      setSearchResults([]);
                      setDropdownOpen(false);
                      searchInputRef.current?.focus();
                    }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-slate-400 hover:text-slate-600"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
                {isSearching && (
                  <div className="absolute right-8 top-1/2 -translate-y-1/2">
                    <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-300 border-t-blue-500" />
                  </div>
                )}
                {dropdownOpen && (
                  <div
                    ref={dropdownRef}
                    className="absolute left-0 z-50 mt-1 max-h-64 w-full overflow-auto rounded-lg border border-slate-200 bg-white shadow-lg"
                  >
                    {searchResults.length > 0
                      ? searchResults.map((result, i) => (
                          <button
                            key={`${result.collection}-${result.name}`}
                            onClick={() => {
                              handleUseGeneSet(result);
                              setDropdownOpen(false);
                            }}
                            onMouseEnter={() => setActiveIndex(i)}
                            className={`flex w-full items-center justify-between px-3 py-2 text-left transition-colors ${
                              i === activeIndex
                                ? "bg-blue-50"
                                : "hover:bg-slate-50"
                            }`}
                          >
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-2">
                                <span className="rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-semibold text-blue-700">
                                  {result.collection}
                                </span>
                                <span className="truncate text-xs font-medium text-slate-800">
                                  {result.name}
                                </span>
                              </div>
                              <span className="text-[10px] text-slate-500">
                                {result.n_genes} genes
                              </span>
                            </div>
                            <span className="ml-2 flex flex-shrink-0 items-center gap-1 text-xs font-medium text-blue-600">
                              Use <ArrowRight className="h-3 w-3" />
                            </span>
                          </button>
                        ))
                      : !isSearching && (
                          <div className="px-3 py-4 text-center text-xs text-slate-400">
                            {searchQuery.trim()
                              ? `No gene sets found for "${searchQuery.trim()}"`
                              : "No gene sets available for this collection"}
                          </div>
                        )}
                  </div>
                )}
              </div>
            </div>

            {/* Gene input + score button + group-by */}
            <div className="flex flex-col gap-2">
              <textarea
                value={geneInput}
                onChange={(e) => setGeneInput(e.target.value)}
                placeholder={
                  "CD3D, CD3E, CD4, CD8A, CD8B\nor paste one gene per line..."
                }
                rows={3}
                className="w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-700 placeholder-slate-400 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
              />
              <div className="flex items-center gap-3">
                <button
                  id="score-btn"
                  onClick={handleScore}
                  disabled={parsedGenes.length === 0 || isScoring}
                  className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isScoring ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Scoring...
                    </>
                  ) : (
                    <>
                      <Check className="h-4 w-4" />
                      Score ({parsedGenes.length} gene
                      {parsedGenes.length !== 1 ? "s" : ""})
                    </>
                  )}
                </button>
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-slate-600">
                    Group by:
                  </label>
                  <select
                    value={currentGroupBy}
                    onChange={handleGroupByChange}
                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
                  >
                    {categoricalColumns.map((col) => (
                      <option key={col.name} value={col.name}>
                        {col.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Score error */}
            {scoreError && (
              <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
                <AlertCircle className="h-4 w-4 flex-shrink-0" />
                {scoreError}
              </div>
            )}

            {/* Gene feedback */}
            {scoreResult && (
              <div className="flex flex-wrap items-center gap-2 text-xs">
                {scoreResult.genes_found.length > 0 && (
                  <span className="rounded bg-green-100 px-2 py-0.5 text-green-700">
                    {scoreResult.genes_found.length} found
                  </span>
                )}
                {scoreResult.genes_missing.length > 0 && (
                  <span className="rounded bg-amber-100 px-2 py-0.5 text-amber-700">
                    {scoreResult.genes_missing.length} missing:{" "}
                    {scoreResult.genes_missing.slice(0, 5).join(", ")}
                    {scoreResult.genes_missing.length > 5 && "..."}
                  </span>
                )}
              </div>
            )}

            {/* Score results: scatter + violin */}
            {scoreResult && (
              <div className="flex gap-4" style={{ height: 400 }}>
                <div className="flex min-w-0 flex-1 flex-col gap-2">
                  <h3 className="text-sm font-semibold text-slate-700">
                    Gene Set Score on Embedding
                  </h3>
                  <div className="relative min-h-0 flex-1 rounded-xl border border-slate-200 bg-white shadow-sm">
                    {embeddingLoading && !positions ? (
                      <div className="flex h-full items-center justify-center">
                        <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
                      </div>
                    ) : (
                      <EmbeddingScatter
                        positions={positions}
                        colorValues={scoreValues}
                        colorType="continuous"
                        pointSize={pointSize}
                        opacity={opacity}
                        selectedIndices={selectedIndices}
                        background={plotBackground}
                        maxRenderedCells={maxRenderedCells}
                        dimensions={dimensions}
                      />
                    )}
                  </div>
                  {scoreValues && (
                    <div className="flex-shrink-0 px-2">
                      <ColorLegend
                        type="continuous"
                        min={scoreMinMax.min}
                        max={scoreMinMax.max}
                        label={scoreName}
                      />
                    </div>
                  )}
                </div>
                <div className="flex w-[380px] flex-shrink-0 flex-col gap-2">
                  <h3 className="text-sm font-semibold text-slate-700">
                    Score by {currentGroupBy || "Group"}
                  </h3>
                  <div className="flex-1 overflow-auto rounded-xl border border-slate-200 bg-white p-2 shadow-sm">
                    {Object.keys(violinData).length === 0 ? (
                      <div className="flex h-full items-center justify-center text-sm text-slate-400">
                        Violin plot will appear after scoring
                      </div>
                    ) : (
                      <ViolinPlot
                        data={violinData}
                        title={scoreName}
                        xLabel={currentGroupBy}
                        yLabel="Score"
                        height={Math.max(
                          300,
                          Object.keys(violinData).length * 30 + 100,
                        )}
                      />
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ================================================================ */}
      {/*  SECTION 2: Pathway Enrichment                                   */}
      {/* ================================================================ */}
      <div className="flex-shrink-0 rounded-xl border border-slate-200 bg-white shadow-sm">
        <button
          onClick={() => setEnrichmentSectionOpen(!enrichmentSectionOpen)}
          className="flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold text-slate-700 hover:bg-slate-50"
        >
          {enrichmentSectionOpen ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          Pathway Enrichment
          {enrichment && (
            <span className="ml-2 rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700">
              {enrichment.results.length} terms ({enrichment.source})
            </span>
          )}
        </button>

        {enrichmentSectionOpen && (
          <div className="space-y-3 border-t border-slate-100 px-4 pb-4 pt-3">
            {/* Controls row */}
            <div className="flex flex-wrap items-end gap-4">
              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-slate-600">
                  Obs Column
                </label>
                <select
                  value={selectedColumn}
                  onChange={(e) => setSelectedColumn(e.target.value)}
                  disabled={isLoadingColumns || availableColumns.length === 0}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100 disabled:opacity-50"
                >
                  {availableColumns.length === 0 && (
                    <option value="">
                      {isLoadingColumns
                        ? "Loading..."
                        : "No columns with markers"}
                    </option>
                  )}
                  {availableColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-slate-600">
                  Cluster / Group
                </label>
                <select
                  value={selectedGroup}
                  onChange={(e) => setSelectedGroup(e.target.value)}
                  disabled={isLoadingGroups || groups.length === 0}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100 disabled:opacity-50"
                >
                  {groups.length === 0 && (
                    <option value="">
                      {isLoadingGroups
                        ? "Loading..."
                        : "No groups available"}
                    </option>
                  )}
                  {groups.map((g) => (
                    <option key={g} value={g}>
                      {g}
                      {enrichmentStatus[g] ? " \u2713" : ""}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-slate-600">
                  Top N genes
                </label>
                <input
                  type="number"
                  value={nGenes}
                  onChange={(e) =>
                    setNGenes(
                      Math.max(
                        10,
                        Math.min(500, parseInt(e.target.value) || 100),
                      ),
                    )
                  }
                  min={10}
                  max={500}
                  className="w-24 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
                />
              </div>

              <button
                onClick={handleComputeEnrichment}
                disabled={
                  isEnrichLoading ||
                  !selectedGroup ||
                  selectedCollections.size === 0
                }
                className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isEnrichLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Computing...
                  </>
                ) : (
                  <>
                    <PlayCircle className="h-4 w-4" />
                    Compute Enrichment
                  </>
                )}
              </button>

              <button
                onClick={handleComputeAllGroups}
                disabled={
                  isBatchComputing ||
                  groups.length === 0 ||
                  selectedCollections.size === 0
                }
                className="flex items-center gap-2 rounded-lg border border-blue-300 bg-blue-50 px-4 py-2 text-sm font-semibold text-blue-700 shadow-sm transition-colors hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isBatchComputing ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    {batchProgress.current}/{batchProgress.total}...
                  </>
                ) : (
                  <>
                    <PlayCircle className="h-4 w-4" />
                    Compute All Groups
                  </>
                )}
              </button>

              {sortedResults.length > 0 && (
                <button
                  onClick={handleExportCSV}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50"
                >
                  <Download className="h-3.5 w-3.5" />
                  Export CSV
                </button>
              )}
            </div>

            {/* MSigDB Collection tree */}
            <MSigDBCollectionTree
              selected={selectedCollections}
              onChange={setSelectedCollections}
            />

            {/* Error */}
            {enrichError && (
              <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
                <AlertCircle className="h-4 w-4 flex-shrink-0" />
                {enrichError}
              </div>
            )}

            {/* No results placeholder */}
            {!enrichment && !isEnrichLoading && !enrichError && (
              <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
                <p className="text-sm text-slate-500">
                  Select a group above and click "Compute Enrichment" to analyse
                  pathways.
                </p>
              </div>
            )}

            {/* Results: bar chart + table */}
            {enrichment && sortedResults.length > 0 && (
              <>
                {/* Bar chart */}
                <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                  <div className="mb-2 flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-700">
                      Top {Math.min(topN, topResults.length)} Enriched Terms
                    </h3>
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-slate-500">
                        Show top:
                      </label>
                      <select
                        value={topN}
                        onChange={(e) => setTopN(parseInt(e.target.value))}
                        className="rounded border border-slate-200 px-2 py-1 text-xs"
                      >
                        <option value={10}>10</option>
                        <option value={20}>20</option>
                        <option value={50}>50</option>
                      </select>
                    </div>
                  </div>
                  <Plot
                    data={[
                      {
                        type: "bar",
                        orientation: "h",
                        y: topResults
                          .map((r) =>
                            r.term.length > 50
                              ? r.term.substring(0, 47) + "..."
                              : r.term,
                          )
                          .reverse(),
                        x: topResults
                          .map((r) =>
                            -Math.log10(Math.max(r.pvalue, 1e-300)),
                          )
                          .reverse(),
                        marker: {
                          color: topResults
                            .map((r) =>
                              -Math.log10(
                                Math.max(r.adjusted_pvalue, 1e-300),
                              ),
                            )
                            .reverse(),
                          colorscale: "Viridis",
                          colorbar: {
                            title: {
                              text: "-log10(adj. p)",
                              font: { size: 10 },
                            },
                            thickness: 12,
                          },
                        },
                        text: topResults
                          .map((r) => `p=${formatPValue(r.pvalue)}`)
                          .reverse(),
                        hovertemplate:
                          "<b>%{y}</b><br>-log10(p): %{x:.2f}<br>%{text}<extra></extra>",
                      },
                    ]}
                    layout={{
                      height: Math.max(300, topResults.length * 22 + 80),
                      margin: { l: 300, r: 80, t: 20, b: 50 },
                      xaxis: {
                        title: {
                          text: "-log10(p-value)",
                          font: { size: 12, color: "#64748b" },
                        },
                        tickfont: { size: 10, color: "#64748b" },
                        gridcolor: "#f1f5f9",
                      },
                      yaxis: {
                        tickfont: { size: 10, color: "#334155" },
                        automargin: true,
                      },
                      paper_bgcolor: "white",
                      plot_bgcolor: "white",
                      font: { family: "Inter, system-ui, sans-serif" },
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    useResizeHandler
                    className="w-full"
                  />
                </div>

                {/* Results table */}
                <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
                  <div className="max-h-[400px] overflow-auto">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 z-10">
                        <tr className="border-b border-slate-200 bg-slate-50">
                          <th
                            onClick={() => handleSort("term")}
                            className="cursor-pointer px-4 py-2.5 text-left font-medium text-slate-600 hover:text-slate-800"
                          >
                            <span className="flex items-center gap-1">
                              Term {sortIcon("term")}
                            </span>
                          </th>
                          <th className="px-3 py-2.5 text-left font-medium text-slate-600">
                            Collection
                          </th>
                          <th
                            onClick={() => handleSort("pvalue")}
                            className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                          >
                            <span className="flex items-center justify-end gap-1">
                              P-value {sortIcon("pvalue")}
                            </span>
                          </th>
                          <th
                            onClick={() => handleSort("adjusted_pvalue")}
                            className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                          >
                            <span className="flex items-center justify-end gap-1">
                              Adj. P-value {sortIcon("adjusted_pvalue")}
                            </span>
                          </th>
                          <th
                            onClick={() => handleSort("overlap_count")}
                            className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                          >
                            <span className="flex items-center justify-end gap-1">
                              Overlap {sortIcon("overlap_count")}
                            </span>
                          </th>
                          <th className="px-4 py-2.5 text-left font-medium text-slate-600">
                            Genes
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {sortedResults.map((r, idx) => (
                          <tr
                            key={`${r.term}-${idx}`}
                            className="cursor-pointer border-b border-slate-50 transition-colors hover:bg-blue-50"
                            onClick={() => handleTermClick(r.term)}
                            title="Click to score this gene set on UMAP"
                          >
                            <td className="max-w-xs truncate px-4 py-2 text-slate-800">
                              {r.term}
                            </td>
                            <td className="px-3 py-2">
                              {r.collection && (
                                <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                                  {r.collection}
                                </span>
                              )}
                            </td>
                            <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                              {formatPValue(r.pvalue)}
                            </td>
                            <td className="px-4 py-2 text-right tabular-nums">
                              <span
                                className={
                                  r.adjusted_pvalue < 0.05
                                    ? "font-medium text-blue-600"
                                    : "text-slate-600"
                                }
                              >
                                {formatPValue(r.adjusted_pvalue)}
                              </span>
                            </td>
                            <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                              {r.overlap_count}/{r.gene_count}
                            </td>
                            <td className="max-w-xs truncate px-4 py-2 font-mono text-xs text-slate-500">
                              {r.genes.slice(0, 5).join(", ")}
                              {r.genes.length > 5 &&
                                ` +${r.genes.length - 5} more`}
                            </td>
                          </tr>
                        ))}
                        {sortedResults.length === 0 && (
                          <tr>
                            <td
                              colSpan={6}
                              className="px-4 py-8 text-center text-slate-400"
                            >
                              No enrichment results
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
