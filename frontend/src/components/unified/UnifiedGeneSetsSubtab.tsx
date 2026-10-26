import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { Loader2, Search, Check, ChevronDown, ChevronRight } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useCustomGeneSetsStore } from "@/stores/customGeneSetsStore";
import { apiFetch } from "@/api/client";
import type { CollectionChoice } from "@/api/genesets";
import { scoreGeneSet, type ScoreMethod } from "@/api/genesetScore";
import type { OnScoreGeneSet } from "@/hooks/useGeneSetScoring";
import { useScoringMethodStore } from "@/stores/scoringMethodStore";
import { ScoringMethodChoice } from "@/components/common/ScoringMethodChoice";
import { errorMessage } from "@/lib/errorMessage";
import { ScoreDepthCheck } from "@/components/unified/ScoreDepthCheck";
import {
  CollectionOptions,
  CustomCollectionsGroup,
} from "@/components/panels/CustomCollectionsGroup";

/** Sets fetched per collection (the search endpoint's cap); the list says so when the collection has more. */
const SETS_LISTED = 200;

interface CollectionMeta extends CollectionChoice {
  category: string;
  subcategory?: string;
  n_sets: number;
  available: boolean;
}

interface GeneSetSearchResult {
  name: string;
  collection: string;
  /** Set for a custom collection. */
  collection_name?: string;
  n_genes: number;
  genes: string[];
}

/** What the subtab says about the last score: counts, AUCell's ranking depth, and
 *  how ssGSEA / GSVA computed it. */
interface ScoreSummary {
  method: ScoreMethod;
  found: number;
  missing: number;
  undetected: number;
  maxRank: number | null;
  shareBelow: number | null;
  note: string | null;
}

interface UnifiedGeneSetsSubtabProps {
  onScoreComplete: OnScoreGeneSet;
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
}

export function UnifiedGeneSetsSubtab({
  onScoreComplete,
  groupByColumn,
  setGroupByColumn,
}: UnifiedGeneSetsSubtabProps) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);

  // Collections from backend
  const [collections, setCollections] = useState<CollectionMeta[]>([]);
  const [selectedCollection, setSelectedCollection] = useState("");

  // Gene sets within selected collection
  const [geneSets, setGeneSets] = useState<GeneSetSearchResult[]>([]);
  const [isLoadingSets, setIsLoadingSets] = useState(false);
  const [filterQuery, setFilterQuery] = useState("");

  // Selected gene set
  const [selectedGeneSet, setSelectedGeneSet] = useState<GeneSetSearchResult | null>(null);

  // Scoring state
  const [isScoring, setIsScoring] = useState(false);
  const [scoreResult, setScoreResult] = useState<ScoreSummary | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const method = useScoringMethodStore((s) => s.method);

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

  // Fetch collections on mount, and again when a custom one is added, renamed or deleted
  const customVersion = useCustomGeneSetsStore((s) => s.version);
  const [showCustom, setShowCustom] = useState(false);
  useEffect(() => {
    if (!datasetId) return;
    apiFetch<{ dataset_id: string; collections: CollectionMeta[] }>(
      `/datasets/${datasetId}/genesets/collections`,
    )
      .then((data) => {
        const available = (data.collections ?? []).filter((c) => c.available);
        setCollections(available);
        // Keep the current choice unless it was a custom collection just deleted.
        setSelectedCollection((cur) =>
          available.some((c) => c.id === cur) ? cur : (available[0]?.id ?? ""),
        );
      })
      .catch(() => {});
  }, [datasetId, customVersion]);

  // Fetch gene sets when collection changes
  useEffect(() => {
    if (!datasetId || !selectedCollection) return;
    setIsLoadingSets(true);
    setSelectedGeneSet(null);
    setScoreResult(null);
    apiFetch<{ results: GeneSetSearchResult[] }>(
      `/datasets/${datasetId}/genesets/search?q=&collection=${encodeURIComponent(selectedCollection)}&limit=${SETS_LISTED}`,
    )
      .then((data) => {
        setGeneSets(data.results ?? []);
      })
      .catch(() => setGeneSets([]))
      .finally(() => setIsLoadingSets(false));
  }, [datasetId, selectedCollection]);

  // Filter gene sets by search query
  const filteredGeneSets = useMemo(() => {
    if (!filterQuery.trim()) return geneSets;
    const q = filterQuery.trim().toLowerCase();
    return geneSets.filter((gs) => gs.name.toLowerCase().includes(q));
  }, [geneSets, filterQuery]);

  // Cache of already-computed scores, so re-selecting a previously-scored gene
  // set restores its overlay instantly instead of forcing a re-score. Keyed by
  // dataset id + gene-set name (scores are per-dataset).
  const scoreCache = useRef<Map<string, ScoreSummary & { scores: Float32Array }>>(new Map());
  // The collection is part of the key: a custom set may share an MSigDB set's name;
  // and the method: each set can be on the plot either way.
  const cacheKey = (gs: GeneSetSearchResult, m: ScoreMethod) => `${datasetId}::${m}::${gs.collection}::${gs.name}`;

  // Select a gene set from the list — re-apply a cached score if we have one.
  const handleSelectGeneSet = useCallback(
    (gs: GeneSetSearchResult) => {
      setSelectedGeneSet(gs);
      setScoreError(null);
      const cached = scoreCache.current.get(cacheKey(gs, method));
      if (cached) {
        const { scores, ...summary } = cached;
        setScoreResult(summary);
        onScoreComplete(scores, gs.name, gs.genes, undefined, cached.method);
      } else {
        setScoreResult(null);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [datasetId, onScoreComplete, method],
  );

  // Score gene set using correct API fields
  const handleScore = useCallback(async () => {
    if (!datasetId || !selectedGeneSet || selectedGeneSet.genes.length === 0) return;
    setIsScoring(true);
    setScoreResult(null);
    setScoreError(null);
    try {
      const response = await scoreGeneSet(datasetId, selectedGeneSet.genes, selectedGeneSet.name, method);
      const scores = new Float32Array(response.scores);
      const summary: ScoreSummary = {
        method: response.method,
        found: response.genes_found.length,
        missing: response.genes_missing.length,
        undetected: response.genes_undetected?.length ?? 0,
        maxRank: response.max_rank,
        shareBelow: response.share_below_max_rank,
        note: response.note ?? null,
      };
      scoreCache.current.set(cacheKey(selectedGeneSet, method), { ...summary, scores });
      setScoreResult(summary);
      onScoreComplete(scores, selectedGeneSet.name, selectedGeneSet.genes, undefined, response.method);
    } catch (err) {
      console.error("Scoring failed:", err);
      setScoreError(errorMessage(err));
    } finally {
      setIsScoring(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, selectedGeneSet, onScoreComplete, method]);

  if (!dataset || !datasetId) {
    return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  }

  return (
    <div className="space-y-3">
      {/* Collection dropdown */}
      <div>
        <label className="mb-1 block text-[11px] font-medium text-slate-500">
          Collection
        </label>
        <select
          value={selectedCollection}
          onChange={(e) => {
            setSelectedCollection(e.target.value);
            setFilterQuery("");
          }}
          aria-label="Gene-set collection"
          className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
        >
          <CollectionOptions collections={collections} />
        </select>
        <button
          type="button"
          onClick={() => setShowCustom((v) => !v)}
          className="mt-1 flex items-center gap-0.5 text-[10px] text-blue-600 hover:underline"
        >
          {showCustom ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          Your own collections (upload, rename, delete)
        </button>
        {showCustom && (
          <div className="mt-1 rounded-lg border border-slate-200 bg-white p-1.5">
            <CustomCollectionsGroup datasetId={datasetId} compact />
          </div>
        )}
      </div>

      {/* Search filter for gene sets */}
      <div className="relative">
        <label className="mb-1 block text-[11px] font-medium text-slate-500">
          Gene Sets
        </label>
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="Filter gene sets..."
            className="w-full rounded border border-slate-300 bg-white py-1.5 pl-8 pr-2 text-xs text-slate-700 placeholder-slate-400"
          />
          {isLoadingSets && (
            <div className="absolute right-2 top-1/2 -translate-y-1/2">
              <Loader2 className="h-3 w-3 animate-spin text-blue-500" />
            </div>
          )}
        </div>
      </div>

      {/* Scrollable gene set list */}
      <div className="max-h-48 overflow-auto rounded-lg border border-slate-200">
        {filteredGeneSets.length === 0 ? (
          <div className="px-3 py-4 text-center text-xs text-slate-400">
            {isLoadingSets ? "Loading..." : geneSets.length === 0 ? "No gene sets available" : "No matches"}
          </div>
        ) : (
          filteredGeneSets.map((gs) => (
            <button
              key={`${gs.collection}-${gs.name}`}
              onClick={() => handleSelectGeneSet(gs)}
              className={`flex w-full items-center justify-between border-b border-slate-50 px-3 py-1.5 text-left text-xs transition-colors hover:bg-blue-50 ${
                selectedGeneSet?.name === gs.name && selectedGeneSet?.collection === gs.collection
                  ? "bg-blue-50 font-medium text-blue-700"
                  : "text-slate-700"
              }`}
            >
              <span className="truncate">{gs.name}</span>
              <span className="ml-2 flex-shrink-0 text-[10px] text-slate-400">
                {gs.n_genes} genes
              </span>
            </button>
          ))
        )}
      </div>

      {(() => {
        const total = collections.find((c) => c.id === selectedCollection)?.n_sets ?? 0;
        return geneSets.length >= SETS_LISTED && total > geneSets.length ? (
          <p className="text-[10px] leading-snug text-slate-400" data-testid="genesets-cap">
            The first {SETS_LISTED} of the collection&apos;s {total.toLocaleString()} sets, in file order; the filter above
            searches only these. Score any other set from the Enrichment tab or a gene list.
          </p>
        ) : null;
      })()}

      {/* Selected gene set info + Score button */}
      {selectedGeneSet && (
        <div className="rounded-lg border border-blue-100 bg-blue-50/50 p-2.5">
          <div className="mb-1 text-xs font-semibold text-blue-700">{selectedGeneSet.name}</div>
          <div className="mb-2 text-[10px] text-blue-500">
            {selectedGeneSet.n_genes} genes in{" "}
            {selectedGeneSet.collection_name || selectedGeneSet.collection}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleScore}
              disabled={isScoring || selectedGeneSet.genes.length === 0}
              className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
            >
              {isScoring ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Check className="h-3 w-3" />
              )}
              Score
            </button>
            <ScoringMethodChoice label="with" />
            {scoreResult && (
              <span className="flex items-center gap-2 text-[11px]">
                <span className="text-emerald-600">{scoreResult.found} found</span>
                {scoreResult.missing > 0 && (
                  <span className="text-amber-600">{scoreResult.missing} missing</span>
                )}
                {scoreResult.undetected > 0 && (
                  <span
                    className="text-slate-500"
                    title={scoreResult.method === "gsva"
                      ? "In the dataset but the same in every cell (mostly never detected): GSVA leaves them out of the set"
                      : "In the dataset but expressed in no cell: AUCell leaves them out of the set"}
                  >
                    {scoreResult.undetected} never detected
                  </span>
                )}
              </span>
            )}
          </div>
          {scoreResult?.method === "aucell" && scoreResult.maxRank != null && (
            <p className="mt-1 text-[10px] leading-snug text-slate-500" data-testid="aucell-note">
              AUCell over each cell&apos;s top {scoreResult.maxRank.toLocaleString()} genes (5% of those detected)
              {scoreResult.shareBelow
                ? `; ${Math.round(scoreResult.shareBelow * 100)}% of cells detect fewer, so their lowest counted ranks are ties among undetected genes.`
                : "."}
            </p>
          )}
          {scoreResult?.note && (
            <p className="mt-1 text-[10px] leading-snug text-slate-500" data-testid="score-note">
              {scoreResult.note}.
            </p>
          )}
          {scoreError && <p className="mt-1 text-[11px] text-red-600" role="alert">{scoreError}</p>}
          <ScoreDepthCheck key={`${datasetId}::${selectedGeneSet.collection}::${selectedGeneSet.name}`} datasetId={datasetId} genes={selectedGeneSet.genes} />
        </div>
      )}

      {/* Group by */}
      <div>
        <label className="mb-1 block text-[11px] font-medium text-slate-500">
          Group by (violin)
        </label>
        <select
          aria-label="Group by (violin)"
          value={groupByColumn}
          onChange={(e) => setGroupByColumn(e.target.value)}
          className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
        >
          {categoricalColumns.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name} ({col.n_unique} groups)
            </option>
          ))}
        </select>
      </div>

      <p className="text-[10px] text-slate-400">
        Select a gene set and click Score to color the scatter plot by its activity
      </p>
    </div>
  );
}
