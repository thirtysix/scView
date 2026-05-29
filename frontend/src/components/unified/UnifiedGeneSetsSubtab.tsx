import { useState, useCallback, useEffect, useMemo } from "react";
import { Loader2, Search, Check } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { apiFetch } from "@/api/client";

interface CollectionMeta {
  id: string;
  name: string;
  category: string;
  subcategory?: string;
  n_sets: number;
  available: boolean;
}

interface GeneSetSearchResult {
  name: string;
  collection: string;
  n_genes: number;
  genes: string[];
}

interface ScoreResponse {
  score_name: string;
  n_cells: number;
  scores: number[];
  min_score: number;
  max_score: number;
  genes_found: string[];
  genes_missing: string[];
}

interface UnifiedGeneSetsSubtabProps {
  onScoreComplete: (scores: Float32Array, name: string) => void;
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
  const [scoreResult, setScoreResult] = useState<{ found: number; missing: number } | null>(null);

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

  // Fetch collections on mount
  useEffect(() => {
    if (!datasetId) return;
    apiFetch<{ dataset_id: string; collections: CollectionMeta[] }>(
      `/datasets/${datasetId}/genesets/collections`,
    )
      .then((data) => {
        const available = (data.collections ?? []).filter((c) => c.available);
        setCollections(available);
        if (available.length > 0 && !selectedCollection) {
          setSelectedCollection(available[0]!.id);
        }
      })
      .catch(() => {});
  }, [datasetId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch gene sets when collection changes
  useEffect(() => {
    if (!datasetId || !selectedCollection) return;
    setIsLoadingSets(true);
    setSelectedGeneSet(null);
    setScoreResult(null);
    apiFetch<{ results: GeneSetSearchResult[] }>(
      `/datasets/${datasetId}/genesets/search?q=&collection=${encodeURIComponent(selectedCollection)}&limit=200`,
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

  // Select a gene set from the list
  const handleSelectGeneSet = useCallback((gs: GeneSetSearchResult) => {
    setSelectedGeneSet(gs);
    setScoreResult(null);
  }, []);

  // Score gene set using correct API fields
  const handleScore = useCallback(async () => {
    if (!datasetId || !selectedGeneSet || selectedGeneSet.genes.length === 0) return;
    setIsScoring(true);
    setScoreResult(null);
    try {
      const response = await apiFetch<ScoreResponse>(
        `/datasets/${datasetId}/genesets/score`,
        {
          method: "POST",
          body: JSON.stringify({
            gene_set: selectedGeneSet.genes,
            score_name: selectedGeneSet.name,
          }),
        },
      );

      const scores = new Float32Array(response.scores);
      setScoreResult({
        found: response.genes_found.length,
        missing: response.genes_missing.length,
      });
      onScoreComplete(scores, selectedGeneSet.name);
    } catch (err) {
      console.error("Scoring failed:", err);
    } finally {
      setIsScoring(false);
    }
  }, [datasetId, selectedGeneSet, onScoreComplete]);

  if (!dataset || !datasetId) {
    return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  }

  return (
    <div className="space-y-3">
      {/* Collection dropdown */}
      <div>
        <label className="mb-1 block text-[11px] font-medium text-slate-500">
          MSigDB Collection
        </label>
        <select
          value={selectedCollection}
          onChange={(e) => {
            setSelectedCollection(e.target.value);
            setFilterQuery("");
          }}
          className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
        >
          {collections.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name} ({c.n_sets} sets)
            </option>
          ))}
        </select>
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

      {/* Selected gene set info + Score button */}
      {selectedGeneSet && (
        <div className="rounded-lg border border-blue-100 bg-blue-50/50 p-2.5">
          <div className="mb-1 text-xs font-semibold text-blue-700">{selectedGeneSet.name}</div>
          <div className="mb-2 text-[10px] text-blue-500">
            {selectedGeneSet.n_genes} genes in {selectedGeneSet.collection}
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
            {scoreResult && (
              <span className="flex items-center gap-2 text-[11px]">
                <span className="text-emerald-600">{scoreResult.found} found</span>
                {scoreResult.missing > 0 && (
                  <span className="text-amber-600">{scoreResult.missing} missing</span>
                )}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Group by */}
      <div>
        <label className="mb-1 block text-[11px] font-medium text-slate-500">
          Group by (violin)
        </label>
        <select
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
