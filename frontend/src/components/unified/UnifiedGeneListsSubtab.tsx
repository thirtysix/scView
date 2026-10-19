import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  FolderPlus,
  Grid2x2,
  Loader2,
  Plus,
  Sparkles,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { enrichGeneLists, type EnrichmentResult, type GeneListEnrichmentResponse } from "@/api/enrichment";
import { collectionLabel, customCollectionFromLists, GeneSetUploadError } from "@/api/genesets";
import { GeneSearch } from "@/components/common/GeneSearch";
import { InfoTip } from "@/components/common/InfoTip";
import { DEFAULT_MSIGDB_COLLECTIONS, MSigDBCollectionTree } from "@/components/panels/MSigDBCollectionTree";
import { EnrichmentTermsTable } from "@/components/unified/EnrichmentTermsTable";
import { useDatasetGenes, type DatasetGenes } from "@/hooks/useDatasetGenes";
import { useEnrichmentSort } from "@/hooks/useEnrichmentSort";
import { useGeneSetScoring } from "@/hooks/useGeneSetScoring";
import { MAX_GRID_GENES, MIN_GRID_GENES } from "@/lib/geneCoexpression";
import { MAX_PLOT_GENES, parseGeneList } from "@/lib/geneList";
import { downloadText, exportGeneLists, genePresence, type GeneListExportFormat } from "@/lib/geneLists";
import { exportFilename } from "@/lib/exportFilename";
import { useCustomGeneSetsStore } from "@/stores/customGeneSetsStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useGeneGridStore } from "@/stores/geneGridStore";
import { useGeneListsStorageStatus, useGeneListsStore, type GeneList } from "@/stores/geneListsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { useViewStore } from "@/stores/viewStore";

interface Props {
  /** A gene chip click: the gene's overlay + violin (UnifiedViewPanel.handleGeneClick) */
  onGeneClick: (gene: string) => void;
  /** A list or term scored as a gene set: the score overlay (handleScoreComplete) */
  onScoreGeneSet: (scores: Float32Array, name: string, genes?: string[]) => void;
  /** The gene on the scatter now, if any (its chip is marked) */
  activeGene?: string | null;
}

const btn =
  "flex items-center gap-1 rounded border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50";

/**
 * The Gene lists tab: the user's named gene lists (kept across reloads and
 * datasets, stores/geneListsStore.ts) on the left, the chosen list on the
 * right, checked against the open dataset, with what can be done with it:
 * overlay a gene, a feature grid, the marker dot plot / heatmap, a score,
 * enrichment, or saving the lists as a gene-set collection.
 */
export function UnifiedGeneListsSubtab({ onGeneClick, onScoreGeneSet, activeGene }: Props) {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const lists = useGeneListsStore((s) => s.lists);
  const selectedId = useGeneListsStore((s) => s.selectedId);
  const select = useGeneListsStore((s) => s.select);
  const create = useGeneListsStore((s) => s.create);
  const importFile = useGeneListsStore((s) => s.importFile);
  const storageFailed = useGeneListsStorageStatus((s) => s.failed);
  const dsGenes = useDatasetGenes(datasetId);
  const [importError, setImportError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const selected = lists.find((l) => l.id === selectedId) ?? lists[0] ?? null;

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setImportError(null);
    try {
      importFile(file.name, await file.text());
    } catch (err) {
      setImportError(err instanceof Error ? err.message : String(err));
    }
  };

  const exportAll = (format: GeneListExportFormat) =>
    downloadText(`scview_gene_lists.${format}`, exportGeneLists(lists, format));

  return (
    <div className="flex flex-col gap-2 text-xs">
      {storageFailed && (
        <p role="alert" className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-amber-800">
          This browser is not keeping the gene lists (storage is blocked or full), so they will be lost on reload.
          Export them to keep them.
        </p>
      )}
      <div className="grid grid-cols-[minmax(110px,36%)_1fr] gap-2">
        {/* Left: the lists */}
        <div className="flex min-w-0 flex-col gap-1.5">
          <div className="flex flex-wrap gap-1">
            <button type="button" className={btn} onClick={() => create("New list")} title="An empty list to add genes to">
              <Plus className="h-3 w-3" />
              New
            </button>
            <button
              type="button"
              className={btn}
              onClick={() => fileRef.current?.click()}
              title="A .txt or .csv of genes, or a .gmt (one list per set)"
            >
              <Upload className="h-3 w-3" />
              Import
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.csv,.tsv,.gmt"
              className="hidden"
              aria-label="Import gene lists from a file"
              onChange={onFile}
            />
          </div>
          {importError && <p className="text-red-600">{importError}</p>}
          {lists.length === 0 ? (
            <p className="py-2 text-slate-400">
              No gene lists yet. Make one here, or add genes from the Markers, DE or Expression tabs.
            </p>
          ) : (
            <ul aria-label="Gene lists" className="flex max-h-[calc(100vh-420px)] min-h-[6rem] flex-col gap-0.5 overflow-y-auto">
              {lists.map((l) => (
                <ListRow
                  key={l.id}
                  list={l}
                  selected={l.id === selected?.id}
                  dsGenes={dsGenes}
                  onSelect={() => select(l.id)}
                />
              ))}
            </ul>
          )}
          {lists.length > 1 && (
            <div className="flex flex-wrap items-center gap-1 text-[11px] text-slate-500">
              All:
              {(["csv", "gmt"] as const).map((f) => (
                <button key={f} type="button" className="text-primary hover:underline" onClick={() => exportAll(f)}>
                  .{f}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Right: the chosen list */}
        <div className="min-w-0">
          {selected ? (
            <ListDetail
              key={selected.id}
              list={selected}
              allLists={lists}
              datasetId={datasetId}
              dsGenes={dsGenes}
              onGeneClick={onGeneClick}
              onScoreGeneSet={onScoreGeneSet}
              activeGene={activeGene ?? null}
            />
          ) : (
            <p className="py-2 text-slate-400">Pick or make a list.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function ListRow({
  list,
  selected,
  dsGenes,
  onSelect,
}: {
  list: GeneList;
  selected: boolean;
  dsGenes: DatasetGenes | null;
  onSelect: () => void;
}) {
  const missing = dsGenes ? list.genes.filter((g) => !dsGenes.set.has(g)).length : null;
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        aria-current={selected}
        className={`w-full rounded px-1.5 py-1 text-left ${selected ? "bg-blue-50 text-blue-800" : "text-slate-700 hover:bg-slate-50"}`}
      >
        <div className="truncate font-medium" title={list.name}>
          {list.name}
        </div>
        <div className="text-[10px] text-slate-400">
          {list.genes.length} gene{list.genes.length === 1 ? "" : "s"}
          {missing ? <span className="text-amber-700"> · {missing} missing</span> : null}
        </div>
      </button>
    </li>
  );
}

function ListDetail({
  list,
  allLists,
  datasetId,
  dsGenes,
  onGeneClick,
  onScoreGeneSet,
  activeGene,
}: {
  list: GeneList;
  allLists: GeneList[];
  datasetId: string | null;
  dsGenes: DatasetGenes | null;
  onGeneClick: (gene: string) => void;
  onScoreGeneSet: (scores: Float32Array, name: string, genes?: string[]) => void;
  activeGene: string | null;
}) {
  const rename = useGeneListsStore((s) => s.rename);
  const duplicate = useGeneListsStore((s) => s.duplicate);
  const remove = useGeneListsStore((s) => s.remove);
  const addGenes = useGeneListsStore((s) => s.addGenes);
  const removeGene = useGeneListsStore((s) => s.removeGene);
  const replaceGenes = useGeneListsStore((s) => s.replaceGenes);
  const geneGrid = useSplitViewStore((s) => s.geneGrid);
  const [name, setName] = useState(list.name);
  useEffect(() => setName(list.name), [list.name]);
  const [panel, setPanel] = useState<"enrich" | "save" | null>(null);

  const presence = useMemo(
    () => (dsGenes ? genePresence(list.genes, dsGenes.set, dsGenes.byUpper) : null),
    [list.genes, dsGenes],
  );
  // Without the dataset's genes (none open, or loading) nothing is marked missing.
  const present = presence ? presence.present : list.genes;
  const missing = useMemo(() => new Set(presence?.missing ?? []), [presence]);
  const ready = !!datasetId && !!presence;

  const { scoreGenes, scoringName } = useGeneSetScoring(datasetId, onScoreGeneSet, onGeneClick);

  const gridGenes = present.slice(0, MAX_GRID_GENES);
  const gridOn =
    geneGrid?.kind === "features" &&
    geneGrid.datasetId === datasetId &&
    geneGrid.genes.length === gridGenes.length &&
    geneGrid.genes.every((g, i) => g === gridGenes[i]);
  const showGrid = () => {
    if (!datasetId) return;
    if (gridOn) {
      useSplitViewStore.getState().setGeneGrid(null);
      return;
    }
    // The Expression tab's "Several genes" picks follow, as when the grid is opened there.
    useGeneGridStore.getState().setGenes(datasetId, gridGenes);
    useSplitViewStore.getState().setGeneGrid({ datasetId, kind: "features", genes: gridGenes });
  };
  const showMarkerPlot = (view: "dotplot" | "heatmap") => {
    const v = useViewStore.getState();
    v.setMarkerGenes(present.slice(0, MAX_PLOT_GENES));
    v.setMarkerView(view);
    v.setPanel("markers");
  };
  const adoptDatasetSpelling = () => {
    if (!presence) return;
    replaceGenes(list.id, list.genes.map((g) => presence.otherCase.get(g) ?? g));
  };
  const commitName = () => {
    if (name.trim() && name !== list.name) rename(list.id, name);
    else setName(list.name);
  };
  const exportOne = (format: GeneListExportFormat) =>
    downloadText(`${exportFilename(null, `gene_list_${list.name}`)}.${format}`, exportGeneLists([list], format));

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={commitName}
          onKeyDown={(e) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
            if (e.key === "Escape") setName(list.name);
          }}
          aria-label="List name"
          maxLength={200}
          className="min-w-0 flex-1 rounded border border-slate-200 px-1.5 py-1 text-xs font-semibold text-slate-700 focus:border-primary focus:outline-none"
        />
        <button type="button" title="Duplicate this list" aria-label="Duplicate list" className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600" onClick={() => duplicate(list.id)}>
          <Copy className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          title="Delete this list"
          aria-label="Delete list"
          className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
          onClick={() => {
            if (window.confirm(`Delete the gene list "${list.name}"?`)) remove(list.id);
          }}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
      {list.source && <p className="-mt-1 truncate text-[10px] text-slate-400" title={list.source}>From {list.source}</p>}

      {datasetId && (
        <GeneSearch
          datasetId={datasetId}
          multiple
          onSelect={(text) => addGenes(list.id, parseGeneList(text))}
          placeholder="Add genes (or paste a list)…"
          className="w-full"
        />
      )}

      <div className="flex flex-wrap items-baseline gap-x-2 text-[11px] text-slate-500">
        <span>
          {list.genes.length} gene{list.genes.length === 1 ? "" : "s"}
        </span>
        {presence && presence.missing.length > 0 && (
          <span className="text-amber-700">{presence.missing.length} not in this dataset</span>
        )}
        {presence && presence.otherCase.size > 0 && (
          <button
            type="button"
            onClick={adoptDatasetSpelling}
            title={[...presence.otherCase].slice(0, 5).map(([a, b]) => `${a} → ${b}`).join(", ")}
            className="text-primary hover:underline"
          >
            Use the dataset&apos;s spelling for {presence.otherCase.size}
          </button>
        )}
      </div>

      {list.genes.length > 0 && (
        <div aria-label="Genes in the list" className="flex max-h-48 flex-wrap gap-1 overflow-y-auto">
          {list.genes.map((g) => {
            const absent = missing.has(g);
            const other = presence?.otherCase.get(g);
            return (
              <span
                key={g}
                className={`flex items-center rounded-full border py-0.5 pl-2 pr-1 ${
                  absent
                    ? "border-dashed border-slate-200 bg-slate-50 text-slate-400"
                    : g === activeGene
                      ? "border-blue-300 bg-blue-50 text-blue-700"
                      : "border-slate-300 bg-white text-slate-700"
                }`}
              >
                <button
                  type="button"
                  disabled={absent || !datasetId}
                  onClick={() => onGeneClick(g)}
                  title={absent ? `Not in this dataset${other ? ` (it has ${other})` : ""}` : `Show ${g} on the plot`}
                  className="disabled:cursor-default"
                >
                  {g}
                </button>
                <button
                  type="button"
                  onClick={() => removeGene(list.id, g)}
                  aria-label={`Remove ${g}`}
                  className="ml-0.5 rounded-full p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            );
          })}
        </div>
      )}

      {datasetId && (
        <div className="flex flex-col gap-1.5 border-t border-slate-100 pt-2">
          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              className={`${btn} ${gridOn ? "border-blue-300 bg-blue-50 text-blue-700" : ""}`}
              aria-pressed={gridOn}
              disabled={!ready || gridGenes.length < MIN_GRID_GENES}
              onClick={showGrid}
              title={`The embedding coloured by each gene, side by side (the first ${MAX_GRID_GENES} genes in the dataset)`}
            >
              <Grid2x2 className="h-3.5 w-3.5" />
              {gridOn ? "Hide feature grid" : "Feature grid"}
            </button>
            <button type="button" className={btn} disabled={!ready || present.length === 0} onClick={() => showMarkerPlot("dotplot")} title="Open the Marker Genes dot plot with these genes">
              Dot plot
            </button>
            <button type="button" className={btn} disabled={!ready || present.length === 0} onClick={() => showMarkerPlot("heatmap")} title="Open the Marker Genes heatmap with these genes">
              Heatmap
            </button>
            <button
              type="button"
              className={btn}
              disabled={!ready || present.length === 0 || !!scoringName}
              onClick={() => scoreGenes(present, list.name)}
              title="Score each cell for these genes together (scanpy score_genes) and colour the plot by it"
            >
              {scoringName ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              Score
            </button>
            <button
              type="button"
              className={`${btn} ${panel === "enrich" ? "border-blue-300 bg-blue-50 text-blue-700" : ""}`}
              aria-pressed={panel === "enrich"}
              disabled={!ready || present.length === 0}
              onClick={() => setPanel(panel === "enrich" ? null : "enrich")}
            >
              Enrichment
            </button>
            <button
              type="button"
              className={`${btn} ${panel === "save" ? "border-blue-300 bg-blue-50 text-blue-700" : ""}`}
              aria-pressed={panel === "save"}
              disabled={list.genes.length === 0}
              onClick={() => setPanel(panel === "save" ? null : "save")}
              title="Save as a custom gene-set collection, usable in enrichment and the gene-set browsers"
            >
              <FolderPlus className="h-3.5 w-3.5" />
              Save as collection
            </button>
          </div>
          {ready && present.length > MAX_PLOT_GENES && (
            <p className="text-[10px] text-amber-700">
              The dot plot and heatmap use the first {MAX_PLOT_GENES} of the {present.length} genes in this dataset.
            </p>
          )}
          {ready && present.length > MAX_GRID_GENES && (
            <p className="text-[10px] text-slate-400">The feature grid shows the first {MAX_GRID_GENES}.</p>
          )}
          {panel === "enrich" && dsGenes && (
            <GeneListEnrichment
              datasetId={datasetId}
              list={list}
              genes={present}
              background={dsGenes.list}
              onScoreGeneSet={onScoreGeneSet}
              onGeneClick={onGeneClick}
            />
          )}
          {panel === "save" && <SaveAsCollection list={list} allLists={allLists} />}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1 text-[11px] text-slate-500">
        <Download className="h-3 w-3" />
        Export:
        {(["txt", "csv", "gmt"] as const).map((f) => (
          <button key={f} type="button" className="text-primary hover:underline" disabled={list.genes.length === 0} onClick={() => exportOne(f)}>
            .{f}
          </button>
        ))}
      </div>
    </div>
  );
}

/** ORA of the list's genes, with every gene the dataset measures as the background. */
function GeneListEnrichment({
  datasetId,
  list,
  genes,
  background,
  onScoreGeneSet,
  onGeneClick,
}: {
  datasetId: string;
  list: GeneList;
  genes: string[];
  background: string[];
  onScoreGeneSet: (scores: Float32Array, name: string, genes?: string[]) => void;
  onGeneClick: (gene: string) => void;
}) {
  const askCopilot = useViewStore((s) => s.askCopilot);
  const [collections, setCollections] = useState<Set<string>>(new Set(DEFAULT_MSIGDB_COLLECTIONS));
  const [excludeRiboMito, setExcludeRiboMito] = useState(true);
  const [showSets, setShowSets] = useState(false);
  const [data, setData] = useState<GeneListEnrichmentResponse | null>(null);
  const [ranOn, setRanOn] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await enrichGeneLists(datasetId, {
        up: genes,
        down: [],
        background,
        collections: Array.from(collections),
        exclude_ribo_mito: excludeRiboMito,
      });
      setData(res);
      setRanOn(genes);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [datasetId, genes, background, collections, excludeRiboMito]);

  const { scoreGenes, scoringName } = useGeneSetScoring(datasetId, onScoreGeneSet, onGeneClick);
  const { sorted, sortField, sortDir, onSort } = useEnrichmentSort(data ? data.up.results : []);
  const askAboutTerm = useCallback(
    (r: EnrichmentResult) =>
      askCopilot(
        `My gene list "${list.name}" is enriched for "${r.term}" (${collectionLabel(r)}, adjusted p ` +
          `${r.adjusted_pvalue.toExponential(2)}, ${r.overlap_count} of its ${r.gene_count} genes). ` +
          `Genes driving it: ${r.genes.slice(0, 12).join(", ")}. What does this suggest biologically?`,
      ),
    [askCopilot, list.name],
  );
  const stale = ranOn !== null && (ranOn.length !== genes.length || ranOn.some((g, i) => g !== genes[i]));
  const fmt = (n: number) => n.toLocaleString();

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-slate-200 p-2">
      <div className="flex items-center gap-2">
        <span className="font-semibold text-slate-700">Pathway enrichment</span>
        <InfoTip width={280}>
          Over-representation (hypergeometric test, Benjamini-Hochberg adjusted) of the list&apos;s genes
          that this dataset measures, against the chosen collections. The background is every gene the
          dataset measures.
        </InfoTip>
        <span className="flex-1" />
        <button
          type="button"
          onClick={run}
          disabled={loading || collections.size === 0}
          className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1 font-medium text-white hover:bg-primary/90 disabled:opacity-50"
        >
          {loading && <Loader2 className="h-3 w-3 animate-spin" />}
          Enrich
        </button>
      </div>
      <button
        type="button"
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
            Exclude ribosomal / mitochondrial genes from the list
          </label>
          <MSigDBCollectionTree selected={collections} onChange={setCollections} datasetId={datasetId} compact />
        </div>
      )}
      {error && <div className="rounded border border-red-200 bg-red-50 p-2 text-red-700">{error}</div>}
      {data && (
        <div className={`flex flex-col gap-1.5 ${stale ? "opacity-60" : ""}`}>
          <p className="text-[10px] text-slate-500">
            {fmt(data.up.n_genes_used)} genes tested
            {data.up.n_excluded_ribo_mito > 0 && ` (${fmt(data.up.n_excluded_ribo_mito)} ribosomal / mito left out)`}
            , against {fmt(data.background_n)} measured genes; {data.up.n_significant} term
            {data.up.n_significant === 1 ? "" : "s"} with adjusted p &lt; 0.05.
          </p>
          {data.warning && (
            <p role="note" className="rounded border border-amber-200 bg-amber-50 px-1.5 py-1 text-amber-800">
              {data.warning}
            </p>
          )}
          {stale && (
            <p role="status" className="text-amber-700">
              The list has changed since; enrich again to match.
            </p>
          )}
          {sorted.length === 0 ? (
            <p className="py-2 text-center text-slate-400">No gene set shares a gene with this list.</p>
          ) : (
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
          )}
        </div>
      )}
    </div>
  );
}

/** Save this list, or every list, as a custom gene-set collection (POST /genesets/custom/from-lists). */
function SaveAsCollection({ list, allLists }: { list: GeneList; allLists: GeneList[] }) {
  const bump = useCustomGeneSetsStore((s) => s.bump);
  const [scope, setScope] = useState<"one" | "all">("one");
  const [name, setName] = useState(list.name);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string; problems?: string[] } | null>(null);

  const save = async () => {
    const chosen = (scope === "all" ? allLists : [list]).filter((l) => l.genes.length > 0);
    setBusy(true);
    setMessage(null);
    try {
      const meta = await customCollectionFromLists(
        name.trim(),
        Object.fromEntries(chosen.map((l) => [l.name, l.genes])),
      );
      bump();
      setMessage({
        ok: true,
        text: `Saved as the collection "${meta.name}" (${meta.n_sets} set${meta.n_sets === 1 ? "" : "s"}); it is listed under Custom in the gene-set and enrichment tabs.`,
      });
    } catch (e) {
      setMessage({
        ok: false,
        text: e instanceof Error ? e.message : String(e),
        problems: e instanceof GeneSetUploadError ? e.problems : undefined,
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-slate-200 p-2">
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1">
          <input type="radio" checked={scope === "one"} onChange={() => setScope("one")} />
          This list
        </label>
        <label className="flex items-center gap-1">
          <input type="radio" checked={scope === "all"} onChange={() => setScope("all")} />
          All {allLists.length} lists
        </label>
      </div>
      <div className="flex items-center gap-1.5">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          aria-label="Collection name"
          placeholder="Collection name"
          maxLength={200}
          className="min-w-0 flex-1 rounded border border-slate-200 px-1.5 py-1 focus:border-primary focus:outline-none"
        />
        <button
          type="button"
          onClick={save}
          disabled={busy || !name.trim()}
          className="flex items-center gap-1 rounded-md bg-primary px-3 py-1 font-medium text-white hover:bg-primary/90 disabled:opacity-50"
        >
          {busy && <Loader2 className="h-3 w-3 animate-spin" />}
          Save
        </button>
      </div>
      <p className="text-[10px] text-slate-400">Collections store gene symbols upper-cased and match them ignoring case.</p>
      {message && (
        <div role={message.ok ? "status" : "alert"} className={message.ok ? "text-emerald-700" : "text-red-700"}>
          {message.text}
          {message.problems && message.problems.length > 0 && (
            <ul className="ml-4 list-disc">
              {message.problems.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
