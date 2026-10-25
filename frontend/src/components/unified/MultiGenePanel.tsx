import { useMemo } from "react";
import { BarChart3, Blend, Grid2x2, Loader2, ScatterChart, X } from "lucide-react";
import { GeneSearch } from "@/components/common/GeneSearch";
import { Panel } from "@/components/common/Panel";
import { StackedViolin } from "@/components/plots/StackedViolin";
import { SelectionChip } from "@/components/common/SelectionChip";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { GeneVsGene } from "@/components/unified/GeneVsGene";
import { JointDensityButton } from "@/components/unified/DensityToggle";
import { useGeneExpression } from "@/hooks/useGeneExpression";
import { useObsCodes } from "@/hooks/useObsCodes";
import { prettyObsLabel } from "@/lib/formatting";
import { mergeGeneLists, parseGeneList } from "@/lib/geneList";
import { MAX_GRID_GENES, MIN_GRID_GENES, strideSample } from "@/lib/geneCoexpression";
import { useGeneGridStore } from "@/stores/geneGridStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

/** Cells the stacked violin draws at most (a stride sample above it). */
const VIOLIN_MAX_CELLS = 30000;

interface MultiGenePanelProps {
  datasetId: string;
  /** The violins' groups (the Expression subtab's group-by) */
  groupByColumn: string;
}

/**
 * The Expression subtab's "Several genes" section: pick 2–9 genes, then see
 * the embedding coloured by each side by side (GeneFeatureGrid over the plot)
 * or a stacked violin of them across the group-by; and for two of them, their
 * blend on the embedding or one against the other, cell by cell.
 */
export function MultiGenePanel({ datasetId, groupByColumn }: MultiGenePanelProps) {
  const storeDataset = useGeneGridStore((s) => s.datasetId);
  const storedGenes = useGeneGridStore((s) => s.genes);
  const setGenes = useGeneGridStore((s) => s.setGenes);
  const showViolins = useGeneGridStore((s) => s.showViolins);
  const setShowViolins = useGeneGridStore((s) => s.setShowViolins);
  const storedPair = useGeneGridStore((s) => s.pair);
  const setPair = useGeneGridStore((s) => s.setPair);
  const showScatter = useGeneGridStore((s) => s.showScatter);
  const setShowScatter = useGeneGridStore((s) => s.setShowScatter);
  const geneGrid = useSplitViewStore((s) => s.geneGrid);
  const setGeneGrid = useSplitViewStore((s) => s.setGeneGrid);
  // Picks made on another dataset do not carry over.
  const genes = useMemo(() => (storeDataset === datasetId ? storedGenes : []), [storeDataset, datasetId, storedGenes]);
  const enough = genes.length >= MIN_GRID_GENES;
  const full = genes.length >= MAX_GRID_GENES;

  const pair = enough && storedPair && storedPair.every((g) => genes.includes(g)) ? storedPair : null;
  const blendOn = geneGrid?.kind === "blend" && geneGrid.datasetId === datasetId;

  const update = (next: string[]) => {
    setGenes(datasetId, next);
    if (geneGrid?.datasetId !== datasetId) return;
    // An open feature grid follows the picks (and closes below two genes)...
    if (geneGrid.kind === "features") {
      setGeneGrid(next.length >= MIN_GRID_GENES ? { datasetId, kind: "features", genes: next.slice(0, MAX_GRID_GENES) } : null);
    }
    // ...and a blend closes when one of its genes goes.
    if (geneGrid.kind === "blend" && !geneGrid.genes.every((g) => next.includes(g))) setGeneGrid(null);
  };
  const choosePair = (which: 0 | 1, gene: string) => {
    if (!pair) return;
    const next: [string, string] = which === 0 ? [gene, pair[1]] : [pair[0], gene];
    setPair(next);
    if (blendOn) setGeneGrid({ datasetId, kind: "blend", genes: next });
  };
  const add = (text: string) => update(mergeGeneLists(genes, parseGeneList(text), MAX_GRID_GENES));
  const remove = (gene: string) => update(genes.filter((g) => g !== gene));

  const featuresOn = geneGrid?.kind === "features" && geneGrid.datasetId === datasetId;

  return (
    <div className="space-y-2 rounded-lg border border-slate-200 p-2">
      <div className="flex items-baseline justify-between gap-2">
        <h4 className="text-xs font-semibold text-slate-600">Several genes</h4>
        <span className="text-[10px] text-slate-400">
          {genes.length}/{MAX_GRID_GENES}
        </span>
      </div>
      <GeneSearch
        datasetId={datasetId}
        onSelect={add}
        multiple
        placeholder={full ? `At most ${MAX_GRID_GENES} genes` : "Add genes (or paste a list)…"}
        className="w-full"
      />
      {genes.length > 0 && (
        <div className="flex flex-wrap items-center gap-1" aria-label="Picked genes">
          {genes.map((g) => (
            <span
              key={g}
              className="flex items-center gap-0.5 rounded-full border border-slate-300 bg-white py-0.5 pl-2 pr-1 text-xs text-slate-700"
            >
              {g}
              <button
                type="button"
                onClick={() => remove(g)}
                aria-label={`Remove ${g}`}
                className="rounded-full p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
          <button
            type="button"
            onClick={() => update([])}
            className="ml-1 text-[11px] text-slate-400 hover:text-slate-600"
          >
            Clear
          </button>
        </div>
      )}
      <div className="flex flex-wrap gap-1.5">
        <button
          type="button"
          disabled={!enough}
          aria-pressed={featuresOn}
          onClick={() => setGeneGrid(featuresOn ? null : { datasetId, kind: "features", genes })}
          title={enough ? "Colour the embedding by each gene, side by side on one camera" : `Pick ${MIN_GRID_GENES} or more genes`}
          className={`flex items-center gap-1 rounded border px-2 py-1 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-50 ${
            featuresOn ? "border-blue-300 bg-blue-50 text-blue-700" : "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Grid2x2 className="h-3.5 w-3.5" />
          {featuresOn ? "Hide feature grid" : "Feature grid"}
        </button>
        <button
          type="button"
          disabled={!enough || !groupByColumn}
          aria-pressed={showViolins}
          onClick={() => setShowViolins(!showViolins)}
          title={enough ? "A violin per gene and group, genes stacked" : `Pick ${MIN_GRID_GENES} or more genes`}
          className={`flex items-center gap-1 rounded border px-2 py-1 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-50 ${
            showViolins ? "border-blue-300 bg-blue-50 text-blue-700" : "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
          }`}
        >
          <BarChart3 className="h-3.5 w-3.5" />
          Stacked violin
        </button>
        <JointDensityButton datasetId={datasetId} genes={genes} onShow={() => geneGrid && setGeneGrid(null)} />
      </div>
      {showViolins && enough && groupByColumn && (
        <StackedViolinSection datasetId={datasetId} genes={genes} groupByColumn={groupByColumn} />
      )}

      {pair && (
        <div className="space-y-1.5 border-t border-slate-100 pt-2">
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-[11px] font-medium text-slate-500">Two genes</span>
            <PairSelect label="Gene A" value={pair[0]} other={pair[1]} genes={genes} onChange={(g) => choosePair(0, g)} />
            <span className="text-slate-400">vs</span>
            <PairSelect label="Gene B" value={pair[1]} other={pair[0]} genes={genes} onChange={(g) => choosePair(1, g)} />
          </div>
          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              aria-pressed={blendOn}
              onClick={() => setGeneGrid(blendOn ? null : { datasetId, kind: "blend", genes: [...pair] })}
              title="The embedding coloured by A, by B, and by both blended (A red, B green, both yellow)"
              className={`flex items-center gap-1 rounded border px-2 py-1 text-xs font-medium ${
                blendOn ? "border-blue-300 bg-blue-50 text-blue-700" : "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
              }`}
            >
              <Blend className="h-3.5 w-3.5" />
              {blendOn ? "Hide blend" : "Blend"}
            </button>
            <button
              type="button"
              aria-pressed={showScatter}
              disabled={!groupByColumn}
              onClick={() => setShowScatter(!showScatter)}
              title="Each cell's A against its B, with correlations and co-expression counts"
              className={`flex items-center gap-1 rounded border px-2 py-1 text-xs font-medium disabled:opacity-50 ${
                showScatter ? "border-blue-300 bg-blue-50 text-blue-700" : "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
              }`}
            >
              <ScatterChart className="h-3.5 w-3.5" />
              Gene vs gene
            </button>
          </div>
          {showScatter && groupByColumn && (
            <Panel title={`${pair[0]} vs ${pair[1]}`} bodyClassName="p-2">
              <GeneVsGene datasetId={datasetId} geneA={pair[0]} geneB={pair[1]} groupColumn={groupByColumn} />
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}

function PairSelect({
  label,
  value,
  other,
  genes,
  onChange,
}: {
  label: string;
  value: string;
  other: string;
  genes: string[];
  onChange: (gene: string) => void;
}) {
  return (
    <select
      aria-label={label}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700"
    >
      {genes
        .filter((g) => g !== other)
        .map((g) => (
          <option key={g} value={g}>
            {g}
          </option>
        ))}
    </select>
  );
}

/** The stacked violin with its data: the genes in one request, the groups from the embedding buffer. */
function StackedViolinSection({ datasetId, genes, groupByColumn }: { datasetId: string; genes: string[]; groupByColumn: string }) {
  const layer = useSettingsStore((s) => s.expressionLayer);
  const embedding = useSettingsStore((s) => s.embedding);
  const { values, missing, isLoading, error } = useGeneExpression(datasetId, genes, layer);
  const { codes, error: codesError } = useObsCodes(datasetId, embedding, groupByColumn);
  const found = useMemo(
    () => (values && codes ? genes.filter((g) => values[g]?.length === codes.length) : []),
    [values, codes, genes],
  );
  const cells = useMemo(() => (codes ? strideSample(codes.length, VIOLIN_MAX_CELLS) : null), [codes]);
  const hasSelection = useLinkedSelection().size > 0;

  let body;
  if (error || codesError) {
    body = <p className="text-xs text-red-600">Could not load the violins.</p>;
  } else if (isLoading || !values || !codes) {
    body = (
      <div className="flex items-center gap-2 py-6 text-xs text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading {genes.length} genes…
      </div>
    );
  } else if (found.length === 0) {
    body = <p className="text-xs text-amber-700">None of these genes is in this expression layer.</p>;
  } else {
    body = (
      <>
        <StackedViolin
          genes={found}
          values={values}
          codes={codes}
          groupColumn={groupByColumn}
          cells={cells}
          exportName={`stacked_violin_${groupByColumn}`}
        />
        <p className="px-1 text-[10px] text-slate-400">
          Each gene on its own scale.
          {cells
            ? ` ${cells.length.toLocaleString()} of ${codes.length.toLocaleString()} cells, sampled evenly.`
            : ` All ${codes.length.toLocaleString()} cells.`}
          {missing.length > 0 ? ` Not in this layer: ${missing.join(", ")}.` : ""}
          {hasSelection ? " Last column (orange): the selected cells, from any group." : ""}
        </p>
        <SelectionChip className="ml-1" />
      </>
    );
  }
  return (
    <Panel title={`Stacked violin by ${prettyObsLabel(groupByColumn)}`} bodyClassName="p-1">
      {body}
    </Panel>
  );
}
