import { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Download, Grid2x2, Link2, Loader2 } from "lucide-react";
import { fetchCoexpression, type CoexpressionMethod, type CoexpressionResponse } from "@/api/coexpression";
import { AddToGeneListMenu } from "@/components/common/AddToGeneListMenu";
import { InfoTip } from "@/components/common/InfoTip";
import { sortCoexpressed, type CoexpressedSort } from "@/lib/coexpressedGenes";
import { downloadCsv } from "@/lib/csv";
import { MAX_GRID_GENES } from "@/lib/geneCoexpression";
import { useGeneGridStore } from "@/stores/geneGridStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

/** The backend's floor on cells (core/coexpression.py MIN_CELLS). */
const MIN_CELLS = 10;
const TOP_N = 25;

type Scope = "all" | "selection" | "group";
type Side = "positive" | "negative";

interface Props {
  datasetId: string;
  /** The gene on the scatter: the query. */
  activeGene: string | null;
  /** Overlay a gene of the list on the scatter. */
  onGeneSelect: (gene: string) => void;
}

/**
 * "Co-expressed genes" in the Expression subtab: the genes whose expression
 * rises (or falls) with the overlaid gene across cells, within every cell, the
 * current selection, or the highlighted group. Click a gene to overlay it;
 * send the query and its top partners to "Several genes".
 */
export function CoexpressedGenes({ datasetId, activeGene, onGeneSelect }: Props) {
  const selected = useSelectionStore((s) => s.selectedCellIndices);
  const highlighted = useSelectionStore((s) => s.highlightedGroup);
  const setGridGenes = useGeneGridStore((s) => s.setGenes);
  const geneGrid = useSplitViewStore((s) => s.geneGrid);
  const setGeneGrid = useSplitViewStore((s) => s.setGeneGrid);

  const [method, setMethod] = useState<CoexpressionMethod>("pearson");
  const [scope, setScope] = useState<Scope>("all");
  const [side, setSide] = useState<Side>("positive");
  const [sort, setSort] = useState<{ field: CoexpressedSort; desc: boolean }>({ field: "r", desc: true });
  const [data, setData] = useState<CoexpressionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const nSelected = selected?.size ?? 0;
  const canSelection = nSelected >= MIN_CELLS;
  // A scope that no longer exists falls back to every cell.
  const effectiveScope: Scope =
    scope === "selection" && !canSelection ? "all" : scope === "group" && !highlighted ? "all" : scope;

  useEffect(() => {
    setData(null);
    setError(null);
  }, [datasetId]);

  const run = async () => {
    if (!activeGene) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetchCoexpression(datasetId, {
        gene: activeGene,
        method,
        top_n: TOP_N,
        ...(effectiveScope === "selection" && selected ? { indices: Array.from(selected) } : {}),
        ...(effectiveScope === "group" && highlighted
          ? { column: highlighted.column, value: highlighted.value }
          : {}),
      });
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const shown = useMemo(
    () => (data ? sortCoexpressed(data[side], sort.field, sort.desc) : []),
    [data, side, sort],
  );

  const onSort = (field: CoexpressedSort) =>
    setSort((s) => (s.field === field ? { field, desc: !s.desc } : { field, desc: field !== "gene" }));

  // The query and its strongest partners on this side, as the "Several genes" picks.
  const sendToGrid = () => {
    if (!data) return;
    const genes = [data.gene, ...data[side].map((g) => g.gene)].slice(0, MAX_GRID_GENES);
    setGridGenes(datasetId, genes);
    if (geneGrid?.datasetId === datasetId && geneGrid.kind === "features") {
      setGeneGrid({ datasetId, kind: "features", genes });
    }
  };

  const exportCsv = () => {
    if (!data) return;
    downloadCsv(
      `coexpressed_${data.gene}_${data.method}_${datasetId}`,
      ["direction", "gene", "r", "n_expressing", "pct_expressing"],
      (["positive", "negative"] as const).flatMap((s) =>
        data[s].map((g) => [s, g.gene, g.r, g.n_expressing, g.pct_expressing]),
      ),
    );
  };

  const fmt = (n: number) => n.toLocaleString();
  const sortIcon = (field: CoexpressedSort) =>
    sort.field === field ? (sort.desc ? <ArrowDown className="h-2.5 w-2.5" /> : <ArrowUp className="h-2.5 w-2.5" />) : null;
  const btn = (on: boolean) =>
    `flex-1 truncate rounded px-1.5 py-0.5 transition ${
      on ? "bg-slate-200 font-medium text-slate-800" : "text-slate-500 hover:text-slate-700"
    }`;

  return (
    <div className="space-y-2 rounded-lg border border-slate-200 p-2 text-xs">
      <div className="flex items-center gap-1.5">
        <Link2 className="h-3.5 w-3.5 text-slate-400" />
        <h4 className="font-semibold text-slate-600">Co-expressed genes</h4>
        <InfoTip width={300}>
          Correlation across cells with the gene on the scatter, on the log-normalised expression
          the overlay shows. Pearson follows the values; Spearman their ranks, so a few very high
          cells weigh less. A gene needs at least 10 expressing cells, or 1% of the chosen cells if
          that is more, and some variation, or it is skipped. Correlation across all cells is
          mostly driven by cell type; within one cell type it points to programmes inside it.
        </InfoTip>
      </div>

      <div className="flex items-center gap-1.5">
        <select
          aria-label="Cells to correlate over"
          value={effectiveScope}
          onChange={(e) => setScope(e.target.value as Scope)}
          className="min-w-0 flex-1 rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700"
        >
          <option value="all">All cells</option>
          <option value="selection" disabled={!canSelection}>
            {canSelection ? `Selection (${fmt(nSelected)} cells)` : `Selection (needs ${MIN_CELLS}+ cells)`}
          </option>
          <option value="group" disabled={!highlighted}>
            {highlighted ? `${highlighted.value} (${highlighted.column})` : "Highlighted group (none)"}
          </option>
        </select>
        <div className="flex rounded border border-slate-300 bg-white p-0.5">
          {(["pearson", "spearman"] as const).map((m) => (
            <button key={m} onClick={() => setMethod(m)} aria-pressed={method === m} className={btn(method === m)}>
              {m === "pearson" ? "Pearson" : "Spearman"}
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={run}
        disabled={!activeGene || loading}
        title={activeGene ? `Genes correlated with ${activeGene}` : "Overlay a gene first"}
        className="flex w-full items-center justify-center gap-1.5 rounded-md bg-primary px-3 py-1.5 font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
      >
        {loading && <Loader2 className="h-3 w-3 animate-spin" />}
        {activeGene ? `Find genes co-expressed with ${activeGene}` : "Overlay a gene to find its partners"}
      </button>

      {error && <div className="rounded border border-red-200 bg-red-50 p-2 text-red-700">{error}</div>}

      {data && (
        <>
          <p className="text-[10px] leading-snug text-slate-500">
            {data.method === "pearson" ? "Pearson" : "Spearman"} r with{" "}
            <span className="font-semibold text-slate-700">{data.gene}</span> over {fmt(data.n_cells)} cells (
            {data.scope}; {fmt(data.query_n_expressing)} express {data.gene}). {fmt(data.n_genes_tested)} genes
            correlated
            {data.min_expressing != null
              ? `; ${fmt(data.n_skipped_rare)} skipped, expressed in fewer than ${fmt(data.min_expressing)} of these cells`
              : "; values are scaled, so no expressing-cell minimum"}
            {data.n_skipped_constant > 0 && `; ${fmt(data.n_skipped_constant)} constant here`}.
            {data.values === "log-normalised from counts" && " Counts were log-normalised first."}
          </p>

          <div className="flex items-center gap-1.5">
            <div className="flex min-w-0 flex-1 rounded border border-slate-300 bg-white p-0.5">
              {(["positive", "negative"] as const).map((s) => (
                <button key={s} onClick={() => setSide(s)} aria-pressed={side === s} className={btn(side === s)}>
                  {s === "positive" ? "Positive" : "Negative"} ({data[s].length})
                </button>
              ))}
            </div>
            <button
              onClick={sendToGrid}
              disabled={data[side].length === 0}
              title={`Put ${data.gene} and its top ${MAX_GRID_GENES - 1} ${side} partners in "Several genes"`}
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              <Grid2x2 className="h-3 w-3" />
              To grid
            </button>
            <button
              onClick={exportCsv}
              title="Export both lists as CSV"
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50"
            >
              <Download className="h-3 w-3" />
              CSV
            </button>
            <AddToGeneListMenu
              genes={() => [data.gene, ...shown.map((g) => g.gene)]}
              defaultName={`Co-expressed with ${data.gene}`}
              source={`${side === "positive" ? "Positively" : "Negatively"} co-expressed with ${data.gene} (${data.scope})`}
              disabled={shown.length === 0}
              title={`Add ${data.gene} and its ${shown.length} ${side} partners (table order) to a gene list`}
            />
          </div>

          {shown.length === 0 ? (
            <p className="py-2 text-center text-slate-400">
              No gene is {side === "positive" ? "positively" : "negatively"} correlated with {data.gene} here.
            </p>
          ) : (
            <div className="max-h-64 overflow-auto rounded border border-slate-100">
              <table className="w-full text-[11px]" aria-label={`${side} co-expressed genes`}>
                <thead className="sticky top-0 bg-slate-50 text-slate-500">
                  <tr>
                    {(
                      [
                        ["gene", "Gene", "text-left"],
                        ["r", "r", "text-right"],
                        ["n", "Expressing", "text-right"],
                      ] as const
                    ).map(([field, label, align]) => (
                      <th key={field} className={`px-1.5 py-1 font-medium ${align}`}>
                        <button
                          onClick={() => onSort(field)}
                          className={`inline-flex items-center gap-0.5 hover:text-slate-700 ${align === "text-right" ? "flex-row-reverse" : ""}`}
                        >
                          {label}
                          {sortIcon(field)}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {shown.map((g) => (
                    <tr
                      key={g.gene}
                      onClick={() => onGeneSelect(g.gene)}
                      className={`cursor-pointer border-t border-slate-50 hover:bg-blue-50 ${
                        g.gene === activeGene ? "bg-blue-50/60" : ""
                      }`}
                    >
                      <td className="px-1.5 py-1 font-mono text-slate-800">{g.gene}</td>
                      <td className={`px-1.5 py-1 text-right tabular-nums ${g.r >= 0 ? "text-red-600" : "text-blue-600"}`}>
                        {g.r.toFixed(3)}
                      </td>
                      <td className="px-1.5 py-1 text-right tabular-nums text-slate-500">
                        {fmt(g.n_expressing)} <span className="text-slate-400">({g.pct_expressing.toFixed(1)}%)</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
