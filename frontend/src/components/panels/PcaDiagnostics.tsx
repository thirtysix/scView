import { useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Panel } from "@/components/common/Panel";
import { InfoTip } from "@/components/common/InfoTip";
import { apiFetchBinary } from "@/api/client";
import { errorMessage } from "@/lib/errorMessage";
import { getPcaDiagnostics, type PcaDiagnostics as Result } from "@/api/pca";
import { decodeArrowBuffer } from "@/lib/arrow";
import { elbowLayout, elbowTraces, fmtPct, loadingLayout, loadingTraces, pcLabel } from "@/lib/pcaDiagnostics";
import { plotConfig } from "@/lib/plotConfig";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";

const FILL_STYLE = { width: "100%", height: "100%" } as const;

/** Colour the main scatter by one gene and open the Unified View (the expression
 *  overlay the Expression tab would set, fetched here as that tab is not mounted). */
export async function showGeneOnScatter(datasetId: string, gene: string): Promise<void> {
  const layer = useSettingsStore.getState().expressionLayer;
  const layerQuery = layer ? `&layer=${encodeURIComponent(layer)}` : "";
  const buffer = await apiFetchBinary(`/datasets/${datasetId}/expression?genes=${encodeURIComponent(gene)}${layerQuery}`);
  const decoded = decodeArrowBuffer(buffer);
  const values = decoded[gene] ?? decoded[Object.keys(decoded)[0]!];
  if (!values) throw new Error(`no expression came back for ${gene}`);
  const f32 = values instanceof Float32Array ? values : Float32Array.from(values as ArrayLike<number>, Number);
  const store = useUnifiedViewStore.getState();
  store.setOverlayValues(f32);
  store.setOverlayLabel(gene);
  // No `layer` on the overlay: that field marks a protein (CITE-seq) overlay,
  // not the expression layer the values were fetched from.
  store.setScatterOverlay({ type: "expression", gene });
  store.setOverlayDatasetId(datasetId);
  useViewStore.getState().setPanel("unified");
}

function Metric({ label, value, title }: { label: string; value: string; title: string }) {
  return (
    <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600" title={title}>
      {label} <span className="font-semibold text-slate-800">{value}</span>
    </span>
  );
}

function GeneChips({
  genes,
  sign,
  onPick,
  busy,
}: {
  genes: { gene: string; loading: number }[];
  sign: "+" | "−";
  onPick: (gene: string) => void;
  busy: string | null;
}) {
  if (genes.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="text-[11px] text-slate-400">{sign}</span>
      {genes.map((g) => (
        <button
          key={g.gene}
          type="button"
          onClick={() => onPick(g.gene)}
          disabled={busy !== null}
          title={`Loading ${g.loading.toFixed(3)}. Click to colour the scatter by ${g.gene}.`}
          className="rounded border border-slate-200 px-1.5 py-0.5 text-[11px] text-slate-600 hover:bg-slate-100 disabled:opacity-50"
        >
          {busy === g.gene ? <Loader2 className="inline h-3 w-3 animate-spin" /> : g.gene}
        </button>
      ))}
    </div>
  );
}

/**
 * PCA diagnostics on the Data Assessment tab, shown on request: each PC's share of
 * the variance with the cumulative curve, the elbow and the number of PCs the
 * neighbour graph used, and the top +/- loading genes of any PC (click a gene to
 * colour the scatter by it).
 */
export function PcaDiagnostics({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pc, setPc] = useState(1);
  const [busyGene, setBusyGene] = useState<string | null>(null);

  // A new dataset or a pipeline run makes the shown diagnostic stale.
  useEffect(() => {
    setData(null);
    setError(null);
    setPc(1);
  }, [datasetId, refreshKey]);

  useEffect(() => {
    if (!open || !datasetId || data) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPcaDiagnostics(datasetId)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, datasetId, data, refreshKey]);

  const ok = data && data.available ? data : null;
  const entry = ok?.loadings.find((l) => l.pc === pc) ?? null;
  const elbow = useMemo(() => (ok ? { data: elbowTraces(ok), layout: elbowLayout(ok) } : null), [ok]);
  const bars = useMemo(() => (entry ? { data: loadingTraces(entry), layout: loadingLayout(entry) } : null), [entry]);

  const pick = async (gene: string) => {
    setBusyGene(gene);
    setError(null);
    try {
      await showGeneOnScatter(datasetId, gene);
    } catch (e) {
      setError(`Could not colour the scatter: ${errorMessage(e)}`);
    } finally {
      setBusyGene(null);
    }
  };

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold text-slate-700">PCA</h3>
          <InfoTip label="About the PCA diagnostics">
            How much of the variance each principal component keeps, where the curve bends (the
            elbow: Kneedle, the PC farthest below the line from the first to the last PC), how many
            leading PCs reach 90% of what the stored PCs keep together, and which genes each PC
            weighs most. A PC led by ribosomal, mitochondrial or cell-cycle genes is a technical
            or cell-state axis rather than a cell-type one.
          </InfoTip>
        </div>
        <span className="flex items-center gap-2">
          {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="rounded border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
          >
            {open ? "Hide" : "Show"}
          </button>
        </span>
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {error}
        </p>
      )}
      {open && data && !data.available && <p className="text-xs text-slate-500">{data.reason}</p>}
      {open && ok && elbow && (
        <>
          <div className="flex flex-wrap items-center gap-1.5">
            <Metric
              label="Elbow"
              value={`PC${ok.elbow}`}
              title="Kneedle: on the scree curve (variance ratio vs PC, both axes scaled to 0..1) the PC farthest below the straight line from the first to the last PC."
            />
            <Metric
              label="90% at"
              value={`${ok.pcs_90} PCs`}
              title={`The smallest number of leading PCs whose variance reaches 90% of what all ${ok.n_pcs} stored PCs retain together (${fmtPct(ok.retained)} of ${ok.ratio_basis === "total" ? "the total variance" : "their own sum"}).`}
            />
            <Metric
              label="Retained"
              value={fmtPct(ok.retained)}
              title={
                ok.ratio_basis === "total"
                  ? `Sum of the ${ok.n_pcs} PCs' variance ratios: their share of the total variance of the (scaled, variable-gene) data PCA was run on.`
                  : "No variance ratios were stored with this PCA; shares are of the stored PCs' own sum, so the curve ends at 100%."
              }
            />
            {ok.neighbors && (
              <Metric
                label="Neighbours"
                value={ok.neighbors.n_pcs !== null ? `${ok.neighbors.n_pcs} PCs of ${ok.neighbors.use_rep}` : ok.neighbors.use_rep}
                title="What the neighbour graph (and so UMAP and clustering) was built from: the recorded n_pcs, otherwise every column of use_rep."
              />
            )}
          </div>
          <div className="grid grid-cols-1 gap-2 xl:grid-cols-2">
            <Panel title="Variance per PC" className="h-[300px]">
              <Plot data={elbow.data} layout={elbow.layout} config={plotConfig("pca_variance")} style={FILL_STYLE} useResizeHandler />
            </Panel>
            <Panel
              title="Loadings"
              className="h-[300px]"
              actions={
                <label className="flex items-center gap-1 text-[11px] text-slate-500">
                  PC
                  <select
                    aria-label="Principal component"
                    value={pc}
                    onChange={(e) => setPc(Number(e.target.value))}
                    className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700"
                  >
                    {ok.loadings.map((l) => (
                      <option key={l.pc} value={l.pc}>
                        {pcLabel(ok, l.pc)}
                      </option>
                    ))}
                  </select>
                </label>
              }
            >
              {bars ? (
                <Plot data={bars.data} layout={bars.layout} config={plotConfig(`pca_loadings_pc${pc}`)} style={FILL_STYLE} useResizeHandler />
              ) : (
                <p className="p-2 text-xs text-slate-500">No loadings stored with this PCA.</p>
              )}
            </Panel>
          </div>
          {entry && (
            <div className="space-y-1" title="Genes with the largest loadings on this PC; click one to colour the scatter by it.">
              <GeneChips genes={entry.positive} sign="+" onPick={pick} busy={busyGene} />
              <GeneChips genes={entry.negative} sign="−" onPick={pick} busy={busyGene} />
            </div>
          )}
          <p className="text-[11px] text-slate-400">
            {ok.n_pcs} PCs from {ok.n_cells.toLocaleString()} cells
            {ok.n_genes_used !== null ? ` and ${ok.n_genes_used.toLocaleString()} of ${ok.n_genes_total.toLocaleString()} genes` : ""}
            {ok.params.use_highly_variable ? " (variable genes)" : ""}
            {ok.params.zero_center === false ? ", not zero-centred" : ""}. Loadings: top {entry ? Math.max(entry.positive.length, entry.negative.length) : 0} genes per sign.
          </p>
        </>
      )}
    </div>
  );
}
