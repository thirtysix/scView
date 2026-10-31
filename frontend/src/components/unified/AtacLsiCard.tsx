import { atacExportUrl, type AtacState } from "@/api/atac";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import Plot from "@/components/plots/Plot";
import { embeddingReady } from "@/hooks/useEmbeddingReady";
import { ATAC_CLUSTERS, ATAC_UMAP, depthChoiceText, planText } from "@/lib/atac";
import { plotConfig } from "@/lib/plotConfig";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { AtacRunStatus } from "./AtacRunStatus";

const SHOWN_COMPONENTS = 20;
const BUTTON = "rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary disabled:opacity-50";

/** LSI (TF-IDF + SVD): run it, see each component's depth correlation and the choice made from it. */
export function AtacLsiCard({ datasetId, state }: { datasetId: string; state: AtacState }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const setColorBy = useSettingsStore((s) => s.setColorBy);
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);
  const run = useAtacRunStore();
  const lsi = state.lsi ?? null;
  const live = run.datasetId === datasetId && run.running;
  const elsewhere = !live && !!state.running;
  const hasClusters = !!dataset?.obs_columns.some((c) => c.name === ATAC_CLUSTERS);
  const hasUmap = embeddingReady(dataset, datasetId, ATAC_UMAP);
  const rs = (lsi?.depth_r ?? []).slice(0, SHOWN_COMPONENTS);
  const names = rs.map((_, i) => `${i + 1}`);

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="LSI and clusters">
      <h4 className="flex items-center gap-1 text-xs font-semibold text-slate-600">
        <span>LSI and clusters</span>
        <InfoTip label="About LSI" width={300}>
          Signac&rsquo;s recipe: peaks above the 5th percentile of counts, TF-IDF (method 1), truncated SVD, 50 components.
          The first component usually follows sequencing depth and is then left out. A neighbour graph of its own (the RNA
          graph is never touched), a UMAP and Leiden clusters follow, then gene activity.
        </InfoTip>
      </h4>
      {state.plan_reason && <p className="text-[11px] leading-snug text-slate-500">{state.plan_reason}</p>}
      {state.plan && <p className="text-[11px] leading-snug text-slate-500" data-testid="atac-plan">{planText(state.plan)}</p>}
      <AtacRunStatus datasetId={datasetId} kind="run" />
      {!live && state.plan && (
        <button type="button" disabled={run.running || elsewhere} onClick={() => void run.run(datasetId, {})}
                className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50">
          {lsi ? "Run ATAC analysis again" : "Run ATAC analysis"}
        </button>
      )}
      {lsi && (
        <div className="space-y-2">
          {lsi.stale && (
            <p className="text-[11px] leading-snug text-amber-700" data-testid="atac-stale">
              The cells changed since this ran (a filter or a merge): run it again.
            </p>
          )}
          <p className="text-[11px] leading-snug text-slate-600" data-testid="atac-depth-choice">{depthChoiceText(lsi)}</p>
          <Plot
            data={[{
              type: "bar" as const, x: names, y: rs.map((r) => Math.abs(r)),
              marker: { color: rs.map((_, i) => (lsi.dims_used.includes(i + 1) ? "#2563eb" : "#cbd5e1")) },
              hovertemplate: "Component %{x}<br>|r| with depth %{y:.2f}<extra></extra>",
            }]}
            layout={{
              height: 150, margin: { t: 6, r: 8, b: 30, l: 36 },
              // Category: component numbers as labels, every one shown.
              xaxis: { type: "category" as const, tickmode: "array" as const, tickvals: names, ticktext: names,
                       tickfont: { size: 9, color: "#64748b" },
                       title: { text: "LSI component", font: { size: 10, color: "#64748b" } } },
              yaxis: { range: [0, 1], tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9",
                       title: { text: "|r| with depth", font: { size: 10, color: "#64748b" } } },
              shapes: [{ type: "line" as const, xref: "paper" as const, x0: 0, x1: 1, y0: lsi.depth_limit, y1: lsi.depth_limit,
                         line: { color: "#d97706", width: 1, dash: "dot" as const } }],
              showlegend: false, paper_bgcolor: "white", plot_bgcolor: "white", font: { family: "Inter, system-ui, sans-serif" },
            }}
            config={plotConfig("atac_lsi_depth_correlation")}
            useResizeHandler
            style={{ width: "100%" }}
          />
          <Caption n={{ value: lsi.n_selected, unit: "peaks" }}>
            Each component&rsquo;s correlation with the cell&rsquo;s total counts (the first {rs.length} of {lsi.depth_r.length}); grey
            components are not in the graph. {lsi.n_clusters} ATAC clusters.
          </Caption>
          <div className="flex flex-wrap gap-1.5">
            <button type="button" className={BUTTON} disabled={!hasUmap} onClick={() => setEmbedding(ATAC_UMAP)}>Show ATAC UMAP</button>
            <button type="button" className={BUTTON} disabled={!hasClusters} onClick={() => setColorBy(ATAC_CLUSTERS)}>Colour by ATAC clusters</button>
          </div>
          <p className="text-[10px] leading-snug text-slate-400">{lsi.method}</p>
        </div>
      )}
      <a href={atacExportUrl(datasetId)} className="inline-block text-[11px] text-primary hover:underline" download>
        Download the peak matrix (.h5ad, for Signac and muon; takes about a minute the first time)
      </a>
    </section>
  );
}
