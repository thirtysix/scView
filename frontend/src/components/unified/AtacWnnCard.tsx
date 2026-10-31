import type { AtacState } from "@/api/atac";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { embeddingReady } from "@/hooks/useEmbeddingReady";
import { WNN_ATAC_CLUSTERS, WNN_ATAC_UMAP, WNN_ATAC_WEIGHT, wnnPlanText } from "@/lib/atac";
import { prettyObsLabel } from "@/lib/formatting";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { AtacRunStatus } from "./AtacRunStatus";

const BUTTON = "rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary disabled:opacity-50";

/** WNN RNA + ATAC: Seurat's FindMultiModalNeighbors with LSI as the second modality. */
export function AtacWnnCard({ datasetId, state, groupby }: { datasetId: string; state: AtacState; groupby: string }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const setColorBy = useSettingsStore((s) => s.setColorBy);
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);
  const run = useAtacRunStore();
  const live = run.datasetId === datasetId && run.running;
  const elsewhere = !live && !!state.running;
  const wnn = state.wnn ?? null;
  const plan = state.wnn_plan ?? null;
  const rows = wnn?.by_group?.rows ?? [];
  const listed = (name: string) => !!dataset?.obs_columns.some((c) => c.name === name);

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Joint RNA + ATAC (WNN)">
      <h4 className="flex items-center gap-1 text-xs font-semibold text-slate-600">
        <span>Joint RNA + ATAC (WNN)</span>
        <InfoTip label="About WNN on ATAC" width={300}>
          Weighted nearest neighbours (Hao et al. 2021): for each cell, how well its RNA neighbours and its ATAC neighbours
          predict it decides how much each modality counts. As Seurat&rsquo;s FindMultiModalNeighbors, with RNA PCs and LSI 2:50.
        </InfoTip>
      </h4>
      {!plan && state.wnn_reason && <p className="text-[11px] leading-snug text-slate-500">{state.wnn_reason}</p>}
      {plan && <p className="text-[11px] leading-snug text-slate-500" data-testid="atac-wnn-plan">{wnnPlanText(plan)}</p>}
      <AtacRunStatus datasetId={datasetId} kind="wnn" />
      {!live && plan && (
        <button type="button" disabled={run.running || elsewhere} onClick={() => void run.runWnn(datasetId, {})}
                className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50">
          {wnn ? "Run WNN again" : "Run WNN"}
        </button>
      )}
      {wnn && (
        <div className="space-y-2">
          {wnn.stale && (
            <p className="text-[11px] leading-snug text-amber-700">The cells changed since this ran (a filter or a merge): run it again.</p>
          )}
          <p className="text-[11px] leading-snug text-slate-500">
            Saved run: mean RNA weight {wnn.mean_rna_weight.toFixed(2)}, {wnn.n_clusters} WNN clusters.
          </p>
          <div className="flex flex-wrap gap-1.5">
            <button type="button" className={BUTTON} disabled={!embeddingReady(dataset, datasetId, WNN_ATAC_UMAP)}
                    onClick={() => setEmbedding(WNN_ATAC_UMAP)}>Show WNN UMAP</button>
            <button type="button" className={BUTTON} disabled={!listed(WNN_ATAC_CLUSTERS)} onClick={() => setColorBy(WNN_ATAC_CLUSTERS)}>Colour by WNN clusters</button>
            <button type="button" className={BUTTON} disabled={!listed(WNN_ATAC_WEIGHT)} onClick={() => setColorBy(WNN_ATAC_WEIGHT)}>Colour by RNA weight</button>
          </div>
          {rows.length > 0 && (
            <>
              <table className="w-full text-[11px] text-slate-600" data-testid="atac-wnn-by-group">
                <thead>
                  <tr className="text-left text-slate-500">
                    <th className="font-medium">{prettyObsLabel(groupby)}</th>
                    <th className="text-right font-medium" title="Mean RNA weight of the group's cells: 0 to 1, with its ATAC weight summing to 1">Mean RNA weight</th>
                    <th className="text-right font-medium">Cells</th>
                  </tr>
                </thead>
                <tbody>
                  {[...rows].sort((a, b) => b.mean_rna_weight - a.mean_rna_weight).map((r) => (
                    <tr key={r.group}>
                      <td>{r.group}</td>
                      <td className="text-right tabular-nums">{r.mean_rna_weight.toFixed(2)}</td>
                      <td className="text-right tabular-nums">{r.n_cells.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <Caption n={{ value: rows.reduce((a, r) => a + r.n_cells, 0), unit: "cells" }}>
                RNA weight by {prettyObsLabel(groupby)}. The weights are relative: a high RNA weight says the RNA predicted a
                cell&rsquo;s neighbours better than its ATAC did, not that its ATAC is poor.
              </Caption>
            </>
          )}
          <p className="text-[10px] leading-snug text-slate-400">{wnn.method}</p>
        </div>
      )}
    </section>
  );
}
