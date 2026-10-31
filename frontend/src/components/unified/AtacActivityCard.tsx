import { useCallback, useState } from "react";
import { Loader2 } from "lucide-react";
import { getAtacActivity, type AtacActivity, type AtacState } from "@/api/atac";
import { Caption, CapNote } from "@/components/common/Caption";
import { GeneSearch } from "@/components/common/GeneSearch";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import Plot from "@/components/plots/Plot";
import { errorMessage } from "@/lib/errorMessage";
import { plotConfig } from "@/lib/plotConfig";

const AXIS = { tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9", zeroline: false };

/** Gene activity: the overlay on the scatter, and the gene's activity against its RNA. */
export function AtacActivityCard({
  datasetId,
  state,
  groupby,
  activeGene,
  onActivity,
}: {
  datasetId: string;
  state: AtacState;
  groupby: string;
  activeGene: string | null;
  onActivity: (a: AtacActivity) => void;
}) {
  const [shown, setShown] = useState<AtacActivity | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const summary = state.activity ?? null;

  const pick = useCallback(
    async (gene: string) => {
      setBusy(true);
      setError(null);
      try {
        const a = await getAtacActivity(datasetId, gene, groupby || undefined);
        setShown(a);
        onActivity(a);
      } catch (e) {
        setError(errorMessage(e));
      } finally {
        setBusy(false);
      }
    },
    [datasetId, groupby, onActivity],
  );

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Gene activity">
      <h4 className="flex items-center gap-1 text-xs font-semibold text-slate-600">
        <span>Gene activity</span>
        <InfoTip label="About gene activity" width={300}>
          Peak counts within the gene body + 2 kb upstream, summed per gene, log1p per 10k. It is peak-based: reads in a gene
          outside every called peak are not counted, unlike Signac&rsquo;s fragment-based GeneActivity.
        </InfoTip>
      </h4>
      {!summary && <p className="text-[11px] leading-snug text-slate-500">Run ATAC analysis to compute gene activity.</p>}
      {summary && !summary.available && (
        <p className="text-[11px] leading-snug text-amber-700" data-testid="atac-activity-reason">{summary.reason ?? "Not computed in the last run."}</p>
      )}
      {summary?.available && (
        <>
          <GeneSearch datasetId={datasetId} onSelect={(g) => void pick(g)} placeholder="Gene for the activity overlay" />
          <Caption>
            {(summary.n_genes_with_peaks ?? 0).toLocaleString()} of {(summary.n_genes ?? 0).toLocaleString()} genes have a peak in
            their region ({summary.source}). {summary.caption}
          </Caption>
        </>
      )}
      {busy && (
        <p className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <Loader2 className="h-3 w-3 animate-spin" /> Reading its activity…
        </p>
      )}
      <RequestError message={error} />
      {shown && shown.rna && (
        <div data-testid="atac-activity-vs-rna">
          <Plot
            data={[{
              type: "scattergl" as const, mode: "markers" as const, x: shown.rna.rna, y: shown.rna.activity,
              marker: { size: 3, color: "#2563eb", opacity: 0.35 },
              hovertemplate: "RNA %{x:.2f}<br>Activity %{y:.2f}<extra></extra>",
            }]}
            layout={{
              height: 210, margin: { t: 6, r: 8, b: 36, l: 44 },
              xaxis: { ...AXIS, title: { text: `${shown.gene} RNA`, font: { size: 10, color: "#64748b" } } },
              yaxis: { ...AXIS, title: { text: `${shown.gene} activity`, font: { size: 10, color: "#64748b" } } },
              showlegend: false, paper_bgcolor: "white", plot_bgcolor: "white", font: { family: "Inter, system-ui, sans-serif" },
            }}
            config={plotConfig(`atac_activity_vs_rna_${shown.gene}`)}
            useResizeHandler
            style={{ width: "100%" }}
          />
          <Caption n={{ value: shown.rna.n_shown, unit: "cells", of: shown.n_cells }}>
            {shown.gene}{shown.gene === activeGene ? " (on the plot)" : ""}: RNA against peak-based activity
            {shown.r !== null ? `, r = ${shown.r.toFixed(2)} over all cells` : ""}. {shown.n_nonzero.toLocaleString()} cells have
            activity.
          </Caption>
          <CapNote shown={shown.rna.n_shown} total={shown.n_cells} by="even spacing through the cells" />
        </div>
      )}
      {shown && !shown.rna && <Caption>{shown.gene} has no RNA values in this dataset to compare with.</Caption>}
    </section>
  );
}
