import { useEffect, useState } from "react";
import { AlertTriangle, Info, Loader2 } from "lucide-react";
import { getAmbient, type AmbientResponse } from "@/api/ambient";
import { isLargeRefusal } from "@/api/client";
import { errorMessage } from "@/lib/errorMessage";
import { InfoTip } from "@/components/common/InfoTip";
import { Caption } from "@/components/common/Caption";
import { pct, rhoStatusText, rhoText } from "@/lib/ambient";

const TH = "whitespace-nowrap px-1.5 py-1 text-left font-medium";
const NUM = "whitespace-nowrap px-1.5 py-1 text-right tabular-nums";
const SOUP_SHOWN = 6;

/** The correction step's setting, as the card shows and changes it (the step list holds it). */
export interface AmbientStepControl {
  selected: boolean;
  onSelectedChange: (on: boolean) => void;
  /** The step's rho: null = each sample's own estimate. */
  rho: number | null;
}

/** Data Assessment card: each sample's ambient ("soup") RNA, its contamination fraction
 *  (rho, SoupX-style) or why there is none, the soup's top genes, how many markers look
 *  ambient, and the opt-in correction step. */
export function AmbientRna({
  datasetId,
  refreshKey = 0,
  step,
}: {
  datasetId: string;
  refreshKey?: number;
  step?: AmbientStepControl;
}) {
  const [data, setData] = useState<AmbientResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getAmbient(datasetId)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(errorMessage(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, refreshKey]);

  const samples = data?.samples ?? [];
  const help = data?.help ?? {};
  const flagged = data?.markers?.n_flagged ?? 0;
  const groups = Object.keys(data?.markers?.groups ?? {}).length;
  const correction = data?.correction ?? null;
  const nCells = samples.reduce((t, s) => t + (s.n_cells ?? 0), 0);

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3" data-testid="ambient-rna">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold text-slate-700">Ambient RNA</h3>
          <InfoTip label="About ambient RNA" width={340}>
            Free-floating RNA from lysed cells lands in every droplet, so abundant transcripts of one cell type (LYZ,
            haemoglobin, IGKC) show up at a low level in all clusters. The soup is read from the empty droplets of the
            raw (unfiltered) matrix, and the contamination fraction rho is estimated as SoupX does (Young &amp; Behjati
            2020, GigaScience). {help.rho ?? ""}
          </InfoTip>
        </div>
        {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>

      {error && (
        <p className={`text-xs ${isLargeRefusal(error) ? "text-slate-500" : "text-red-600"}`}>{error}</p>
      )}

      {data && !data.available && (
        <p className="flex items-start gap-1.5 text-xs text-slate-600" data-testid="ambient-no-raw">
          <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-slate-400" />
          <span>
            No raw (unfiltered) matrix is attached, so there are no empty droplets to read the soup from and rho
            can&apos;t be estimated: attach <code>raw_feature_bc_matrix.h5</code> in Sequencing QC. Until then the
            cells&apos; own counts stand in for the soup (approximate), and no markers are flagged.
          </span>
        </p>
      )}

      {samples.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] text-slate-600">
            <thead className="text-slate-500">
              <tr>
                <th className={TH}>Sample</th>
                <th className={`${TH} text-right`}>Cells</th>
                <th className={`${TH} text-right`} title={help.rho}>
                  Contamination (rho)
                </th>
                <th className={TH} title={help.soup}>
                  Top soup genes
                </th>
              </tr>
            </thead>
            <tbody>
              {samples.map((s) => (
                <tr key={s.sample} className="border-t border-slate-100" data-testid="ambient-sample">
                  <td className="px-1.5 py-1 font-medium text-slate-700">
                    {s.sample}
                    {s.source === "cells" && (
                      <span className="ml-1 text-[10px] font-normal text-slate-400" title={help.approximate}>
                        (approximate)
                      </span>
                    )}
                  </td>
                  <td className={NUM}>{(s.n_cells ?? 0).toLocaleString()}</td>
                  <td
                    className={`${NUM} ${s.rho !== null && s.rho >= 0.2 ? "font-semibold text-amber-700" : ""}`}
                    title={[s.rho !== null ? `From ${s.n_markers} marker genes, ${s.n_estimates} estimates.` : "", ...(s.notes ?? [])]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    {s.rho !== null ? rhoText(s) : <span className="text-slate-400">{rhoStatusText(s)}</span>}
                  </td>
                  <td className="px-1.5 py-1">
                    {(s.top_genes ?? []).slice(0, SOUP_SHOWN).map((g) => `${g.gene} ${pct(g.share)}`).join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {samples.flatMap((s) => s.notes ?? []).length > 0 && (
        <ul className="space-y-0.5 text-[11px] text-slate-500">
          {samples.flatMap((s) => (s.notes ?? []).map((n, i) => <li key={`${s.sample}-${i}`}>{s.sample}: {n}</li>))}
        </ul>
      )}

      {data && flagged > 0 && (
        <p className="flex items-start gap-1.5 text-xs text-slate-600" data-testid="ambient-flag-count">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-amber-500" />
          <span>
            {flagged} top marker{flagged === 1 ? "" : "s"} in {groups} cluster{groups === 1 ? "" : "s"} of{" "}
            {data.markers.column} {data.markers.approximate ? "may spill as ambient RNA" : "look ambient outside their cluster"}:
            flagged in the Markers table and the cluster QC card.
          </span>
        </p>
      )}
      {data?.hb_note && <p className="text-xs text-slate-600">{data.hb_note}</p>}
      {(data?.notes ?? []).map((n, i) => (
        <p key={i} className="text-[11px] text-slate-500">
          {n}
        </p>
      ))}

      {correction ? (
        <p className="text-xs text-slate-600" data-testid="ambient-corrected">
          Counts corrected for ambient RNA:{" "}
          {(correction.samples ?? []).map((r) => `${r.sample} rho ${pct(r.rho)} (${r.rho_source})`).join("; ")}
          {correction.removed_share !== undefined ? `, ${pct(correction.removed_share)} of the UMIs removed` : ""}. The
          originals are kept; reset to counts brings them back.
        </p>
      ) : (
        step &&
        data && (
          <label
            className={`flex items-start gap-1.5 text-xs ${data.available ? "text-slate-600" : "text-slate-400"}`}
            title={data.available ? help.correction : "Needs each sample's raw matrix (the soup) attached."}
          >
            <input
              type="checkbox"
              checked={step.selected}
              disabled={!data.available}
              onChange={(e) => step.onSelectedChange(e.target.checked)}
              className="mt-0.5 h-3.5 w-3.5"
              data-testid="ambient-step-toggle"
            />
            <span>
              Correct the counts for ambient RNA in the next run (the Ambient RNA Correction step, off by default):{" "}
              {step.rho === null ? "each sample's own rho" : `rho ${pct(step.rho)} for every sample`}.
            </span>
          </label>
        )
      )}

      {data && samples.length > 0 && (
        <Caption n={{ value: nCells, unit: "cells" }}>
          {data.available
            ? `Soup from droplets with 1 to ${(samples[0]?.umi_max ?? 100) - 1} UMIs in the raw matrix; rho by SoupX's autoEstCont defaults on ${data.column ?? "the clusters"}.`
            : "Approximate soup: the cells' own summed counts per capture channel."}
        </Caption>
      )}
    </div>
  );
}
