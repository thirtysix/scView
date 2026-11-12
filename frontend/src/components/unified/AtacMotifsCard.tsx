import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { cancelScan, getMotifState, scanPeaks, type ChromVarMotif, type MotifState } from "@/api/atacMotifs";
import { Caption } from "@/components/common/Caption";
import { HoverTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { scanBlockedText, scanDownloadText, scanEstimateText } from "@/lib/atacMotifs";
import { errorMessage } from "@/lib/errorMessage";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { AtacChromvar } from "./AtacChromvar";
import { AtacMotifEnrichment } from "./AtacMotifEnrichment";

/**
 * "Motifs in peaks": the JASPAR 2024 motifs in every peak (backend core/atac_motifs.py), then
 * which are enriched in a group's differential peaks and each cell's chromVAR activity.
 */
export function AtacMotifsCard({
  datasetId,
  groupby,
  activeMotif,
  onMotif,
}: {
  datasetId: string;
  groupby: string;
  activeMotif: string | null;
  onMotif: (m: ChromVarMotif) => void;
}) {
  const version = useAtacRunStore((s) => s.version);
  const [state, setState] = useState<MotifState | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const following = useRef<AbortController | null>(null);

  const reload = useCallback(() => {
    return getMotifState(datasetId)
      .then(setState)
      .catch((e) => setError(errorMessage(e)));
  }, [datasetId]);

  useEffect(() => {
    void reload();
  }, [reload, version]);

  /** Starts the scan, or follows the one already running (it goes on without this tab). */
  const follow = useCallback(async () => {
    following.current?.abort();
    const controller = new AbortController();
    following.current = controller;
    setScanning(true);
    setError(null);
    try {
      const out = await scanPeaks(datasetId, { signal: controller.signal, onProgress: (p) => setMessage(p.message ?? null) });
      if (out === "cancelled") setMessage("The scan was stopped; nothing was kept.");
      else setMessage(null);
      await reload();
    } catch (e) {
      if (!controller.signal.aborted) setError(errorMessage(e));
    } finally {
      if (!controller.signal.aborted) setScanning(false);
    }
  }, [datasetId, reload]);

  const running = !!state?.scan?.running;
  useEffect(() => {
    if (running && !following.current) void follow();
  }, [running, follow]);
  useEffect(
    () => () => {
      following.current?.abort();
      following.current = null;
    },
    [datasetId],
  );

  const stop = async () => {
    try {
      await cancelScan(datasetId);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const scan = state?.scan;
  const meta = scan?.meta ?? null;
  const blocked = state ? scanBlockedText(state) : null;
  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Motifs in peaks" data-testid="atac-motifs">
      <h4 className="text-xs font-semibold text-slate-600">
        <HoverTip
          width={340}
          tip={<>
            The 879 JASPAR 2024 motifs in every peak, with the same hits as Signac&rsquo;s AddMotifs (motifmatchr: p &lt; 5e-5, the peaks&rsquo;
            own base frequencies, both strands). A motif hit is not binding. Paralogous TFs share motifs, so the name stands for a family.
            GC-rich motifs enrich easily even after matching the background on GC. A chromVAR score is relative to matched peaks across these
            cells, not an amount of TF.
          </>}
        >
          Motifs in peaks
        </HoverTip>
      </h4>
      <RequestError message={error} />
      {state && !state.available && <p className="text-[11px] leading-snug text-slate-500">{state.reason}</p>}
      {state?.available && scan && (
        <>
          {!meta && (
            <div className="space-y-1.5" data-testid="atac-motif-scan">
              {!scanning && (
                <p className="text-[11px] leading-snug text-slate-500">
                  {scanDownloadText(state)} {scanEstimateText(state)}
                </p>
              )}
              {blocked && <p className="text-[11px] leading-snug text-amber-700">{blocked}</p>}
              {!scanning && (
                <button type="button" disabled={!!blocked} onClick={() => void follow()}
                        className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50">
                  Scan peaks
                </button>
              )}
              {scanning && (
                <div className="space-y-1" data-testid="atac-motif-scan-progress">
                  <p className="flex items-center gap-1.5 text-[11px] text-slate-600">
                    <Loader2 className="h-3 w-3 animate-spin" /> {message ?? "Starting…"}
                  </p>
                  <button type="button" onClick={() => void stop()}
                          className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-600 hover:border-slate-400">
                    Cancel
                  </button>
                </div>
              )}
              {!scanning && message && <p className="text-[11px] leading-snug text-slate-500">{message}</p>}
            </div>
          )}
          {meta && (
            <>
              <Caption n={[{ value: meta.n_scanned, unit: "peaks" }, { value: meta.n_motifs, unit: "motifs" }]} testId="atac-motif-scan-done">
                {state.genome}: {(100 * meta.hit_rate).toFixed(1)}% of peaks x motifs have a hit.
                {meta.n_left_out > 0 && ` ${meta.n_left_out.toLocaleString()} peaks on contigs the genome file names differently are left out.`}
                {meta.n_best_word_only > 0 && ` ${meta.n_best_word_only} short motifs cannot reach p < 5e-5, so they hit only on their best word, as in motifmatchr.`}
              </Caption>
              <AtacMotifEnrichment datasetId={datasetId} />
              <AtacChromvar datasetId={datasetId} state={state} groupby={groupby} activeMotif={activeMotif} onMotif={onMotif} />
            </>
          )}
        </>
      )}
    </section>
  );
}
