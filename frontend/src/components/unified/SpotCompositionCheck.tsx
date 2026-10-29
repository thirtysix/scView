import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { getSpotCheck, runSpotCheck, type SpotCheck } from "@/api/spotDeconvolution";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { errorMessage } from "@/lib/errorMessage";
import { SPOT_METHOD_LABELS } from "@/lib/spotComposition";

const num = (v: number | null | undefined, d = 3) => (v == null ? "-" : v.toFixed(d));

/**
 * The spot composition self-check: synthetic spots of held-out reference cells, with
 * known shares, fitted by both methods (backend spot_deconvolution.iter_check).
 */
export function SpotCompositionCheck({
  datasetId,
  referenceId,
  celltypeColumn,
  disabled,
}: {
  datasetId: string;
  referenceId: string;
  celltypeColumn: string;
  disabled?: boolean;
}) {
  const [check, setCheck] = useState<SpotCheck | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    let live = true;
    setCheck(null);
    getSpotCheck(datasetId)
      .then((c) => live && setCheck(c))
      .catch(() => {}); // null when none yet
    return () => {
      live = false;
      abort.current?.abort();
    };
  }, [datasetId]);

  const run = async () => {
    setRunning(true);
    setError(null);
    abort.current = new AbortController();
    try {
      setCheck(
        await runSpotCheck(
          datasetId,
          { reference_id: referenceId, celltype_column: celltypeColumn },
          { signal: abort.current.signal, onProgress: (m) => setProgress(m) },
        ),
      );
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(errorMessage(e));
    } finally {
      setRunning(false);
      setProgress("");
    }
  };

  const best = check?.runs.find((r) => r.method === "rctd" && r.depth === check.depths[0]) ?? check?.runs[0];

  return (
    <div className="space-y-2 rounded border border-slate-200 p-2" data-testid="spot-check">
      <div className="flex items-center gap-1.5 font-medium text-slate-700">
        Self-check
        <InfoTip label="About the self-check" width={320}>
          Half of each reference type&apos;s cells make the profiles; the other half are pooled into synthetic spots of 2
          to 10 cells with known shares, thinned to Visium-like depths, and fitted by both methods. RMSE and JSD: how far
          the estimates are from the truth (0 is perfect). AUPR: how well a type&apos;s presence is told from its absence.
        </InfoTip>
        <button
          className="ml-auto rounded border px-2 py-0.5 font-normal hover:bg-slate-50 disabled:opacity-50"
          disabled={disabled || running}
          onClick={run}
        >
          {running ? <Loader2 className="inline h-3 w-3 animate-spin" /> : null} Run self-check
        </button>
      </div>
      {running && (
        <p className="text-slate-600">
          {progress}{" "}
          <button className="text-slate-500 hover:underline" onClick={() => abort.current?.abort()}>
            Cancel
          </button>
        </p>
      )}
      {error && <p className="text-red-600">{error}</p>}
      {check && !running && (
        <>
          <table className="w-full" data-testid="spot-check-table">
            <thead>
              <tr className="text-left text-slate-500">
                <th>Method</th>
                <th className="text-right" title="Counts per synthetic spot (median)">Depth</th>
                <th className="text-right" title="Root mean squared error of the RNA shares">RMSE</th>
                <th className="text-right" title="Jensen-Shannon divergence per spot (0-1), mean">JSD</th>
                <th className="text-right" title="Area under the precision-recall curve for a type being present">AUPR</th>
              </tr>
            </thead>
            <tbody>
              {check.runs.map((r) => (
                <tr key={`${r.method}-${r.depth}`}>
                  <td>{SPOT_METHOD_LABELS[r.method]}</td>
                  <td className="text-right">{Math.round(r.median_umi).toLocaleString()}</td>
                  <td className="text-right">{num(r.rmse)}</td>
                  <td className="text-right">{num(r.jsd)}</td>
                  <td className="text-right">{num(r.aupr, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {best && (
            <p className="text-slate-500">
              Per type ({SPOT_METHOD_LABELS[best.method]}, r between estimate and truth):{" "}
              {best.per_type.map((t) => `${t.type} ${num(t.r, 2)}`).join(", ")}.
            </p>
          )}
          <Caption n={{ value: check.n_spots, unit: "synthetic spots per run" }}>
            From {check.reference.name} ({check.reference.celltype_column}). {check.note}
          </Caption>
        </>
      )}
    </div>
  );
}
