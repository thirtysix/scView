import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { getClusterEvidence, type ClusterEvidence, type ClusterEvidenceReport, type EvidenceLine } from "@/api/qc";
import { Help } from "@/components/common/Help";
import { errorMessage } from "@/lib/errorMessage";

/** The kind of an evidence line, with its tip (literal ids, so the help test can check them). */
function Kind({ kind }: { kind: EvidenceLine["kind"] }) {
  const cls = "shrink-0 font-medium text-slate-500";
  switch (kind) {
    case "markers":
      return <Help id="evidence.markers" className={cls}>Markers</Help>;
    case "qc":
      return <Help id="evidence.qc" className={cls}>QC</Help>;
    case "doublet":
      return <Help id="evidence.doublet" className={cls}>Doublet-like</Help>;
    case "content":
      return <Help id="evidence.content" className={cls}>Low RNA</Help>;
    case "makeup":
      return <Help id="evidence.makeup" className={cls}>Marker make-up</Help>;
    case "typing":
      return <Help id="evidence.types" className={cls}>Types</Help>;
    default:
      return <Help id="evidence.decision" className={cls}>Label</Help>;
  }
}

/** "Low-quality cells" from a recorded run, or what an AI annotation would give it. */
function LabelChip({ c }: { c: ClusterEvidence }) {
  if (!c.label) return null;
  return (
    <span
      className={`rounded border px-1 py-px text-[10px] font-medium ${
        c.decided ? "border-slate-300 bg-white text-slate-700" : "border-dashed border-slate-300 bg-white text-slate-500"
      }`}
    >
      {c.decided ? `scView: ${c.label}` : `would be ${c.label}`}
    </span>
  );
}

/** A request still pending after this long is a first (uncached) call: say what it is doing. */
const EVIDENCE_SLOW_MS = 1000;
const EVIDENCE_SLOW_TEXT =
  "Comparing each cluster's markers with random genes. The first time takes about 10 to 15 seconds.";

function Cluster({ c, open }: { c: ClusterEvidence; open: boolean }) {
  const fired = c.lines.some((l) => l.fired);
  return (
    <details open={open} className="group rounded border border-slate-100 px-2 py-1" data-testid="evidence-cluster">
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-1.5 text-xs">
        <span className={`font-medium ${fired ? "text-amber-800" : "text-slate-700"}`}>{c.group}</span>
        <span className="text-[11px] text-slate-400">{c.n_cells.toLocaleString()} cells</span>
        <LabelChip c={c} />
      </summary>
      <ul className="mt-1 space-y-0.5 text-[11px] text-slate-600">
        {c.lines.map((l, i) => (
          <li key={i} className="flex items-start gap-1.5">
            <Kind kind={l.kind} />
            <span className={l.fired ? "text-amber-800" : undefined}>{l.text}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

/**
 * Each cluster's evidence in plain words (`GET /clusters/evidence`): markers against random genes,
 * QC against the typical cluster, doublet-like parents, database types as shares of the evidence
 * and the label scView gave itself. Never a probability. `only` limits it to some groups (the ones
 * an annotation decided); clusters where a rule fired start open.
 */
export function ClusterEvidenceList({
  datasetId,
  column,
  only,
  refreshKey = 0,
}: {
  datasetId: string;
  column?: string | null;
  only?: string[];
  refreshKey?: number;
}) {
  const [data, setData] = useState<ClusterEvidenceReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Only a slow (first, uncached) answer says so; a cached one comes back at once.
  const [slow, setSlow] = useState(false);

  useEffect(() => {
    let live = true;
    setSlow(false);
    setError(null);
    const timer = setTimeout(() => live && setSlow(true), EVIDENCE_SLOW_MS);
    getClusterEvidence(datasetId, column || undefined)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(errorMessage(e)))
      .finally(() => {
        clearTimeout(timer);
        if (live) setSlow(false);
      });
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [datasetId, column, refreshKey]);

  const working = slow && (
    <p className="flex items-start gap-1 text-[11px] text-slate-500" role="status" data-testid="evidence-slow">
      <Loader2 className="mt-0.5 h-3 w-3 shrink-0 animate-spin" /> {EVIDENCE_SLOW_TEXT}
    </p>
  );
  if (error) return <p className="text-[11px] text-slate-400">Evidence unavailable: {error}</p>;
  if (!data) return working || null;
  const keep = only ? new Set(only) : null;
  const clusters = keep ? data.clusters.filter((c) => keep.has(c.group)) : data.clusters;
  return (
    <div className="space-y-1" data-testid="cluster-evidence">
      {working}
      {clusters.map((c) => (
        <Cluster key={c.group} c={c} open={!!keep || c.lines.some((l) => l.fired)} />
      ))}
      {!keep &&
        data.notes.map((n) => (
          <p key={n} className="text-[11px] text-slate-400">
            {n}
          </p>
        ))}
    </div>
  );
}
