import { useEffect, useState } from "react";
import { getLibraryConcordance, type LibraryConcordance } from "@/api/sampleQc";
import { Help } from "@/components/common/Help";

/** Libraries of one sample that disagree, for the samples the per-sample table shows
 *  (``column`` null: the backend's own pick, the same detection). Fetched once per
 *  column, not on every threshold change; a failure just leaves the cautions out. */
export function useLibraryConcordance(datasetId: string, column: string | null | undefined, refreshKey = 0,
  enabled = true) {
  const [data, setData] = useState<LibraryConcordance | null>(null);
  useEffect(() => {
    if (!enabled || !datasetId) return;
    let cancelled = false;
    getLibraryConcordance(datasetId, column ?? null)
      .then((d) => {
        if (!cancelled) setData(d && Array.isArray(d.samples) ? d : null);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, column, refreshKey, enabled]);
  return data;
}

/** The flagged rows of ``libs`` by sample name, when it judged the table's own samples. */
export function flaggedBySample(libs: LibraryConcordance | null | undefined, column: string | null) {
  const out = new Map<string, string[]>();
  if (!libs || libs.sample_column !== column) return out;
  for (const s of libs.samples) {
    if (s.flagged && s.message) out.set(s.sample, [...(out.get(s.sample) ?? []), s.message]);
  }
  return out;
}

/** A sample's caution chip in the table's Flags column. */
export function LibraryChip({ messages }: { messages: string[] }) {
  return (
    <span
      data-testid="libraries-differ"
      className="rounded border border-amber-200 bg-amber-50 px-1 text-[10px] text-amber-800"
    >
      <Help id="qc.libraries-differ" align="right">
        Libraries differ{messages.length > 1 ? ` (${messages.length})` : ""}
      </Help>
    </span>
  );
}

/** Under the table: each flagged sample's numbers, and what the check rests on. */
export function LibraryCaption({ libs }: { libs: LibraryConcordance }) {
  if (libs.status !== "checked" || libs.samples.length === 0) return null;
  const flagged = libs.samples.filter((s) => s.flagged && s.message);
  const n = libs.samples.length;
  return (
    <div className="space-y-1 text-[11px]" data-testid="library-caption">
      <p className={flagged.length ? "text-amber-800" : "text-slate-500"}>
        <Help id="qc.libraries">Libraries per sample</Help>: {n} {n === 1 ? "sample has" : "samples have"} two or
        more &apos;{libs.library_column}&apos; libraries
        {libs.split_by.length ? ` (compared within one ${libs.split_by.join(", ")})` : ""};{" "}
        {flagged.length ? `${flagged.length} with libraries that disagree.` : "their libraries agree."}
      </p>
      {flagged.map((s) => (
        <p key={s.name} className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-amber-900">
          {s.message}
        </p>
      ))}
      {libs.note && <p className="text-[10px] text-slate-400">{libs.note}</p>}
    </div>
  );
}
