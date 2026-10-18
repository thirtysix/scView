import { useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { getSexCheck, type SexCall, type SexCheck, type SexGroup } from "@/api/sampleQc";

export const SEX_LABELS: Record<SexCall, string> = {
  female: "F",
  male: "M",
  mixed: "F+M",
  ambiguous: "?",
  undetermined: "–",
};

const SEX_WORDS: Record<SexCall, string> = {
  female: "female",
  male: "male",
  mixed: "mixed (both sexes)",
  ambiguous: "ambiguous",
  undetermined: "undetermined",
};

/** The sex check for the samples the per-sample table shows (``column`` null: the
 *  backend's own pick, which is the same detection). Fetched once per column, not
 *  on every threshold change; a failure just leaves the column out. */
export function useSexCheck(datasetId: string, column: string | null | undefined, refreshKey = 0, enabled = true) {
  const [data, setData] = useState<SexCheck | null>(null);
  useEffect(() => {
    if (!enabled || !datasetId) return;
    let cancelled = false;
    getSexCheck(datasetId, column ?? null)
      .then((d) => {
        if (!cancelled) setData(d);
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

function title(g: SexGroup, metaColumn: string | null): string {
  const conf = g.confidence === "low" ? " (low confidence)" : "";
  const meta =
    metaColumn && g.metadata ? ` ${metaColumn}: ${g.metadata}${g.mismatch ? ", which DISAGREES" : ""}.` : "";
  return `${SEX_WORDS[g.call]}${conf}: ${g.evidence}${meta}`;
}

/** One sample's call as a small chip; red when the sex metadata says otherwise. */
export function SexChip({ group, metaColumn }: { group: SexGroup; metaColumn: string | null }) {
  const tone = group.mismatch
    ? "border-red-300 bg-red-50 text-red-800"
    : group.call === "mixed"
      ? "border-amber-200 bg-amber-50 text-amber-800"
      : "border-slate-200 bg-slate-50 text-slate-600";
  return (
    <span
      title={title(group, metaColumn)}
      data-testid={`sex-${group.name}`}
      className={`rounded border px-1 text-[10px] ${tone} ${group.confidence === "low" ? "italic" : ""}`}
    >
      {SEX_LABELS[group.call]}
      {group.mismatch ? " ≠" : ""}
    </span>
  );
}

/** Under the per-sample table: what the calls rest on, in one line, then any
 *  mismatch / mixed-sample / confounding warnings. */
export function SexCaption({ sex }: { sex: SexCheck }) {
  const n = sex.groups.length;
  const counts = (Object.keys(SEX_LABELS) as SexCall[])
    .filter((c) => sex.summary[c])
    .map((c) => `${sex.summary[c]} ${c}`)
    .join(", ");
  const genes = [
    sex.female_genes.length ? sex.female_genes.join("/") : null,
    sex.male_genes.length ? `Y genes (${sex.male_genes.join(", ")})` : null,
  ]
    .filter(Boolean)
    .join(" vs ");
  return (
    <div className="space-y-1" data-testid="sex-caption">
      <p className="text-[10px] text-slate-500">
        <span className="font-medium text-slate-600">Sex from expression</span>
        {genes ? ` (${genes}, share of cells with any count; on ≥ ${pct(sex.thresholds.on)}, off ≤ ${pct(sex.thresholds.off)})` : ""}
        {sex.column ? ` by ${sex.column}, ${n} samples` : ", all cells"}: {counts || "nothing to call"}.
        {sex.metadata_column ? ` Compared with '${sex.metadata_column}'.` : ""} Hover a chip for its numbers.
      </p>
      {sex.notes.map((t) => (
        <p key={t} className="text-[10px] text-slate-400">
          {t}
        </p>
      ))}
      {sex.warnings.map((w) => (
        <p key={w} className="flex items-start gap-1 rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
          <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
          <span>{w}</span>
        </p>
      ))}
    </div>
  );
}

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}
