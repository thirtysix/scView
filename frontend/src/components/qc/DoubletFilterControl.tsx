import type { FilterPreview, FilterThresholds } from "@/api/qc";
import { InfoTip } from "@/components/common/InfoTip";
import { asDoubletSource, choiceText, sourceOptionLabel } from "@/lib/doubletSource";

/**
 * The filtering step's doublet options under the cell-loss summary: drop predicted
 * doublets, whose calls (a choice once cell hashing and Scrublet both exist; auto =
 * hashing), and cell hashing's negatives, a separate opt-in (they are not doublets).
 */
export function DoubletFilterControl({
  preview,
  thresholds,
  onChange,
}: {
  preview: FilterPreview;
  thresholds: FilterThresholds;
  onChange: (patch: Partial<FilterThresholds>) => void;
}) {
  if (!preview.doublet_calls) return null;
  const choices = preview.doublets ?? null; // older backends: the checkbox alone
  const source = asDoubletSource(thresholds.qc_doublet_source);
  // auto plus at least hashing and Scrublet: a real choice to make.
  const choose = choices !== null && choices.options.length > 2;
  return (
    <div className="space-y-1" data-testid="doublet-filter">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-slate-600">
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={thresholds.drop_doublets}
            onChange={(e) => onChange({ drop_doublets: e.target.checked })}
            className="h-3 w-3"
          />
          Also drop predicted doublets
        </label>
        {choose && (
          <label className="flex items-center gap-1">
            <span className="text-slate-500">by</span>
            <select
              aria-label="Doublet source"
              value={source}
              onChange={(e) => onChange({ qc_doublet_source: asDoubletSource(e.target.value) })}
              className="rounded border border-slate-200 bg-white px-1 py-0.5 text-[11px]"
            >
              {choices.options.map((o) => (
                <option key={o} value={o}>
                  {sourceOptionLabel(o, choices)}
                </option>
              ))}
            </select>
            <InfoTip label="About the doublet sources">
              Cell hashing sees doublets of two samples by their tags: the ground truth for those. It cannot
              see two cells of one sample; Scrublet, from simulated doublets of the cells&apos; own profiles,
              is the only call that can (the scDblFinder-style classifier too, when the doublet step ran it).
              Auto uses hashing when it was run, else the cells both detectors call when both ran, else the
              one that ran; Either drops a cell any of them calls. The counts are cells called on the current
              data.
            </InfoTip>
          </label>
        )}
        {choices && !choose && <span className="text-slate-400">({choices.auto.label})</span>}
      </div>
      {thresholds.drop_doublets && choices && (
        <p className="text-[11px] text-slate-500" data-testid="doublet-filter-choice">
          {choiceText(choices)}
        </p>
      )}
      {choices && choices.hash_negatives != null && (
        <div>
          <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
            <input
              type="checkbox"
              checked={Boolean(thresholds.drop_hash_negatives)}
              onChange={(e) => onChange({ drop_hash_negatives: e.target.checked })}
              className="h-3 w-3"
            />
            Also drop hashing negatives ({choices.hash_negatives.toLocaleString()} cells)
          </label>
          <p className="pl-[18px] text-[11px] text-slate-400">{choices.negative_note}</p>
        </div>
      )}
    </div>
  );
}
