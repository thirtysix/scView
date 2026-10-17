import { useMemo } from "react";
import type { TranscriptomeSize } from "@/api/de";
import { ViolinPlot } from "@/components/plots/ViolinPlot";
import { sizeCaution } from "@/lib/transcriptomeSize";

interface Props {
  size: TranscriptomeSize;
  /** Display names of the two sides. */
  labelA: string;
  labelB: string;
  /** Obs column the two sides are values of, so the violin's colours match the scatter. */
  groupColumn?: string;
}

/**
 * The transcriptome-size caution under a DE result whose two sides differ a lot
 * in RNA content: what it does to fold changes, and each side's total counts per
 * cell as a violin (the Unified View's violin).
 */
export function DETranscriptomeSize({ size, labelA, labelB, groupColumn }: Props) {
  // "rest" can also be a group's name: keep the two violins apart.
  const nameB = labelB === labelA ? `${labelB} (reference)` : labelB;
  const data = useMemo(
    () => ({ [labelA]: size.values_a, [nameB]: size.values_b }),
    [size, labelA, nameB],
  );
  return (
    <div role="alert" className="rounded-lg border border-amber-200 bg-amber-50 p-2 text-[10px] leading-snug text-amber-900">
      <p>
        <span className="font-semibold">Transcriptome size differs: </span>
        {sizeCaution(size, labelA, labelB)}
      </p>
      <div className="mt-1.5 rounded border border-amber-100 bg-white">
        <ViolinPlot
          data={data}
          groupColumn={groupColumn}
          yLabel="Total counts per cell"
          height={170}
        />
      </div>
      <p className="mt-1 text-amber-800/80">
        Raw counts from <code>{size.source}</code>: medians over every cell, violins from up to
        2,000 cells a side. The caution shows at {size.threshold}x or more.
      </p>
    </div>
  );
}
