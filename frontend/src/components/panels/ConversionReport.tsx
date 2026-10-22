import { FileCode2 } from "lucide-react";

import type { ConversionReport as Report } from "@/api/types";
import { InfoTip } from "@/components/common/InfoTip";
import { conversionHeadline } from "@/lib/conversion";

/**
 * What the R converter took from an .rds / .RData file and where it put it,
 * and what it left out (with why). Shown on Data Assessment for a converted
 * dataset; the same report is stored in the dataset (uns scview_conversion).
 */
export function ConversionReport({ report }: { report: Report }) {
  const kept = report.kept ?? [];
  const skipped = report.skipped ?? [];
  const listed = kept.length + skipped.length > 0;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm" data-testid="conversion-report">
      <div className="flex items-start gap-2">
        <FileCode2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-slate-400" />
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1.5 text-slate-700">
            <span>{conversionHeadline(report)}</span>
            <InfoTip label="About the conversion" width={300}>
              The R converter read the object with its own Bioconductor or Seurat accessors and wrote
              an AnnData file: assays become X and layers, colData the cell table (obs), rowData the
              gene table (var), reducedDims embeddings (obsm). This list is stored with the dataset
              (uns scview_conversion), so a later export still says where everything came from.
            </InfoTip>
          </p>
          {listed && (
            <details className="mt-1.5" open={skipped.length > 0}>
              <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-700">
                {kept.length} kept
                {skipped.length > 0 ? `, ${skipped.length} not converted` : ", nothing left out"}
              </summary>
              <div className="mt-2 grid gap-3 text-xs sm:grid-cols-2">
                <div>
                  <p className="mb-1 font-medium text-slate-600">Kept</p>
                  <ul className="list-disc space-y-0.5 pl-4 text-slate-600">
                    {kept.map((k) => (
                      <li key={k}>{k}</li>
                    ))}
                  </ul>
                </div>
                {skipped.length > 0 && (
                  <div>
                    <p className="mb-1 font-medium text-amber-700">Not converted</p>
                    <ul className="list-disc space-y-0.5 pl-4 text-slate-600">
                      {skipped.map((k) => (
                        <li key={k}>{k}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}
