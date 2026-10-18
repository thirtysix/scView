import { blendColor } from "@/lib/geneCoexpression";

const STEPS = 6;

/**
 * The two-gene blend's colour key: gene A rises left to right, gene B bottom
 * to top (grey neither, red A, green B, yellow both), as the blend panel
 * colours each cell.
 */
export function BlendLegend({ geneA, geneB }: { geneA: string; geneB: string }) {
  const rows = Array.from({ length: STEPS }, (_, r) => STEPS - 1 - r); // B high at the top
  return (
    <div className="text-[10px] text-slate-500" aria-label={`Blend key: ${geneA} red, ${geneB} green, both yellow`}>
      <div className="mb-1 text-[11px] font-semibold text-slate-600">Blend</div>
      <div className="flex items-stretch gap-1">
        <div className="flex items-center">
          <span className="max-w-[60px] truncate [writing-mode:vertical-rl] rotate-180" title={geneB}>
            {geneB} →
          </span>
        </div>
        <div
          className="grid h-16 w-16 overflow-hidden rounded-sm border border-slate-200"
          style={{ gridTemplateColumns: `repeat(${STEPS}, 1fr)` }}
        >
          {rows.flatMap((r) =>
            Array.from({ length: STEPS }, (_, c) => {
              const [red, green, blue] = blendColor(c / (STEPS - 1), r / (STEPS - 1));
              return <div key={`${r}-${c}`} style={{ background: `rgb(${red},${green},${blue})` }} />;
            }),
          )}
        </div>
      </div>
      <div className="ml-4 mt-0.5 max-w-[80px] truncate" title={geneA}>
        {geneA} →
      </div>
    </div>
  );
}
