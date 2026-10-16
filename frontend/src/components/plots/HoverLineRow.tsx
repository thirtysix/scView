import type { HoverLine } from "@/lib/hoverInfo";
import { rgbCss } from "@/lib/categoryColors";

/** One "coloured by" line of a scatter tooltip: swatch, label, the cell's value. */
export function HoverLineRow({ line }: { line: HoverLine }) {
  return (
    <div className="flex items-center gap-1.5" data-testid="hover-colour-line">
      {line.color && (
        <span
          className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full"
          style={{ background: rgbCss(line.color) }}
          aria-hidden
        />
      )}
      <span className="flex-shrink-0 text-slate-500">{line.label}:</span>
      <span className="truncate font-semibold text-slate-800">{line.value}</span>
    </div>
  );
}
