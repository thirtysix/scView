import type { ReactNode } from "react";
import { HoverTip } from "@/components/common/InfoTip";
import { DOUBLET_METHOD_POINTS } from "@/lib/doubletMethods";

/** The doublet step's detector label, with what each choice does on hover. */
export function DoubletMethodInfo({ children }: { children: ReactNode }) {
  return (
    <HoverTip
      width={320}
      tip={
        <span className="block space-y-1">
          {DOUBLET_METHOD_POINTS.map((p) => (
            <span key={p} className="block">
              {p}
            </span>
          ))}
        </span>
      }
    >
      {children}
    </HoverTip>
  );
}
