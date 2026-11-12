import type { ReactNode } from "react";
import { FAST_LEIDEN_POINTS } from "@/lib/fastLeiden";
import { HoverTip } from "./InfoTip";

/** Hovering `children` (a fast-Leiden choice's label or sentence) tells what it changes, and what it does not. */
export function FastLeidenInfo({ children, align = "left" }: { children: ReactNode; align?: "left" | "right" }) {
  return (
    <HoverTip
      width={320}
      align={align}
      tip={
        <span className="block space-y-1">
          {FAST_LEIDEN_POINTS.map((p) => (
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
