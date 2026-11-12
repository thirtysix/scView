import type { ReactNode } from "react";
import { HoverTip } from "./InfoTip";
import { helpText } from "@/lib/help";

/**
 * A plain-language tip from the help catalog (`lib/help`) on the thing it explains: wrap the
 * heading, label text or column header, and hovering it opens the tip. The text lives in the
 * catalog so the same wording is used wherever the thing appears. An unknown id renders the
 * children alone.
 *
 *   <Help id="de.min-fc">Min |log2FC|</Help>
 */
export function Help({
  id,
  children,
  side = "above",
  align = "left",
  width = 300,
  className,
}: {
  id: string;
  children: ReactNode;
  side?: "above" | "below";
  align?: "left" | "right";
  width?: number;
  /** Wrapper classes, e.g. "flex items-center gap-1" when it wraps a row rather than text. */
  className?: string;
}) {
  const text = helpText(id);
  if (!text) return <>{children}</>;
  return (
    <HoverTip tip={text} side={side} align={align} width={width} className={className}>
      {children}
    </HoverTip>
  );
}
