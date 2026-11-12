import { InfoTip } from "./InfoTip";
import { helpText } from "@/lib/help";

/**
 * An info icon whose tip comes from the help catalog (`lib/help`). Use it next to a card
 * heading, a control's label or a column header; the text lives in the catalog so the
 * same wording is used wherever the thing appears. Renders nothing for an unknown id.
 */
export function Help({
  id,
  align = "left",
  width = 280,
  label = "More information",
  side = "above",
}: {
  id: string;
  align?: "left" | "right";
  width?: number;
  /** What the icon explains, read out by screen readers ("About the volcano plot"). */
  label?: string;
  /** Where the tip opens: "below" near the top of a scrolling box (see InfoTip). */
  side?: "above" | "below";
}) {
  const text = helpText(id);
  if (!text) return null;
  return (
    <InfoTip align={align} width={width} label={label} side={side}>
      {text}
    </InfoTip>
  );
}
