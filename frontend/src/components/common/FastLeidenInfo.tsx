import { FAST_LEIDEN_POINTS } from "@/lib/fastLeiden";
import { InfoTip } from "./InfoTip";

/** The info icon beside a fast-Leiden choice: what it changes, and what it does not. */
export function FastLeidenInfo({ align = "left" }: { align?: "left" | "right" }) {
  return (
    <InfoTip width={320} align={align} label="What fast Leiden changes">
      <span className="block space-y-1">
        {FAST_LEIDEN_POINTS.map((p) => (
          <span key={p} className="block">
            {p}
          </span>
        ))}
      </span>
    </InfoTip>
  );
}
