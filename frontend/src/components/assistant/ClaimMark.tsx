import type { ReactNode } from "react";
import type { Claim } from "@/api/assistant";
import { HoverTip } from "@/components/common/InfoTip";

// Quiet by design: a supported sentence is barely underlined, an uncertain one dotted,
// a refuted one wavy red on a light red ground (light classes only; index.css remaps them).
const MARK: Record<Claim["verdict"], string> = {
  supported: "underline decoration-dotted decoration-emerald-500/30 underline-offset-4",
  uncertain: "underline decoration-dotted decoration-amber-500/70 underline-offset-4",
  refuted: "rounded-sm bg-red-50 underline decoration-wavy decoration-red-500 underline-offset-4",
};

const LABEL: Record<Claim["verdict"], { text: string; cls: string }> = {
  supported: { text: "Supported", cls: "text-emerald-600" },
  uncertain: { text: "Uncertain", cls: "text-amber-600" },
  refuted: { text: "Refuted", cls: "text-red-600" },
};

const TIP_LINES = 4;
const TIP_LINE_CHARS = 160;

/** The fact lines a claim was checked against, short enough for a tip. */
export function tipEvidence(evidence: string): string {
  const lines = evidence.split("\n").filter((l) => l.trim());
  const shown = lines
    .slice(0, TIP_LINES)
    .map((l) => (l.length > TIP_LINE_CHARS ? `${l.slice(0, TIP_LINE_CHARS)}…` : l));
  return lines.length > TIP_LINES ? [...shown, "…"].join("\n") : shown.join("\n");
}

/** One checked sentence of an answer, marked in place: hovering it shows the verdict
 *  and the fact it was checked against. */
export function ClaimMark({ claim, children }: { claim: Claim; children: ReactNode }) {
  const label = LABEL[claim.verdict];
  const tip = (
    <span className="block">
      <span className={`block font-medium ${label.cls}`}>
        {label.text}
        {claim.checked_by === "model" ? " (checked by AI)" : ""}
      </span>
      {claim.reason && <span className="block">{claim.reason}</span>}
      {claim.evidence && (
        <span className="mt-1 block whitespace-pre-wrap break-words text-[10px] text-slate-500">
          {tipEvidence(claim.evidence)}
        </span>
      )}
    </span>
  );
  return (
    <HoverTip tip={tip} width={340}>
      <span data-claim={claim.verdict} className={MARK[claim.verdict]}>
        {children}
      </span>
    </HoverTip>
  );
}
