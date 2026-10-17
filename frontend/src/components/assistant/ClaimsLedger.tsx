import { useState } from "react";
import {
  CheckCircle2, ChevronDown, ChevronRight, HelpCircle, Loader2, ShieldCheck, XCircle,
} from "lucide-react";
import type { Claim } from "@/api/assistant";

/** Mirrors MAX_JUDGE_CLAIMS on the server: claims per Check claims call. */
export const MAX_CHECK = 12;

const VERDICT = {
  supported: { icon: CheckCircle2, cls: "text-emerald-600", label: "supported" },
  refuted: { icon: XCircle, cls: "text-red-600", label: "refuted" },
  uncertain: { icon: HelpCircle, cls: "text-amber-600", label: "uncertain" },
} as const;

export function tally(claims: Claim[]): Record<Claim["verdict"], number> {
  const t = { supported: 0, refuted: 0, uncertain: 0 };
  for (const c of claims) t[c.verdict] += 1;
  return t;
}

/** The claims a Check claims call would send: uncertain by rule and not judged yet. */
export function toCheck(claims: Claim[]): Claim[] {
  return claims
    .filter((c) => c.verdict === "uncertain" && c.checked_by === "rules")
    .slice(0, MAX_CHECK);
}

/** Fold the model's verdicts into the ledger; a claim it judged is never re-sent. */
export function mergeVerdicts(
  claims: Claim[],
  judged: Pick<Claim, "id" | "verdict" | "reason">[],
): Claim[] {
  const byId = new Map(judged.map((j) => [j.id, j]));
  return claims.map((c) => {
    const j = byId.get(c.id);
    return j ? { ...c, verdict: j.verdict, reason: j.reason || c.reason, checked_by: "model" } : c;
  });
}

function ClaimRow({ c }: { c: Claim }) {
  const v = VERDICT[c.verdict];
  const Icon = v.icon;
  return (
    <li className="flex gap-1.5">
      <Icon className={`mt-0.5 h-3.5 w-3.5 flex-shrink-0 ${v.cls}`} aria-label={v.label} />
      <div className="min-w-0 flex-1">
        <div className="text-foreground">{c.text}</div>
        <div className="text-muted-foreground">
          {c.checked_by === "model" && (
            <span className="mr-1 rounded bg-muted px-1 py-px text-[10px]">checked by AI</span>
          )}
          {c.reason}
        </div>
        {(c.evidence || c.cites.length > 0) && (
          <details className="mt-0.5">
            <summary className="cursor-pointer text-[10px] text-muted-foreground hover:text-foreground">
              {c.cites.length ? c.cites.map((t) => `[${t}]`).join(" ") : "no citation"}
            </summary>
            {c.evidence ? (
              <pre className="mt-0.5 whitespace-pre-wrap break-words rounded bg-muted px-1.5 py-1 font-sans text-[10px] text-muted-foreground">
                {c.evidence}
              </pre>
            ) : (
              <div className="mt-0.5 text-[10px] italic text-muted-foreground">
                No matching fact was found.
              </div>
            )}
          </details>
        )}
      </div>
    </li>
  );
}

/** Under each co-pilot answer: every sentence that states a number, a gene or cites
 *  a fact, checked against the fact it cites. Rule checks are free and automatic;
 *  "Check claims" sends only the uncertain ones to one small model call. */
export function ClaimsLedger({
  claims,
  canCheck,
  checking = false,
  error,
  onCheck,
}: {
  claims: Claim[];
  canCheck: boolean;
  checking?: boolean;
  error?: string;
  onCheck?: () => void;
}) {
  const [open, setOpen] = useState(false);
  if (!claims.length) return null;
  const t = tally(claims);
  const pending = toCheck(claims).length;
  const refuted = claims.filter((c) => c.verdict === "refuted");
  const counts = (["supported", "refuted", "uncertain"] as const).filter((k) => t[k] > 0);

  return (
    <div className="mt-2 rounded-md border bg-muted/30 px-2 py-1.5 text-xs">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          title="Each sentence with a number, a gene or a citation, checked against the fact it cites"
          className="flex items-center gap-1 text-muted-foreground hover:text-foreground"
        >
          {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          <ShieldCheck className="h-3.5 w-3.5" />
          <span className="font-medium">Claims</span>
        </button>
        <span className="text-muted-foreground">
          {counts.map((k, i) => (
            <span key={k}>
              {i > 0 && " · "}
              <span className={VERDICT[k].cls}>
                {t[k]} {VERDICT[k].label}
              </span>
            </span>
          ))}
        </span>
        <span className="flex-1" />
        {canCheck && pending > 0 && (
          <button
            type="button"
            onClick={onCheck}
            disabled={checking}
            title={`Ask the language model to judge the ${pending} uncertain claim${
              pending === 1 ? "" : "s"
            } against the facts they cite (one small call)`}
            className="flex items-center gap-1 rounded border px-1.5 py-0.5 text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-50"
          >
            {checking ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <ShieldCheck className="h-3 w-3" />
            )}
            {checking ? "Checking…" : "Check claims"}
          </button>
        )}
      </div>
      {!open && refuted.length > 0 && (
        <ul className="mt-1.5 space-y-1.5">
          {refuted.map((c) => (
            <ClaimRow key={c.id} c={c} />
          ))}
        </ul>
      )}
      {open && (
        <ul className="mt-1.5 space-y-1.5 border-l-2 border-muted pl-2">
          {claims.map((c) => (
            <ClaimRow key={c.id} c={c} />
          ))}
        </ul>
      )}
      {error && <div className="mt-1 text-red-600">{error}</div>}
    </div>
  );
}
