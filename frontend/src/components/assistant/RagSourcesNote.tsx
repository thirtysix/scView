import { useEffect, useState } from "react";
import { BookOpen } from "lucide-react";
import { getRagStatus, type RagCorpus, type RagStatus } from "@/api/assistant";

// The drawer and the AI Co-pilot panel both show this note: one request serves both.
let pending: Promise<RagStatus> | null = null;
function loadStatus(): Promise<RagStatus> {
  pending ??= getRagStatus().catch((e) => {
    pending = null; // try again next time
    throw e;
  });
  return pending;
}
/** Tests only: forget the shared request. */
export function resetRagStatus() {
  pending = null;
}

const plural = (n: number, one: string, many: string) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

function day(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? "" : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

/** The one-line summary and its tooltip, from a status response. */
export function describeSources(s: RagStatus): { line: string; detail: string } {
  if (!s.enabled) {
    return {
      line: "Your dataset and the app only; literature and methods search is not set up.",
      detail:
        s.reason === "no_api_key"
          ? "A literature database is configured but DEEPINFRA_API_KEY is not, so retrieval is off."
          : "No RAG database is configured (RAG_DATABASE_URL), so answers draw on your dataset and scView itself.",
    };
  }
  if (s.error) {
    return {
      line: "Literature index unreachable; answers use your dataset and the app.",
      detail: "The RAG database could not be reached just now.",
    };
  }
  const lit = s.corpora.literature;
  const tut = s.corpora.tutorials;
  const parts = [
    lit ? plural(lit.documents, "paper", "papers") : "no papers",
    tut ? plural(tut.documents, "tutorial doc", "tutorial docs") : "no tutorial docs",
  ];
  const latest = [lit?.last_added, tut?.last_added].filter(Boolean).sort().pop();
  const per = (name: string, c: RagCorpus | undefined) =>
    c ? `${name}: ${c.documents.toLocaleString()} documents in ${c.chunks.toLocaleString()} chunks` +
        (c.last_added ? `, newest added ${day(c.last_added)}` : "") : `${name}: empty`;
  return {
    line: `Your dataset + ${parts.join(" + ")} indexed` + (latest ? ` · updated ${day(latest)}` : ""),
    detail: `Methods and biology questions retrieve from these. ${per("Literature", lit)}. ${per("Tutorials", tut)}.`,
  };
}

/** A small "Sources" line for the co-pilot: what retrieval can draw on. The
 *  per-corpus breakdown is in the tooltip (a popover would clip at the drawer top). */
export function RagSourcesNote() {
  const [status, setStatus] = useState<RagStatus | null>(null);

  useEffect(() => {
    let live = true;
    loadStatus()
      .then((s) => live && setStatus(s))
      .catch(() => live && setStatus(null));
    return () => {
      live = false;
    };
  }, []);

  if (!status) return null;
  const { line, detail } = describeSources(status);
  return (
    <p title={detail} className="flex items-center gap-1 text-[11px] text-muted-foreground">
      <BookOpen className="h-3 w-3 flex-shrink-0" />
      <span className="font-medium">Sources:</span>
      <span className="truncate">{line}</span>
    </p>
  );
}
