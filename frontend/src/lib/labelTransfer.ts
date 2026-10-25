/** Helpers for the reference mapping card (backend core/label_transfer.py). */
import type { LabelTransferCluster, LabelTransferMethod } from "@/api/labelTransfer";

export const METHOD_LABELS: Record<LabelTransferMethod, string> = {
  projection: "Project onto reference",
  integration: "Integrate first (Harmony)",
  celltypist: "Train CellTypist on reference",
};

/** A cluster's unmapped share as a level: most unmapped = candidate, >= 10% (ten times chance) = elevated. */
export function unmappedLevel(c: Pick<LabelTransferCluster, "unmapped_share" | "candidate_absent">): "candidate" | "elevated" | "normal" {
  if (c.candidate_absent) return "candidate";
  return c.unmapped_share >= 0.1 ? "elevated" : "normal";
}

/** "about 1.2 GB" from bytes. */
export function aboutGb(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  return gb >= 10 ? `about ${gb.toFixed(0)} GB` : `about ${gb.toFixed(1)} GB`;
}

/** The library datasets usable as a reference for `currentId`: ready, not itself. */
export function referenceChoices<T extends { id: string; status: string }>(all: T[], currentId: string): T[] {
  return all.filter((d) => d.id !== currentId && d.status === "ready");
}
