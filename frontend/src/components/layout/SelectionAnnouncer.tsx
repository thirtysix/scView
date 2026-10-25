import { useEffect, useRef, useState } from "react";
import { useSelectionStore } from "@/stores/selectionStore";

/** What a screen reader hears when the cell selection changes. */
export function selectionAnnouncement(n: number | null, prev: number | null): string {
  if (n !== null && n > 0) return `${n.toLocaleString("en-US")} ${n === 1 ? "cell" : "cells"} selected`;
  return prev !== null && prev > 0 ? "Selection cleared" : "";
}

/**
 * A visually hidden live region that announces every change of the cell
 * selection (lasso, legend, select by group, keyboard rectangle, filters),
 * from the one selection store. Mounted once in AppLayout.
 */
export function SelectionAnnouncer() {
  const selection = useSelectionStore((s) => s.selectedCellIndices);
  const [message, setMessage] = useState("");
  const prev = useRef(selection);
  useEffect(() => {
    if (prev.current === selection) return;
    setMessage(selectionAnnouncement(selection?.size ?? null, prev.current?.size ?? null));
    prev.current = selection;
  }, [selection]);
  return (
    <div role="status" aria-live="polite" className="sr-only" data-testid="selection-announcer">
      {message}
    </div>
  );
}
