import { useEffect, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Maximize2, Minimize2 } from "lucide-react";

interface PanelProps {
  title?: ReactNode;
  /** Extra header controls, shown left of the fullscreen button. */
  actions?: ReactNode;
  /** Show the fullscreen toggle (default true). */
  expandable?: boolean;
  /** Classes for the outer card. */
  className?: string;
  /** Classes for the body wrapper (default "p-2"). */
  bodyClassName?: string;
  children: ReactNode;
}

/**
 * Card with an optional header and a fullscreen-expand toggle, mirroring the
 * wnt-hub `.panel` / `.panel.fullscreen` pattern. Going fullscreen moves the
 * SAME node into a body-level portal (fixed inset-0) — it is not remounted, so
 * deck.gl / Plotly children keep their WebGL context and state. Escape exits.
 *
 * Children should fill their container (e.g. width/height 100% + Plotly
 * autosize) so the plot grows to fill the screen when expanded.
 */
export function Panel({
  title,
  actions,
  expandable = true,
  className,
  bodyClassName,
  children,
}: PanelProps) {
  const [full, setFull] = useState(false);

  useEffect(() => {
    if (!full) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFull(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [full]);

  const hasHeader = title != null || actions != null || expandable;

  const card = (
    <div
      className={`flex h-full min-h-0 flex-col overflow-hidden border border-slate-200 bg-white shadow-sm ${
        full ? "fixed inset-0 z-[1050] rounded-none" : "rounded-xl"
      } ${className ?? ""}`}
    >
      {hasHeader && (
        <div className="flex flex-shrink-0 items-center justify-between gap-3 border-b border-slate-100 bg-slate-50 px-3 py-1.5">
          <div className="min-w-0 truncate text-xs font-semibold text-slate-600">{title}</div>
          <div className="flex flex-shrink-0 items-center gap-1.5">
            {actions}
            {expandable && (
              <button
                type="button"
                onClick={() => setFull((f) => !f)}
                title={full ? "Exit full screen (Esc)" : "Full screen"}
                aria-label={full ? "Exit full screen" : "Full screen"}
                className="rounded p-1 text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-600"
              >
                {full ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
              </button>
            )}
          </div>
        </div>
      )}
      <div className={`min-h-0 flex-1 ${full ? "overflow-auto" : ""} ${bodyClassName ?? "p-2"}`}>
        {children}
      </div>
    </div>
  );

  return full ? createPortal(card, document.body) : card;
}
