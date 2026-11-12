import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Info } from "lucide-react";

/** Focus from the keyboard, not a dialog moving focus to its first control after a
 *  click (that would pop the tip open, often off the top of the screen). */
function keyboardFocus(el: Element): boolean {
  try {
    return el.matches(":focus-visible");
  } catch {
    return true; // no :focus-visible support: keep the tip reachable
  }
}

const GAP = 6; // between the icon and the tip
const MARGIN = 8; // between the tip and the window edge

/**
 * Where the tip goes: on the preferred side when it fits, else the other side; along the
 * icon's left (or right) edge, kept inside the window. Fixed coordinates, so no scrolling
 * box or card with overflow hidden can cut it off.
 */
export function placeTip(
  icon: { top: number; bottom: number; left: number; right: number },
  tip: { width: number; height: number },
  view: { width: number; height: number },
  side: "above" | "below",
  align: "left" | "right",
): { top: number; left: number; side: "above" | "below" } {
  const roomAbove = icon.top - GAP - MARGIN >= tip.height;
  const roomBelow = view.height - icon.bottom - GAP - MARGIN >= tip.height;
  const placed = side === "above" ? (roomAbove || !roomBelow ? "above" : "below") : roomBelow || !roomAbove ? "below" : "above";
  const top = placed === "above" ? icon.top - GAP - tip.height : icon.bottom + GAP;
  const want = align === "right" ? icon.right - tip.width : icon.left;
  const left = Math.max(MARGIN, Math.min(want, view.width - tip.width - MARGIN));
  return { top: Math.max(MARGIN, top), left, side: placed };
}

/**
 * The tip itself, drawn over everything next to `anchor` (fixed, in a portal), on the preferred
 * side when it fits. Closes (via `onClose`) when anything scrolls, since a fixed tip would stay
 * behind while its anchor moves. Shared by `InfoTip` and the tab hints (`TabList`).
 */
export function TipPopup({
  anchor,
  id,
  side = "above",
  align = "left",
  width = 300,
  onClose,
  children,
}: {
  anchor: HTMLElement | null;
  id?: string;
  side?: "above" | "below";
  align?: "left" | "right";
  width?: number;
  onClose: () => void;
  children: ReactNode;
}) {
  const [pos, setPos] = useState<{ top: number; left: number; side: "above" | "below" } | null>(null);
  const tipRef = useRef<HTMLSpanElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useLayoutEffect(() => {
    const icon = anchor?.getBoundingClientRect();
    const tip = tipRef.current?.getBoundingClientRect();
    if (!icon || !tip) return;
    setPos(placeTip(icon, { width: tip.width, height: tip.height }, { width: window.innerWidth, height: window.innerHeight }, side, align));
  }, [anchor, side, align]);

  useEffect(() => {
    const close = () => closeRef.current();
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, []);

  const shown = Math.min(width, window.innerWidth - 2 * MARGIN);
  return createPortal(
    <span
      ref={tipRef}
      id={id}
      role="tooltip"
      data-side={pos?.side ?? side}
      style={{
        width: shown,
        top: pos?.top ?? 0,
        left: pos?.left ?? 0,
        // measured first, shown once placed (no flash at the corner)
        visibility: pos ? "visible" : "hidden",
      }}
      className="pointer-events-none fixed z-[1000] block rounded-md border border-slate-200 bg-white px-2.5 py-2 text-left text-xs font-normal normal-case leading-relaxed tracking-normal text-slate-700 shadow-lg"
    >
      {children}
    </span>,
    document.body,
  );
}

/**
 * Small explanatory popover on an info icon.
 *
 * Native `title=` is used elsewhere in the app for short hints, but it is slow to
 * appear, unstyled, truncates, and never reaches keyboard users. Anything that has
 * to actually teach the reader something belongs here instead.
 */
export function InfoTip({
  children,
  align = "left",
  width = 300,
  label = "More information",
  side = "above",
}: {
  children: ReactNode;
  align?: "left" | "right";
  width?: number;
  label?: string;
  /** The side to open on when both fit; the tip flips when its side has no room. */
  side?: "above" | "below";
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const iconRef = useRef<HTMLButtonElement>(null);

  return (
    <span className="relative inline-flex">
      <button
        ref={iconRef}
        type="button"
        data-infotip=""
        aria-label={label}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={(e) => setOpen(keyboardFocus(e.currentTarget))}
        onBlur={() => setOpen(false)}
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        className="inline-flex items-center text-slate-300 transition-colors hover:text-slate-500 focus:text-slate-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 rounded-sm"
      >
        <Info className="h-3 w-3" />
      </button>
      {open && (
        <TipPopup anchor={iconRef.current} id={id} side={side} align={align} width={width} onClose={() => setOpen(false)}>
          {children}
        </TipPopup>
      )}
    </span>
  );
}

const HOVER_DELAY_MS = 300;

/**
 * A tip on the thing itself: hovering `children` (a heading, a label, a column header) for a
 * moment opens `tip`, as does keyboard focus on a control inside it. No icon and no marker:
 * the reader finds tips by pointing at what they want explained.
 */
export function HoverTip({
  tip,
  children,
  side = "above",
  align = "left",
  width = 300,
  className = "",
}: {
  tip: ReactNode;
  children: ReactNode;
  side?: "above" | "below";
  align?: "left" | "right";
  width?: number;
  /** Classes of the wrapper (an inline span): e.g. "flex items-center gap-1" when it wraps a row. */
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const timer = useRef<number | undefined>(undefined);
  const id = useId();
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const close = () => {
    window.clearTimeout(timer.current);
    setOpen(false);
  };
  return (
    <span
      ref={ref}
      className={className}
      onMouseEnter={() => {
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setOpen(true), HOVER_DELAY_MS);
      }}
      onMouseLeave={close}
      onFocusCapture={(e) => {
        if (e.target instanceof Element && keyboardFocus(e.target)) setOpen(true);
      }}
      onBlurCapture={close}
      onMouseDown={close}
    >
      {children}
      {open && (
        <TipPopup anchor={ref.current} id={id} side={side} align={align} width={width} onClose={close}>
          {tip}
        </TipPopup>
      )}
    </span>
  );
}
