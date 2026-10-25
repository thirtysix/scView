/**
 * Reduced motion: the system's "reduce motion" setting
 * (prefers-reduced-motion), or accessibility mode, which also stops
 * transitions. Plots read it to drop their own animation: deck.gl's smooth
 * scroll zoom and Plotly's layout transitions. CSS transitions are handled in
 * index.css.
 */
import { useSyncExternalStore } from "react";
import { useA11y } from "@/lib/accessibility";

const QUERY = "(prefers-reduced-motion: reduce)";

function mediaQuery(): MediaQueryList | null {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return null;
  return window.matchMedia(QUERY);
}

export function prefersReducedMotion(): boolean {
  return mediaQuery()?.matches ?? false;
}

function subscribe(onChange: () => void): () => void {
  const mq = mediaQuery();
  if (!mq) return () => {};
  mq.addEventListener?.("change", onChange);
  return () => mq.removeEventListener?.("change", onChange);
}

/** True while the system asks for reduced motion or accessibility mode is on. */
export function useReducedMotion(): boolean {
  const system = useSyncExternalStore(subscribe, prefersReducedMotion, () => false);
  const a11y = useA11y();
  return system || a11y;
}
