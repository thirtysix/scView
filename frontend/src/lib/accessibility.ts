/**
 * Accessibility mode: one switch (sidebar, under Theme) that trades the default
 * look, tuned for clarity and aesthetics, for settings that meet WCAG AA.
 *
 * On, `.a11y` on <html> (src/index.css) darkens faint text and tinted text to
 * at least 4.5:1, raises 10-11 px text to 12 px, draws a strong focus outline,
 * underlines links and stops transitions/animations. Plotly plots get a 12 px
 * font floor (`a11yLayout`, applied in components/plots/Plot). Turning it on
 * also picks the colour-blind-safe categorical palette (the default anyway).
 * Persisted in localStorage; off by default.
 */
import type { Layout } from "plotly.js";
import { create } from "zustand";
import { useSettingsStore } from "@/stores/settingsStore";

export const A11Y_STORAGE_KEY = "scview.a11y.v1";

/** Plotly text below this size (px) is raised to it while the mode is on. */
export const A11Y_MIN_FONT = 12;

export function readStoredA11y(): boolean {
  try {
    return window.localStorage.getItem(A11Y_STORAGE_KEY) === "on";
  } catch {
    return false;
  }
}

function storeA11y(on: boolean): void {
  try {
    window.localStorage.setItem(A11Y_STORAGE_KEY, on ? "on" : "off");
  } catch {
    // private window / blocked storage: the choice lasts for this page only
  }
}

interface A11yState {
  enabled: boolean;
  setEnabled: (on: boolean) => void;
}

export const useA11yStore = create<A11yState>((set) => ({
  enabled: readStoredA11y(),
  setEnabled: (enabled) => {
    storeA11y(enabled);
    set({ enabled });
    if (enabled) useSettingsStore.getState().setCategoricalPalette("cvd");
  },
}));

function applyA11y(on: boolean): void {
  if (typeof document === "undefined") return;
  document.documentElement.classList.toggle("a11y", on);
}

/** Put `.a11y` on <html> now and whenever the switch changes. */
export function initA11y(): () => void {
  applyA11y(useA11yStore.getState().enabled);
  return useA11yStore.subscribe((s) => applyA11y(s.enabled));
}

export function useA11y(): boolean {
  return useA11yStore((s) => s.enabled);
}

type FontLike = { size?: number } | undefined;

function floorFont<T extends FontLike>(font: T, fallback: boolean): T | { size: number } {
  const size = font?.size;
  if (size === undefined) return fallback ? { ...(font ?? {}), size: A11Y_MIN_FONT } : (font as T);
  return size < A11Y_MIN_FONT ? { ...font, size: A11Y_MIN_FONT } : (font as T);
}

const AXIS_KEY = /^[xy]axis\d*$/;

/**
 * The layout with every font at least A11Y_MIN_FONT px: the global font,
 * titles, axis ticks and titles, legend, hover labels, colour bars and
 * annotations. Off, the layout is returned unchanged (same object).
 */
export function a11yLayout(layout: Partial<Layout> | undefined, on: boolean): Partial<Layout> | undefined {
  if (!on || !layout) return layout;
  const out: Record<string, unknown> = { ...layout };
  const l = layout as Record<string, unknown>;
  out.font = floorFont(l.font as FontLike, true);
  for (const [key, value] of Object.entries(l)) {
    if (!AXIS_KEY.test(key) || !value || typeof value !== "object") continue;
    const axis = { ...(value as Record<string, unknown>) };
    if (axis.tickfont) axis.tickfont = floorFont(axis.tickfont as FontLike, false);
    if (axis.title && typeof axis.title === "object") {
      const t = axis.title as Record<string, unknown>;
      if (t.font) axis.title = { ...t, font: floorFont(t.font as FontLike, false) };
    }
    out[key] = axis;
  }
  if (l.title && typeof l.title === "object") {
    const t = l.title as Record<string, unknown>;
    if (t.font) out.title = { ...t, font: floorFont(t.font as FontLike, false) };
  }
  if (l.legend && typeof l.legend === "object") {
    const lg = l.legend as Record<string, unknown>;
    if (lg.font) out.legend = { ...lg, font: floorFont(lg.font as FontLike, false) };
  }
  if (l.hoverlabel && typeof l.hoverlabel === "object") {
    const h = l.hoverlabel as Record<string, unknown>;
    if (h.font) out.hoverlabel = { ...h, font: floorFont(h.font as FontLike, false) };
  }
  if (Array.isArray(l.annotations)) {
    out.annotations = (l.annotations as Record<string, unknown>[]).map((a) =>
      a.font ? { ...a, font: floorFont(a.font as FontLike, false) } : a,
    );
  }
  return out as Partial<Layout>;
}
