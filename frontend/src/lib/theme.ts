/**
 * App colour theme: Light / Dark / System (default System, follows
 * `prefers-color-scheme`), persisted in localStorage.
 *
 * The chrome flips through CSS: `.dark` on <html> redefines the slate ramp and
 * the tinted 50-300 / 600-950 shades (src/index.css), so component classes stay
 * as they are. Plotly plots flip through `themedLayout` / `themedData`, applied
 * once in the shared `components/plots/Plot` wrapper, so every plot (and every
 * plot added later) picks the theme up. In light mode both return their input
 * unchanged. Downloads stay on white (`exportFigure`, used by plotConfig).
 */
import { useEffect } from "react";
import type { Data, Layout, PlotlyHTMLElement } from "plotly.js";
import { create } from "zustand";
import { useSettingsStore } from "@/stores/settingsStore";

export type ThemePreference = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "scview.theme.v1";
const PREFERENCES: readonly ThemePreference[] = ["light", "dark", "system"];

export function readStoredPreference(): ThemePreference {
  try {
    const v = window.localStorage.getItem(THEME_STORAGE_KEY);
    return PREFERENCES.includes(v as ThemePreference) ? (v as ThemePreference) : "system";
  } catch {
    return "system";
  }
}

function storePreference(p: ThemePreference): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, p);
  } catch {
    // private window / blocked storage: the choice lasts for this page only
  }
}

const DARK_QUERY = "(prefers-color-scheme: dark)";

export function systemPrefersDark(): boolean {
  try {
    return typeof window.matchMedia === "function" && window.matchMedia(DARK_QUERY).matches;
  } catch {
    return false;
  }
}

export function resolveTheme(preference: ThemePreference, systemDark: boolean): ResolvedTheme {
  if (preference === "system") return systemDark ? "dark" : "light";
  return preference;
}

interface ThemeState {
  preference: ThemePreference;
  systemDark: boolean;
  setPreference: (p: ThemePreference) => void;
  setSystemDark: (dark: boolean) => void;
}

export const useThemeStore = create<ThemeState>((set) => ({
  preference: readStoredPreference(),
  systemDark: systemPrefersDark(),
  setPreference: (preference) => {
    storePreference(preference);
    set({ preference });
    // Picking a theme is also picking the scatter background: it follows again.
    useSettingsStore.setState({ plotBackgroundChosen: false });
  },
  setSystemDark: (systemDark) => set({ systemDark }),
}));

/** The theme in effect now (outside React, e.g. at download time). */
export function currentTheme(): ResolvedTheme {
  const s = useThemeStore.getState();
  return resolveTheme(s.preference, s.systemDark);
}

export function useTheme(): {
  preference: ThemePreference;
  resolvedTheme: ResolvedTheme;
  setPreference: (p: ThemePreference) => void;
} {
  const preference = useThemeStore((s) => s.preference);
  const systemDark = useThemeStore((s) => s.systemDark);
  const setPreference = useThemeStore((s) => s.setPreference);
  return { preference, resolvedTheme: resolveTheme(preference, systemDark), setPreference };
}

export function useResolvedTheme(): ResolvedTheme {
  return useThemeStore((s) => resolveTheme(s.preference, s.systemDark));
}

export function applyTheme(theme: ResolvedTheme, root: HTMLElement = document.documentElement): void {
  root.classList.toggle("dark", theme === "dark");
  root.style.colorScheme = theme;
}

/** Keeps <html> in step with the theme and the OS setting. Call once at startup. */
export function initTheme(): () => void {
  applyTheme(currentTheme());
  const unsubscribe = useThemeStore.subscribe(() => applyTheme(currentTheme()));
  let mql: MediaQueryList | null = null;
  const onChange = (e: MediaQueryListEvent) => useThemeStore.getState().setSystemDark(e.matches);
  try {
    if (typeof window.matchMedia === "function") {
      mql = window.matchMedia(DARK_QUERY);
      mql.addEventListener?.("change", onChange);
    }
  } catch {
    mql = null;
  }
  return () => {
    unsubscribe();
    mql?.removeEventListener?.("change", onChange);
  };
}

/** Hook form of initTheme for the app root. */
export function useThemeSync(): void {
  useEffect(() => initTheme(), []);
}

// ---------------------------------------------------------------------------
// Scatter background (deck.gl)
// ---------------------------------------------------------------------------

/** The theme decides, unless the user picked Light/Dark in the plot toolbar
 *  since the theme was last changed. */
export function scatterBackground(
  chosen: boolean,
  setting: "white" | "dark",
  theme: ResolvedTheme,
): "white" | "dark" {
  if (chosen) return setting;
  return theme === "dark" ? "dark" : "white";
}

export function useScatterBackground(): "white" | "dark" {
  const theme = useResolvedTheme();
  const chosen = useSettingsStore((s) => s.plotBackgroundChosen);
  const setting = useSettingsStore((s) => s.plotBackground);
  return scatterBackground(chosen, setting, theme);
}

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

/** Dark-theme values of white and the slate ramp; src/index.css uses the same. */
export const DARK_SURFACE = "#1a2436";
export const DARK_PAGE = "#111a2b";
export const DARK_SLATE: Record<number, string> = {
  50: "#131c2e",
  100: "#222d42",
  200: "#2f3b52",
  300: "#3e4b63",
  400: "#8a98ad",
  500: "#9aa8bc",
  600: "#b4c0d0",
  700: "#cbd5e1",
  800: "#e2e8f0",
  900: "#f1f5f9",
  950: "#f8fafc",
};
const LIGHT_SLATE: Record<number, string> = {
  50: "#f8fafc",
  100: "#f1f5f9",
  200: "#e2e8f0",
  300: "#cbd5e1",
  400: "#94a3b8",
  500: "#64748b",
  600: "#475569",
  700: "#334155",
  800: "#1e293b",
  900: "#0f172a",
  950: "#020617",
};

/** Light-theme neutrals as plots write them -> their dark-theme counterpart. */
const NEUTRALS: Map<string, string> = (() => {
  const m = new Map<string, string>();
  for (const [shade, hex] of Object.entries(LIGHT_SLATE)) m.set(hex, DARK_SLATE[Number(shade)]!);
  for (const w of ["#fff", "#ffffff", "white"]) m.set(w, DARK_SURFACE);
  for (const k of ["#000", "#000000", "black", "#111", "#222", "#333", "#444", "#555"]) m.set(k, DARK_SLATE[800]!);
  for (const k of ["#666", "#777", "#888", "gray", "grey"]) m.set(k, DARK_SLATE[500]!);
  for (const k of ["#999", "#aaa", "#bbb"]) m.set(k, DARK_SLATE[400]!);
  for (const k of ["#ccc", "#ddd", "#eee", "lightgray", "lightgrey"]) m.set(k, DARK_SLATE[200]!);
  return m;
})();

const WHITE_RGBA = /^rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*([0-9.]+)\s*\)$/;

/** The dark-theme counterpart of a light-theme neutral; other colours (and
 *  non-strings) come back unchanged. */
export function darkNeutral(color: unknown): unknown {
  if (typeof color !== "string") return color;
  const c = color.trim().toLowerCase();
  const hit = NEUTRALS.get(c);
  if (hit) return hit;
  const w = WHITE_RGBA.exec(c);
  if (w) return `rgba(26, 36, 54, ${w[1]})`;
  return color;
}

function isTransparent(color: unknown): boolean {
  if (typeof color !== "string") return false;
  const c = color.replace(/\s+/g, "").toLowerCase();
  return c === "transparent" || /^rgba\(.*,0(\.0+)?\)$/.test(c);
}

export interface PlotTheme {
  paper: string;
  plot: string;
  text: string;
  muted: string;
  grid: string;
  axisLine: string;
  zeroLine: string;
  legendBg: string;
  hoverBg: string;
  hoverText: string;
  hoverBorder: string;
}

const LIGHT_PLOT: PlotTheme = {
  paper: "#ffffff",
  plot: "#ffffff",
  text: LIGHT_SLATE[700]!,
  muted: LIGHT_SLATE[500]!,
  grid: LIGHT_SLATE[100]!,
  axisLine: LIGHT_SLATE[300]!,
  zeroLine: LIGHT_SLATE[200]!,
  legendBg: "rgba(255, 255, 255, 0.85)",
  hoverBg: "#ffffff",
  hoverText: LIGHT_SLATE[800]!,
  hoverBorder: LIGHT_SLATE[300]!,
};

const DARK_PLOT: PlotTheme = {
  paper: DARK_SURFACE,
  plot: DARK_SURFACE,
  text: DARK_SLATE[700]!,
  muted: DARK_SLATE[500]!,
  grid: DARK_SLATE[100]!,
  axisLine: "#56637a",
  zeroLine: DARK_SLATE[300]!,
  legendBg: "rgba(26, 36, 54, 0.85)",
  hoverBg: "#0f172a",
  hoverText: DARK_SLATE[900]!,
  hoverBorder: "#56637a",
};

/** Plotly layout colours for a theme (paper/plot bg, font, grid, axes, legend,
 *  hover label; colorbars take the layout font). */
export function plotTheme(theme: ResolvedTheme): PlotTheme {
  return theme === "dark" ? DARK_PLOT : LIGHT_PLOT;
}

// ---------------------------------------------------------------------------
// Theming a Plotly figure: a list of path -> value overrides
// ---------------------------------------------------------------------------

type Path = (string | number)[];
type Override = [Path, unknown];
type Obj = Record<string | number, unknown>;

function isObj(v: unknown): v is Obj {
  return typeof v === "object" && v !== null && !ArrayBuffer.isView(v);
}

function getIn(obj: unknown, path: Path): unknown {
  let cur: unknown = obj;
  for (const k of path) {
    if (!isObj(cur)) return undefined;
    cur = cur[k];
  }
  return cur;
}

/** Copy-on-write set (undefined deletes); `obj` is never mutated. */
function setIn<T>(obj: T, path: Path, value: unknown): T {
  const [k, ...rest] = path;
  const base: Obj = isObj(obj) ? obj : {};
  const copy: Obj = (Array.isArray(base) ? [...base] : { ...base }) as Obj;
  if (rest.length === 0) {
    if (value === undefined) delete copy[k!];
    else copy[k!] = value;
  } else {
    copy[k!] = setIn(base[k!], rest, value);
  }
  return copy as T;
}

function applyOverrides<T>(obj: T, overrides: Override[]): T {
  let out = obj;
  for (const [path, value] of overrides) out = setIn(out, path, value);
  return out;
}

function neutralAt(src: unknown, path: Path, out: Override[]): void {
  const v = getIn(src, path);
  const mapped = darkNeutral(v);
  if (mapped !== v) out.push([path, mapped]);
}

function bgAt(src: unknown, path: Path, value: string, out: Override[]): void {
  if (!isTransparent(getIn(src, path))) out.push([path, value]);
}

function fontAt(src: unknown, path: Path, out: Override[]): void {
  neutralAt(src, [...path, "color"], out);
}

function layoutOverrides(layout: Partial<Layout>, t: PlotTheme): Override[] {
  const out: Override[] = [];
  const l = layout as Obj;
  bgAt(l, ["paper_bgcolor"], t.paper, out);
  bgAt(l, ["plot_bgcolor"], t.plot, out);
  const font = getIn(l, ["font", "color"]);
  out.push([["font", "color"], font === undefined ? t.text : darkNeutral(font)]);
  fontAt(l, ["title", "font"], out);
  const axes = new Set(["xaxis", "yaxis"]);
  for (const k of Object.keys(l)) if (/^[xy]axis\d*$/.test(k)) axes.add(k);
  for (const ax of axes) {
    out.push([[ax, "gridcolor"], t.grid], [[ax, "linecolor"], t.axisLine], [[ax, "zerolinecolor"], t.zeroLine]);
    if (getIn(l, [ax, "tickcolor"]) !== undefined) out.push([[ax, "tickcolor"], t.axisLine]);
    fontAt(l, [ax, "tickfont"], out);
    fontAt(l, [ax, "title", "font"], out);
  }
  if (isObj(l.legend)) {
    neutralAt(l, ["legend", "bgcolor"], out);
    neutralAt(l, ["legend", "bordercolor"], out);
    fontAt(l, ["legend", "font"], out);
    fontAt(l, ["legend", "title", "font"], out);
  }
  if (isObj(l.hoverlabel)) {
    neutralAt(l, ["hoverlabel", "bgcolor"], out);
    neutralAt(l, ["hoverlabel", "bordercolor"], out);
    fontAt(l, ["hoverlabel", "font"], out);
  }
  if (isObj(l.coloraxis)) {
    fontAt(l, ["coloraxis", "colorbar", "tickfont"], out);
    fontAt(l, ["coloraxis", "colorbar", "title", "font"], out);
  }
  if (Array.isArray(l.annotations)) {
    l.annotations.forEach((_, i) => {
      fontAt(l, ["annotations", i, "font"], out);
      neutralAt(l, ["annotations", i, "bgcolor"], out);
      neutralAt(l, ["annotations", i, "bordercolor"], out);
      neutralAt(l, ["annotations", i, "arrowcolor"], out);
    });
  }
  if (Array.isArray(l.shapes)) {
    l.shapes.forEach((_, i) => {
      neutralAt(l, ["shapes", i, "line", "color"], out);
      neutralAt(l, ["shapes", i, "fillcolor"], out);
      fontAt(l, ["shapes", i, "label", "font"], out);
    });
  }
  return out;
}

/** Relative luminance of a #rrggbb / #rgb colour (null for anything else). */
function luminance(color: string): number | null {
  let h = color.trim().toLowerCase();
  if (/^#[0-9a-f]{3}$/.test(h)) h = `#${h[1]}${h[1]}${h[2]}${h[2]}${h[3]}${h[3]}`;
  if (!/^#[0-9a-f]{6}$/.test(h)) return null;
  const ch = [1, 3, 5].map((i) => {
    const c = parseInt(h.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * ch[0]! + 0.7152 * ch[1]! + 0.0722 * ch[2]!;
}

const TRACE_FONTS: Path[] = [
  ["textfont"],
  ["insidetextfont"],
  ["outsidetextfont"],
  ["colorbar", "tickfont"],
  ["colorbar", "title", "font"],
  ["marker", "colorbar", "tickfont"],
  ["marker", "colorbar", "title", "font"],
];
// Outlines and lines in a neutral (box/violin edges, white marker rims).
const TRACE_LINES: Path[] = [["line", "color"], ["marker", "line", "color"], ["box", "line", "color"], ["meanline", "color"], ["colorbar", "outlinecolor"]];

function dataOverrides(data: Data[]): Override[] {
  const out: Override[] = [];
  data.forEach((trace, i) => {
    if (!isObj(trace)) return;
    for (const p of TRACE_FONTS) fontAt(data, [i, ...p], out);
    for (const p of TRACE_LINES) neutralAt(data, [i, ...p], out);
    // Only DARK neutral markers flip (they would vanish on dark); light-grey
    // "other" points stay as they are.
    const mc = getIn(trace, ["marker", "color"]);
    if (typeof mc === "string") {
      const lum = luminance(mc);
      if (lum !== null && lum < 0.1 && darkNeutral(mc) !== mc) out.push([[i, "marker", "color"], darkNeutral(mc)]);
    }
  });
  return out;
}

/** The layout for the theme: unchanged in light mode. */
export function themedLayout<L extends Partial<Layout>>(layout: L, theme: ResolvedTheme): L {
  if (theme === "light") return layout;
  return applyOverrides(layout ?? ({} as L), layoutOverrides(layout ?? {}, plotTheme(theme)));
}

/** The traces for the theme (text, colorbar and outline neutrals): unchanged in light mode. */
export function themedData(data: Data[], theme: ResolvedTheme): Data[] {
  if (theme === "light" || !Array.isArray(data)) return data;
  return applyOverrides(data, dataOverrides(data));
}

/** Put back, in `current`, the light values of every path theming changed in `base`. */
function restoreLight<T>(current: T, base: T, overrides: Override[]): T {
  let out = current;
  for (const [path] of overrides) out = setIn(out, path, getIn(base, path));
  return out;
}

// The figures each themed plot was handed, by graph div (for light exports).
const baseFigures = new WeakMap<HTMLElement, { data: Data[]; layout: Partial<Layout> }>();

export function rememberBaseFigure(gd: HTMLElement, data: Data[], layout: Partial<Layout>): void {
  baseFigures.set(gd, { data, layout });
}

type GraphDiv = PlotlyHTMLElement & {
  _fullLayout?: { width?: number; height?: number };
  _context?: unknown;
};

/**
 * What a download renders: the plot itself in light mode; in dark mode a copy
 * of its current state (zoom, hidden traces) with the light colours put back,
 * at the on-screen size, so figures stay on white for journals.
 */
export function exportFigure(
  gd: PlotlyHTMLElement,
  theme: ResolvedTheme = currentTheme(),
): PlotlyHTMLElement | { data: Data[]; layout: Partial<Layout>; config?: unknown } {
  const base = baseFigures.get(gd);
  if (theme === "light" || !base) return gd;
  const g = gd as GraphDiv;
  const t = plotTheme(theme);
  let layout = restoreLight(g.layout as Partial<Layout>, base.layout, layoutOverrides(base.layout ?? {}, t));
  const w = g._fullLayout?.width;
  const h = g._fullLayout?.height;
  layout = { ...layout, ...(w ? { width: w } : {}), ...(h ? { height: h } : {}) };
  const data = restoreLight(g.data as Data[], base.data, dataOverrides(base.data ?? []));
  return { data, layout, config: g._context };
}
