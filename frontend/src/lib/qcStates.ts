import type { QcStateGroup, QcStates } from "@/api/qcStates";
import { rgbCss, sortByCategoryOrder } from "@/lib/categoryColors";

type Rgb = readonly [number, number, number];

/** The flagged state's colour when one group is shown (main stays slate). */
export const FLAGGED_COLOR = "#dc2626";
export const MAIN_COLOR = "#64748b";
/** Up to this many condition levels get a column each; more get one min-max column. */
export const MAX_LEVEL_COLUMNS = 8;
/** The backend's rule for a level's share (core/qc_states.py MIN_LEVEL_CELLS). */
export const MIN_LEVEL_CELLS = 20;

/** Same column-name rule as the backend (core/qc_states.py check_name). */
export function validStateName(name: string): boolean {
  return /^[A-Za-z][A-Za-z0-9_.-]{0,59}$/.test(name) && !name.startsWith("scview_");
}

/** The server's reason from an apiFetch error ("API error 400: {detail}"). */
export function errorText(e: unknown): string {
  const body = (e instanceof Error ? e.message : String(e)).replace(/^API error \d+: /, "");
  try {
    return String(JSON.parse(body).detail ?? body);
  } catch {
    return body;
  }
}

export function pctText(v: number | null | undefined, digits = 0): string {
  return v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(digits)}%`;
}

/** A deterministic jitter in [-w, w] for point i (a strip plot that does not move on re-render). */
export function jitter(i: number, w = 0.32): number {
  let h = (i + 1) * 2654435761;
  h = (h ^ (h >>> 16)) >>> 0;
  return ((h % 10007) / 10006 - 0.5) * 2 * w;
}

interface Trace {
  type: "scattergl";
  mode: "markers";
  name: string;
  x: number[];
  y: number[];
  marker: Record<string, unknown>;
  hovertemplate: string;
  legendgroup?: string;
  showlegend?: boolean;
}

/**
 * Plotly traces for the QC-state scatter.
 *
 * With mito (2-D): RP share (x) vs % mito (y). All groups: one trace per group and
 * state in the group's colour, the flagged state drawn as crosses. One group
 * (`focus`): main in slate, the flagged state in red.
 *
 * Without mito (1-D): RP share (y) per group (x, jittered), main slate and the
 * flagged state red, as each group is its own column already.
 */
export function stateTraces(
  data: QcStates,
  order: readonly string[],
  colorOf: (name: string) => Rgb,
  focus: string | null,
): Trace[] {
  const sc = data.scatter;
  if (!sc) return [];
  const names = data.groups.map((g) => g.name);
  const flagLabel = (gi: number) => data.groups[gi]?.flagged_label ?? "second state";
  const sorted = sortByCategoryOrder(names, order);
  const rank = new Map(sorted.map((n, i) => [n, i]));
  const twoD = sc.y !== null;
  const buckets = new Map<string, { gi: number; state: number; x: number[]; y: number[] }>();
  sc.x.forEach((x, i) => {
    const gi = sc.group[i]!;
    const name = names[gi];
    if (name === undefined || (focus && name !== focus)) return;
    const state = sc.state[i]!;
    const key = twoD && !focus ? `${gi}:${state}` : `${state}`;
    let b = buckets.get(key);
    if (!b) buckets.set(key, (b = { gi, state, x: [], y: [] }));
    if (twoD) {
      b.x.push(x);
      b.y.push(sc.y![i]!);
    } else {
      b.x.push(rank.get(name)! + jitter(i));
      b.y.push(x);
    }
  });
  const traces: Trace[] = [];
  const keys = [...buckets.keys()].sort((a, b) => {
    const A = buckets.get(a)!;
    const B = buckets.get(b)!;
    return (rank.get(names[A.gi]!)! - rank.get(names[B.gi]!)!) || A.state - B.state;
  });
  const hover = twoD
    ? "%{fullData.name}<br>RP share %{x:.1f}%<br>mito %{y:.1f}%<extra></extra>"
    : "%{fullData.name}<br>RP share %{y:.1f}%<extra></extra>";
  for (const key of keys) {
    const b = buckets.get(key)!;
    const name = names[b.gi]!;
    const flagged = b.state === 1;
    if (twoD && !focus) {
      const color = rgbCss(colorOf(name));
      traces.push({
        type: "scattergl",
        mode: "markers",
        name: flagged ? `${name}: ${flagLabel(b.gi)}` : name,
        legendgroup: name,
        x: b.x,
        y: b.y,
        marker: flagged
          ? { size: 5, color, symbol: "x", opacity: 0.9 }
          : { size: 3, color, opacity: 0.45 },
        hovertemplate: hover,
      });
    } else {
      // One trace per state: the group is named on the axis (1-D) or the title (focus).
      traces.push({
        type: "scattergl",
        mode: "markers",
        name: flagged ? (focus ? flagLabel(b.gi) : "flagged state") : "main",
        x: b.x,
        y: b.y,
        marker: flagged
          ? { size: 4, color: FLAGGED_COLOR, opacity: 0.8 }
          : { size: 3, color: MAIN_COLOR, opacity: 0.45 },
        hovertemplate: hover,
      });
    }
  }
  return traces;
}

/** Category axis for the 1-D strip plot: group names at 0, 1, 2 ... in the column's order. */
export function stripAxis(data: QcStates, order: readonly string[]): { tickvals: number[]; ticktext: string[] } {
  const sorted = sortByCategoryOrder(
    data.groups.map((g) => g.name),
    order,
  );
  return { tickvals: sorted.map((_, i) => i), ticktext: sorted };
}

/** Condition columns for the table: one per level (up to MAX_LEVEL_COLUMNS), else a range. */
export function levelColumns(data: QcStates): { levels: string[]; asRange: boolean } {
  const levels = data.condition?.levels ?? [];
  return { levels, asRange: levels.length > MAX_LEVEL_COLUMNS };
}

/** "12% (CTRL) – 65% (STIM)" over levels with enough cells; null when under two. */
export function shareRange(g: QcStateGroup): string | null {
  const lv = Object.entries(g.per_condition).filter(([, v]) => v.share != null && v.n >= MIN_LEVEL_CELLS);
  if (lv.length < 2) return null;
  lv.sort((a, b) => a[1].share! - b[1].share!);
  const [lo, hi] = [lv[0]!, lv[lv.length - 1]!];
  return `${pctText(lo[1].share)} (${lo[0]}) – ${pctText(hi[1].share)} (${hi[0]})`;
}

/** The summary sentence above the table. */
export function summaryText(data: QcStates): string {
  const fitted = data.groups.filter((g) => g.fitted);
  const two = fitted.filter((g) => g.n_states === 2);
  const skipped = data.groups.length - fitted.length;
  const min = data.rules?.min_cells ?? 100;
  const head =
    two.length === 0
      ? `No group of ${data.column} shows a second state: one state in each of the ${fitted.length} fitted groups.`
      : `Two states in ${two.length} of ${fitted.length} fitted groups of ${data.column} (${two
          .map((g) => `${g.name} ${pctText(g.flagged_share)}`)
          .join(", ")} in the flagged state).`;
  const tail = skipped > 0 ? ` ${skipped} ${skipped === 1 ? "group" : "groups"} under ${min} cells not fitted.` : "";
  return head + tail;
}

/** Why the column was picked, for the caption. */
export function columnReason(reason: QcStates["reason"]): string {
  if (reason === "celltype") return "the cell-type column";
  if (reason === "clustering") return "the clustering";
  return "";
}
