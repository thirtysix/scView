/**
 * Geometry of an UpSet plot (components/plots/UpSetPlot.tsx): intersection-size
 * bars on top, one column per intersection; set-size bars on the left, growing
 * leftwards from the set names; a dot matrix below the bars, one row per set,
 * filled where the column's intersection contains the set, with a line joining
 * the filled dots. Pure, so the plot's shape is tested without a DOM.
 */

export interface UpSetSetInput {
  name: string;
  n: number;
}

export interface UpSetColumnInput {
  /** Indices into the sets */
  sets: number[];
  n: number;
}

export interface UpSetBar {
  index: number;
  x: number;
  y: number;
  w: number;
  h: number;
  n: number;
}

export interface UpSetDot {
  column: number;
  set: number;
  cx: number;
  cy: number;
  filled: boolean;
}

export interface UpSetLine {
  column: number;
  x: number;
  y1: number;
  y2: number;
}

export interface UpSetLayout {
  width: number;
  height: number;
  /** Top of the intersection bars' area (their baseline is `barsBottom`) */
  barsTop: number;
  barsBottom: number;
  matrixTop: number;
  matrixLeft: number;
  /** Right edge of the set-size bars; the set names are right-aligned at matrixLeft - 6 */
  setBarsRight: number;
  colW: number;
  rowH: number;
  bars: UpSetBar[];
  setBars: UpSetBar[];
  dots: UpSetDot[];
  lines: UpSetLine[];
  /** Vertical centre of each set's row */
  rowY: number[];
  maxColumn: number;
  maxSet: number;
}

/** Room left of the longest set-size bar for its count */
export const COUNT_W = 30;

export const UPSET_DEFAULTS = {
  colW: 18,
  rowH: 18,
  barsH: 110,
  /** Room above the tallest bar for its count */
  top: 14,
  /** Set-size bars' area; its left COUNT_W px hold the counts */
  setBarW: 72,
  nameW: 170,
  gap: 6,
  right: 8,
  bottom: 16,
};

export function upsetLayout(
  sets: readonly UpSetSetInput[],
  columns: readonly UpSetColumnInput[],
  opts: Partial<typeof UPSET_DEFAULTS> = {},
): UpSetLayout {
  const o = { ...UPSET_DEFAULTS, ...opts };
  const maxColumn = Math.max(0, ...columns.map((c) => c.n));
  const maxSet = Math.max(0, ...sets.map((s) => s.n));
  const barsTop = o.top;
  const barsBottom = o.top + o.barsH;
  const matrixTop = barsBottom + o.gap;
  const setBarsRight = o.setBarW;
  const matrixLeft = o.setBarW + o.nameW;
  const width = matrixLeft + Math.max(columns.length, 1) * o.colW + o.right;
  const height = matrixTop + sets.length * o.rowH + o.bottom;
  const rowY = sets.map((_, j) => matrixTop + j * o.rowH + o.rowH / 2);
  const colX = (i: number) => matrixLeft + i * o.colW + o.colW / 2;

  const bars = columns.map((c, i) => {
    const h = maxColumn > 0 ? (c.n / maxColumn) * o.barsH : 0;
    const w = o.colW * 0.7;
    return { index: i, x: colX(i) - w / 2, y: barsBottom - h, w, h, n: c.n };
  });
  const setBars = sets.map((s, j) => {
    const w = maxSet > 0 ? (s.n / maxSet) * (o.setBarW - COUNT_W) : 0;
    const h = o.rowH * 0.6;
    return { index: j, x: setBarsRight - w, y: rowY[j]! - h / 2, w, h, n: s.n };
  });
  const dots: UpSetDot[] = [];
  const lines: UpSetLine[] = [];
  columns.forEach((c, i) => {
    const members = new Set(c.sets);
    sets.forEach((_, j) => dots.push({ column: i, set: j, cx: colX(i), cy: rowY[j]!, filled: members.has(j) }));
    const ys = c.sets.filter((j) => j >= 0 && j < sets.length).map((j) => rowY[j]!);
    if (ys.length > 1) lines.push({ column: i, x: colX(i), y1: Math.min(...ys), y2: Math.max(...ys) });
  });
  return {
    width,
    height,
    barsTop,
    barsBottom,
    matrixTop,
    matrixLeft,
    setBarsRight,
    colW: o.colW,
    rowH: o.rowH,
    bars,
    setBars,
    dots,
    lines,
    rowY,
    maxColumn,
    maxSet,
  };
}

/** "CD14 Mono: ... and B: ..." for an intersection's accessible name and tooltip. */
export function intersectionName(setNames: readonly string[], members: readonly number[]): string {
  const names = members.map((j) => setNames[j] ?? `set ${j + 1}`);
  if (names.length === 1) return `only ${names[0]}`;
  if (names.length === setNames.length) return `all ${names.length} sets`;
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]} only`;
}
