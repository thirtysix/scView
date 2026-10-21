import { useMemo } from "react";
import { upsetLayout } from "@/lib/upset";

export interface UpSetPlotSet {
  name: string;
  n: number;
  /** CSS colour (from the categorical palette) */
  color: string;
  /** Tooltip: what the set is, with its counts */
  title?: string;
}

export interface UpSetPlotColumn {
  /** Indices into `sets` */
  sets: number[];
  n: number;
  /** Tooltip and accessible name: which sets, how many genes */
  title: string;
}

interface Props {
  sets: readonly UpSetPlotSet[];
  /** Exclusive intersections, in the order to draw them (largest first) */
  columns: readonly UpSetPlotColumn[];
  selected: number | null;
  onSelect: (column: number) => void;
  /** Accessible name of the whole plot */
  label: string;
}

const BAR = "#475569"; // slate-600
const DOT_ON = "#334155"; // slate-700
const DOT_OFF = "#e2e8f0"; // slate-200
const STRIPE = "#f8fafc"; // slate-50
const MAX_NAME = 27; // fits the names' column (lib/upset.ts nameW) at 10px

const truncate = (s: string) => (s.length > MAX_NAME ? `${s.slice(0, MAX_NAME - 1)}…` : s);

/**
 * UpSet plot (lib/upset.ts geometry): one column per exclusive intersection with
 * its size as a bar on top, the sets as rows of a dot matrix (filled = the
 * intersection contains the set), and each set's size as a bar on the left in
 * its palette colour. A column is a button: click (or Enter) selects it.
 */
export function UpSetPlot({ sets, columns, selected, onSelect, label }: Props) {
  const L = useMemo(() => upsetLayout(sets, columns), [sets, columns]);

  return (
    <svg
      role="group"
      aria-label={label}
      viewBox={`0 0 ${L.width} ${L.height}`}
      width="100%"
      style={{ maxWidth: L.width, minWidth: Math.min(L.width, 320) }}
      className="block select-none text-slate-600"
      data-testid="upset-plot"
    >
      {/* Row stripes */}
      {sets.map((_, j) =>
        j % 2 === 0 ? (
          <rect
            key={`stripe-${j}`}
            x={L.matrixLeft - 4}
            y={L.rowY[j]! - L.rowH / 2}
            width={L.width - L.matrixLeft}
            height={L.rowH}
            fill={STRIPE}
          />
        ) : null,
      )}

      {/* Set names, set-size bars and counts */}
      {sets.map((s, j) => {
        const b = L.setBars[j]!;
        return (
          <g key={`set-${j}`} data-testid="upset-set">
            <title>{s.title ?? `${s.name}: ${s.n.toLocaleString()} genes`}</title>
            <rect x={b.x} y={b.y} width={b.w} height={b.h} fill={s.color} rx={1} data-testid="upset-set-bar" />
            <text x={b.x - 3} y={L.rowY[j]!} dy="0.32em" textAnchor="end" fontSize={8} fill="#64748b">
              {s.n.toLocaleString()}
            </text>
            <text x={L.matrixLeft - 8} y={L.rowY[j]!} dy="0.32em" textAnchor="end" fontSize={10} fill="#334155">
              {truncate(s.name)}
            </text>
          </g>
        );
      })}

      {/* Intersection columns: bar, count, dots, line; the whole column is the button */}
      {columns.map((c, i) => {
        const b = L.bars[i]!;
        const on = selected === i;
        const x0 = L.matrixLeft + i * L.colW;
        const line = L.lines.find((l) => l.column === i);
        const select = () => onSelect(i);
        return (
          <g
            key={`col-${i}`}
            role="button"
            tabIndex={0}
            aria-label={c.title}
            aria-pressed={on}
            onClick={select}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                select();
              }
            }}
            className="cursor-pointer"
            data-testid="upset-column"
          >
            <title>{c.title}</title>
            <rect
              className="hit"
              x={x0}
              y={L.barsTop}
              width={L.colW}
              height={L.height - L.barsTop - 4}
              fill={on ? "var(--color-primary)" : "transparent"}
              fillOpacity={on ? 0.1 : 0}
              strokeWidth={1}
            />
            <rect
              x={b.x}
              y={b.y}
              width={b.w}
              height={Math.max(b.h, c.n > 0 ? 1 : 0)}
              fill={on ? "var(--color-primary)" : BAR}
              data-testid="upset-bar"
            />
            <text x={b.x + b.w / 2} y={b.y - 3} textAnchor="middle" fontSize={8} fill="#475569">
              {c.n.toLocaleString()}
            </text>
            {line && <line x1={line.x} x2={line.x} y1={line.y1} y2={line.y2} stroke={DOT_ON} strokeWidth={2} />}
            {L.dots
              .filter((d) => d.column === i)
              .map((d) => (
                <circle
                  key={d.set}
                  cx={d.cx}
                  cy={d.cy}
                  r={4}
                  fill={d.filled ? (on ? "var(--color-primary)" : DOT_ON) : DOT_OFF}
                  data-testid={d.filled ? "upset-dot-on" : "upset-dot-off"}
                />
              ))}
          </g>
        );
      })}
      <text x={L.matrixLeft} y={L.height - 3} fontSize={8} fill="#94a3b8">
        Bars: genes in exactly these sets
      </text>
    </svg>
  );
}
