import type { ReactNode } from "react";
import { Caption } from "@/components/common/Caption";
import type { ReportPart, ReportSection, SectionStatus } from "@/api/report";

const STATUS_TEXT: Record<SectionStatus, string> = {
  ok: "",
  not_computed: "Not computed",
  not_applicable: "Not applicable",
};

export function StatusBadge({ status }: { status: SectionStatus }) {
  if (status === "ok") return null;
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
        status === "not_computed" ? "bg-amber-100 text-amber-800" : "bg-slate-100 text-slate-500"
      }`}
      title={
        status === "not_computed"
          ? "Its input was never computed; the note says where to compute it"
          : "It cannot apply to this dataset; the note says why"
      }
    >
      {STATUS_TEXT[status]}
    </span>
  );
}

/** A note under a section or part: amber when something can be computed, slate otherwise. */
export function Note({ status, children }: { status: SectionStatus; children: ReactNode }) {
  return (
    <p
      className={`rounded border-l-2 px-2 py-1 text-[11px] leading-snug ${
        status === "not_computed" ? "border-amber-300 bg-amber-50 text-amber-900" : "border-slate-300 bg-slate-50 text-slate-600"
      }`}
    >
      {children}
    </p>
  );
}

/**
 * One report section as a card: title (+ status), caption with the n, the body,
 * a note when the section or part of it is missing, and the method line in muted text.
 */
export function ReportCard({ section, children }: { section: ReportSection; children?: ReactNode }) {
  return (
    <section
      className="space-y-1.5 rounded-lg border border-slate-200 bg-white p-2.5"
      data-testid={`report-section-${section.id}`}
      aria-label={section.title}
    >
      <div className="flex items-center gap-2">
        <h4 className="text-xs font-semibold text-slate-700">{section.title}</h4>
        <StatusBadge status={section.status} />
      </div>
      {section.caption && <Caption>{section.caption}</Caption>}
      {children}
      {section.note && <Note status={section.status === "ok" ? "not_applicable" : section.status}>{section.note}</Note>}
      {section.method && (
        <p className="text-[10px] leading-snug text-slate-400">
          <span className="font-medium">How: </span>
          {section.method}
        </p>
      )}
    </section>
  );
}

/** A sub-part with its own status: its body when ok, else its note. */
export function Part({ title, part, children }: { title: string; part?: ReportPart | null; children?: ReactNode }) {
  if (!part) return null;
  return (
    <div className="space-y-1 pt-1" data-testid={`report-part-${title}`}>
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold text-slate-600">{title}</span>
        <StatusBadge status={part.status} />
      </div>
      {part.status === "ok" ? children : part.note && <Note status={part.status}>{part.note}</Note>}
    </div>
  );
}

export interface Column<T> {
  label: string;
  /** What the column is and how it is computed (header tooltip). */
  title: string;
  numeric?: boolean;
  cell: (row: T) => ReactNode;
}

/** A compact table; every header carries its tooltip. */
export function ReportTable<T>({ rows, columns, rowKey, label }: {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T, i: number) => string;
  label?: string;
}) {
  if (!rows.length) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px]" aria-label={label}>
        <thead>
          <tr className="border-b border-slate-200 text-slate-500">
            {columns.map((c) => (
              <th key={c.label} title={c.title} className={`px-1.5 py-1 font-medium ${c.numeric ? "text-right" : "text-left"}`}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={rowKey(r, i)} className="border-b border-slate-50">
              {columns.map((c) => (
                <td key={c.label} className={`px-1.5 py-1 align-top ${c.numeric ? "text-right tabular-nums" : ""}`}>
                  {c.cell(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A horizontal bar for a value in [0, max]. */
export function Bar({ value, max, title }: { value: number | null; max: number; title?: string }) {
  const share = value == null || max <= 0 ? 0 : Math.max(0, Math.min(1, value / max));
  return (
    <div className="h-1.5 w-full min-w-[40px] overflow-hidden rounded-full bg-slate-100" title={title}>
      <div className="h-full rounded-full bg-primary" style={{ width: `${share * 100}%` }} />
    </div>
  );
}

/** One row of dots on a shared 0..max axis (a donor per dot; hover names it). */
export function DotStrip({ points, max, format }: {
  points: { value: number | null; label: string }[];
  max: number;
  format: (v: number) => string;
}) {
  return (
    <div className="relative h-3 w-full rounded bg-slate-50">
      <div className="absolute inset-x-0 top-1/2 h-px bg-slate-200" />
      {points.map((p, i) =>
        p.value == null ? null : (
          <span
            key={`${p.label}-${i}`}
            title={`${p.label}: ${format(p.value)}`}
            className="absolute top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-700/75"
            style={{ left: `${max > 0 ? Math.min(1, Math.max(0, p.value / max)) * 100 : 0}%` }}
          />
        ),
      )}
    </div>
  );
}

/** A gene name that opens its gene report. */
export function GeneLink({ gene, onGene }: { gene: string; onGene: (gene: string) => void }) {
  return (
    <button
      type="button"
      onClick={() => onGene(gene)}
      className="font-mono font-medium text-slate-800 hover:text-primary hover:underline"
      title={`Open the gene report for ${gene}`}
    >
      {gene}
    </button>
  );
}
