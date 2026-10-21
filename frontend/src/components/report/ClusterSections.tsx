import type { ReactNode } from "react";
import type { ClusterReport, ReportPart, ReportSection } from "@/api/report";
import { CapNote } from "@/components/common/Caption";
import { Bar, DotStrip, GeneLink, Part, ReportTable } from "./ReportCard";
import { count, num, pct, pval } from "./format";
import { MethodsBody } from "./MethodsBody";

interface Identity {
  n_cells: number;
  fraction: number;
  n_groups: number;
  condition: string | null;
  donor: string | null;
  conditions: string[] | null;
  by_condition: { value: string; n_cells: number; fraction: number }[] | null;
  by_donor: { donor: string; n_cells: number; fraction: number; by_condition?: Record<string, number> }[] | null;
  by_donor_total?: number;
}

interface Labels {
  columns: { column: string; kind: string; top: { label: string; n_cells: number; fraction: number }[];
    n_labels: number; n_unlabelled: number }[];
  confidence: ReportPart & {
    checks: { column: string; alpha: number | null; counts: Record<string, number>; stale: boolean;
      own_label: { confident?: number; ambiguous?: number; unsure?: number; not_fit?: number } | null }[];
  };
}

interface MarkerRow {
  gene: string;
  rank: number | null;
  logfc: number | null;
  padj: number | null;
  pct_in: number | null;
  pct_out: number | null;
}

interface Markers {
  genes?: MarkerRow[];
  n_passing?: number;
  n_shown?: number;
  database?: ReportPart & {
    verdict_text?: string;
    db_type?: string | null;
    match?: string | null;
    p?: number | null;
    overlap?: string[];
    suggestions?: { cell_type: string; p: number | null; overlap: string[]; sources: string[] }[];
    verdict_sources?: string[];
    suggestion_sources?: string[];
    species_note?: string | null;
  };
}

interface Enrichment {
  terms: { term: string; padj: number | null; overlap: number; set_size: number; genes: string[] }[];
  n_tested?: number;
}

interface Qc {
  medians: { metric: string; label: string; cluster: number | null; rest: number | null; note: string | null }[];
  flags: ReportPart & { flags?: { flag: string; reason: string; help: string }[]; notes?: string[] };
  doublets: ReportPart & { n_doublets?: number | null; fraction?: number | null; fraction_rest?: number | null;
    median_score?: number | null; median_score_rest?: number | null };
  states: ReportPart & { column?: string; states?: { state: string; n_cells: number; fraction: number }[] };
  removed: ReportPart & { n_removed?: number; fraction_lost?: number | null; by_criterion?: Record<string, number>;
    truncated?: boolean };
}

interface Composition {
  condition: string | null;
  donor: string | null;
  conditions: string[];
  by_condition: { condition: string; n_cells: number; n_total: number; share: number | null }[] | null;
  samples: { donor: string; condition: string; n_total: number; n_cells: number; share: number | null;
    few_cells: boolean }[] | null;
  stored_test: { test: string | null; condition_key: string; sample_key: string | null; design: string | null;
    props: Record<string, number | null>; fdr: number | null } | null;
  min_sample_cells: number;
}


function IdentityBody({ d }: { d: Identity }) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-3 text-[11px] text-slate-600">
        <span title="Cells in this group">
          <span className="font-semibold text-slate-800">{count(d.n_cells)}</span> cells
        </span>
        <span title="Cells in this group / all cells">
          <span className="font-semibold text-slate-800">{pct(d.fraction)}</span> of the dataset
        </span>
        <span title="Groups of the column">{count(d.n_groups)} groups in the column</span>
      </div>
      {d.by_condition && (
        <ReportTable
          label={`Cells per ${d.condition}`}
          rows={d.by_condition}
          rowKey={(r) => r.value}
          columns={[
            { label: d.condition ?? "Condition", title: "Condition group", cell: (r) => r.value },
            { label: "Cells", title: "Cells of this cluster in the condition", numeric: true, cell: (r) => count(r.n_cells) },
            { label: "Share of cluster", title: "Cells in the condition / cells of the cluster", numeric: true, cell: (r) => pct(r.fraction) },
          ]}
        />
      )}
      {d.by_donor && (
        <>
          <ReportTable
            label={`Cells per ${d.donor}`}
            rows={d.by_donor}
            rowKey={(r) => r.donor}
            columns={[
              { label: d.donor ?? "Donor", title: "Donor or sample", cell: (r) => r.donor },
              { label: "Cells", title: "Cells of this cluster from the donor", numeric: true, cell: (r) => count(r.n_cells) },
              { label: "Share", title: "Donor's cells / cells of the cluster", numeric: true, cell: (r) => pct(r.fraction) },
              ...(d.conditions ?? []).map((c) => ({
                label: c,
                title: `Cells of this cluster from the donor in ${c}`,
                numeric: true,
                cell: (r: NonNullable<Identity["by_donor"]>[number]) => count(r.by_condition?.[c] ?? 0),
              })),
            ]}
          />
          {(d.by_donor_total ?? 0) > d.by_donor.length && (
            <p className="text-[11px] text-slate-500">
              First {d.by_donor.length} of {count(d.by_donor_total ?? 0)} donors.
            </p>
          )}
        </>
      )}
    </div>
  );
}

function LabelsBody({ d }: { d: Labels }) {
  return (
    <div className="space-y-1">
      <ReportTable
        label="Other label columns"
        rows={d.columns ?? []}
        rowKey={(r) => r.column}
        columns={[
          { label: "Column", title: "Another label column", cell: (r) => <span className="font-mono">{r.column}</span> },
          { label: "Made by", title: "How the labels were made (author, CellTypist, LLM, consensus, hand labels, clustering)", cell: (r) => r.kind },
          {
            label: "Top labels",
            title: "The commonest labels among these cells, as a share of the cluster's cells",
            cell: (r) => (
              <span className="flex flex-wrap gap-1">
                {r.top.map((t) => (
                  <span key={t.label} className="rounded bg-slate-100 px-1 py-0.5 text-[10px] text-slate-700" title={`${count(t.n_cells)} cells`}>
                    {t.label} {pct(t.fraction, 0)}
                  </span>
                ))}
              </span>
            ),
          },
          { label: "Unlabelled", title: "Cells of the cluster without a label in this column", numeric: true, cell: (r) => count(r.n_unlabelled) },
        ]}
      />
      <Part title="Label confidence" part={d.confidence}>
        <ReportTable
          label="Label confidence"
          rows={d.confidence?.checks ?? []}
          rowKey={(r) => r.column}
          columns={[
            { label: "Checked column", title: "Label column the split-conformal check ran on", cell: (r) => <span className="font-mono">{r.column}{r.stale ? " (stale)" : ""}</span> },
            { label: "α", title: "Error rate of the prediction sets", numeric: true, cell: (r) => num(r.alpha) },
            { label: "Confident", title: "Cells whose set is their own label alone", numeric: true, cell: (r) => count(r.counts["confident"]) },
            { label: "Ambiguous", title: "Own label and others in the set", numeric: true, cell: (r) => count(r.counts["ambiguous"]) },
            { label: "Unsure", title: "Empty set: no label fits well enough", numeric: true, cell: (r) => count(r.counts["unsure"]) },
            { label: "Doesn't fit", title: "Other labels in the set, not its own", numeric: true, cell: (r) => count(r.counts["doesn't fit"]) },
          ]}
        />
        {typeof d.confidence?.note === "string" && <p className="text-[10px] text-slate-400">{d.confidence.note}</p>}
      </Part>
    </div>
  );
}

function MarkersBody({ d, onGene }: { d: Markers; onGene: (gene: string) => void }) {
  const db = d.database;
  return (
    <div className="space-y-1">
      {d.genes && d.genes.length > 0 && (
        <ReportTable
          label="Marker genes"
          rows={d.genes}
          rowKey={(r) => r.gene}
          columns={[
            { label: "Gene", title: "Marker gene (click for its report)", cell: (r) => <GeneLink gene={r.gene} onGene={onGene} /> },
            { label: "log2FC", title: "Log2 fold change vs the rest of the column", numeric: true, cell: (r) => num(r.logfc) },
            { label: "% in", title: "Share of the group's cells expressing it", numeric: true, cell: (r) => pct(r.pct_in, 0) },
            { label: "% rest", title: "Share of the other cells expressing it", numeric: true, cell: (r) => pct(r.pct_out, 0) },
            { label: "Adj. p", title: "Adjusted p-value of the stored test", numeric: true, cell: (r) => pval(r.padj) },
          ]}
        />
      )}
      {d.n_passing != null && d.n_shown != null && (
        <CapNote shown={d.n_shown} total={d.n_passing} by="log fold change" />
      )}
      <Part title="Marker databases" part={db}>
        {db && (
          <div className="space-y-1 text-[11px] text-slate-700">
            <p>
              <span className="font-semibold">{db.verdict_text}</span>
              {db.db_type && <> for {db.db_type}</>}
              {db.overlap && db.overlap.length > 0 && (
                <> (p {pval(db.p)}; shared: {db.overlap.join(", ")})</>
              )}
              {db.match && <span className="text-slate-500"> · name matched {db.match}</span>}
            </p>
            <ReportTable
              label="Best database types"
              rows={db.suggestions ?? []}
              rowKey={(r) => r.cell_type}
              columns={[
                { label: "Best database types", title: "Database cell types that best match the group's top marker genes", cell: (r) => r.cell_type },
                { label: "p", title: "Hypergeometric p of the overlap", numeric: true, cell: (r) => pval(r.p) },
                { label: "Shared genes", title: "Group markers in the type's list", cell: (r) => <span className="text-slate-500">{r.overlap.join(", ")}</span> },
              ]}
            />
            <p className="text-[10px] text-slate-400">
              Verdict from {(db.verdict_sources ?? []).join(", ")}; suggestions from {(db.suggestion_sources ?? []).join(", ")}.
              {db.species_note ? ` ${db.species_note}` : ""}
            </p>
          </div>
        )}
      </Part>
    </div>
  );
}

function EnrichmentBody({ d, onGene }: { d: Enrichment; onGene: (gene: string) => void }) {
  return (
    <ReportTable
      label="HALLMARK enrichment"
      rows={d.terms ?? []}
      rowKey={(r) => r.term}
      columns={[
        { label: "Set", title: "MSigDB HALLMARK gene set", cell: (r) => <span className="break-all">{r.term.replace(/^HALLMARK_/, "")}</span> },
        { label: "Overlap", title: "Query genes in the set / the set's size in the background", numeric: true, cell: (r) => `${r.overlap}/${r.set_size}` },
        { label: "Adj. p", title: "Benjamini-Hochberg adjusted p", numeric: true, cell: (r) => pval(r.padj) },
        {
          label: "Genes",
          title: "Query genes in the set (click for a gene's report)",
          cell: (r) => (
            <span className="flex flex-wrap gap-x-1 text-[10px]">
              {r.genes.slice(0, 8).map((g) => <GeneLink key={g} gene={g} onGene={onGene} />)}
              {r.genes.length > 8 && <span className="text-slate-400">+{r.genes.length - 8}</span>}
            </span>
          ),
        },
      ]}
    />
  );
}

function QcBody({ d }: { d: Qc }) {
  const removed = d.removed;
  const crit = Object.entries(removed?.by_criterion ?? {}).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(", ");
  return (
    <div className="space-y-1">
      <ReportTable
        label="QC medians"
        rows={d.medians ?? []}
        rowKey={(r) => r.metric}
        columns={[
          { label: "Metric", title: "Per-cell QC value", cell: (r) => r.label },
          { label: "Cluster", title: "Median over this cluster's cells", numeric: true, cell: (r) => (r.note ? <span className="text-slate-400">{r.note}</span> : num(r.cluster, r.metric.startsWith("pct") ? 2 : 0)) },
          { label: "Rest", title: "Median over the other cells", numeric: true, cell: (r) => (r.note ? "" : num(r.rest, r.metric.startsWith("pct") ? 2 : 0)) },
        ]}
      />
      <Part title="Cluster QC flags" part={d.flags}>
        {d.flags?.flags && d.flags.flags.length > 0 ? (
          <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-slate-700">
            {d.flags.flags.map((f) => (
              <li key={f.flag} title={f.help}>
                <span className="font-semibold text-amber-800">{f.flag}</span>: {f.reason}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[11px] text-slate-600">No flag: this cluster does not stand out from the others.</p>
        )}
        {(d.flags?.notes ?? []).map((n) => <p key={n} className="text-[11px] text-slate-600">{n}</p>)}
        {typeof d.flags?.note === "string" && <p className="text-[10px] text-slate-400">{d.flags.note}</p>}
      </Part>
      <Part title="Doublets" part={d.doublets}>
        <p className="text-[11px] text-slate-700">
          {d.doublets?.fraction != null && (
            <span title="Predicted doublets among the cluster's cells">
              {count(d.doublets.n_doublets ?? null)} predicted doublets ({pct(d.doublets.fraction)}; rest {pct(d.doublets.fraction_rest)}).{" "}
            </span>
          )}
          {d.doublets?.median_score != null && (
            <span title="Median doublet score">
              Median doublet score {num(d.doublets.median_score)} (rest {num(d.doublets.median_score_rest)}).
            </span>
          )}
        </p>
      </Part>
      <Part title="QC states" part={d.states}>
        <ReportTable
          label="QC states"
          rows={d.states?.states ?? []}
          rowKey={(r) => r.state}
          columns={[
            { label: d.states?.column ?? "State", title: "Saved QC state (RP share vs mito)", cell: (r) => r.state },
            { label: "Cells", title: "Cells of this cluster", numeric: true, cell: (r) => count(r.n_cells) },
            { label: "Share", title: "Share of the cluster", numeric: true, cell: (r) => pct(r.fraction) },
          ]}
        />
      </Part>
      <Part title="Removed by filtering" part={removed}>
        <p className="text-[11px] text-slate-700">
          The filtering step removed {removed?.truncated ? "at least " : ""}
          {count(removed?.n_removed ?? null)} cells of this group ({pct(removed?.fraction_lost ?? null)} of it)
          {crit ? `: ${crit}` : ""}.
        </p>
      </Part>
    </div>
  );
}

function CompositionBody({ d }: { d: Composition }) {
  const maxShare = Math.max(
    0.01,
    ...(d.by_condition ?? []).map((r) => r.share ?? 0),
    ...(d.samples ?? []).map((s) => s.share ?? 0),
  );
  const conds = d.conditions.length ? d.conditions : [""];
  const few = (d.samples ?? []).filter((s) => s.few_cells).length;
  return (
    <div className="space-y-2">
      {d.by_condition && (
        <div className="space-y-1" aria-label="Share per condition">
          {d.by_condition.map((r) => (
            <div key={r.condition} className="flex items-center gap-2 text-[11px]">
              <span className="w-20 truncate text-slate-600" title={r.condition}>{r.condition}</span>
              <Bar value={r.share} max={maxShare} title={`${count(r.n_cells)} of ${count(r.n_total)} cells`} />
              <span className="w-24 text-right tabular-nums text-slate-700" title={`${count(r.n_cells)} of ${count(r.n_total)} cells`}>
                {pct(r.share)}
              </span>
            </div>
          ))}
        </div>
      )}
      {d.samples && d.samples.length > 0 && (
        <div className="space-y-1" aria-label="Share per donor">
          <p className="text-[10px] font-medium text-slate-500">Per {d.donor} (one point per donor; hover for its name)</p>
          {conds.map((c) => {
            const pts = (d.samples ?? []).filter((s) => s.condition === c);
            if (!pts.length) return null;
            return (
              <div key={c || "all"} className="flex items-center gap-2 text-[11px]">
                <span className="w-20 truncate text-slate-600" title={c || "All cells"}>{c || "All cells"}</span>
                <DotStrip points={pts.map((s) => ({ value: s.share, label: `${s.donor} (${s.n_cells}/${s.n_total} cells)` }))} max={maxShare} format={(v) => pct(v)} />
                <span className="w-24 text-right text-[10px] text-slate-400">0 – {pct(maxShare, 0)}</span>
              </div>
            );
          })}
          {few > 0 && (
            <p className="text-[10px] text-slate-400">
              {few} donor sample{few === 1 ? " has" : "s have"} fewer than {d.min_sample_cells} cells: their share is mostly noise.
            </p>
          )}
        </div>
      )}
      {d.stored_test && (
        <p className="text-[11px] text-slate-700" title="The last composition test run in Compare > Composition">
          Stored test ({d.stored_test.test ?? "not tested"}, {d.stored_test.design ?? ""}, samples from{" "}
          {d.stored_test.sample_key ?? "none"}): mean shares{" "}
          {Object.entries(d.stored_test.props).map(([k, v]) => `${k} ${pct(v)}`).join(", ")}; FDR {pval(d.stored_test.fdr)}.
        </p>
      )}
    </div>
  );
}

/** The body of one cluster-report section (the card is ReportCard). */
export function ClusterSectionBody({ section, report, onGene }: {
  section: ReportSection;
  report: ClusterReport;
  onGene: (gene: string) => void;
}): ReactNode {
  const d = section.data as never;
  switch (section.id) {
    case "identity":
      return section.status === "ok" ? <IdentityBody d={d} /> : null;
    case "labels":
      return <LabelsBody d={d} />;
    case "markers":
      return <MarkersBody d={d} onGene={onGene} />;
    case "enrichment":
      return section.status === "ok" ? <EnrichmentBody d={d} onGene={onGene} /> : null;
    case "qc":
      return section.status === "ok" ? <QcBody d={d} /> : null;
    case "composition":
      return section.status === "ok" ? <CompositionBody d={d} /> : null;
    case "methods":
      return section.status === "ok" ? <MethodsBody d={d} generatedAt={report.generated_at} version={report.scview_version} /> : null;
    default:
      return null;
  }
}
