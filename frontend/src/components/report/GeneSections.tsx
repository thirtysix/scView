import type { ReactNode } from "react";
import { GeneLinkOuts } from "@/components/unified/GeneLinkOuts";
import type { GeneReport, ReportSection } from "@/api/report";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { rgbCss } from "@/lib/categoryColors";
import { Bar, DotStrip, GeneLink, ReportTable } from "./ReportCard";
import { count, num, pct, pval } from "./format";
import { MethodsBody } from "./MethodsBody";

interface Expression {
  layer_label: string;
  groups: { group: string; n_cells: number; mean: number | null; mean_expressing: number | null; fraction: number | null }[];
}

interface GeneMarkers {
  groups: { group: string; logfc: number | null; padj: number | null; pct_in: number | null; pct_out: number | null;
    marker_rank: number; n_markers: number }[];
}

interface Databases {
  types: { cell_type: string; rank: number; n_genes: number; sources: string[] }[];
  n_types: number;
}

interface PerDonor {
  groups: string[];
  conditions: string[];
  summary: { group: string; condition: string; n_donors: number; median: number | null; min: number | null; max: number | null }[];
  points: { group: string; condition: string; sample: string; n_cells: number; value: number }[];
  n_points: number;
  paired_tests: Record<string, { n_pairs: number; mean_diff: number | null; wilcoxon_p: number | null; ttest_p: number | null; caution: string | null }>;
}

interface Coexpression {
  positive: { gene: string; r: number; pct_expressing: number }[];
  negative: { gene: string; r: number; pct_expressing: number }[];
}

interface Hvg {
  stats: { field: string; label: string; value: number | null }[];
}

interface GeneSets {
  sets: { collection: string; collection_name: string; name: string }[];
  n_sets: number;
  by_collection: { collection: string; collection_name: string; n_sets: number }[];
}

interface Protein {
  proteins: { protein: string; control: boolean; spearman_rna: number | null; groups: { group: string; mean_clr: number | null }[] }[];
}

function GroupLink({ group, column, onGroup }: { group: string; column: string; onGroup: (group: string) => void }) {
  const colorOf = useCategoryColors(column);
  return (
    <button
      type="button"
      onClick={() => onGroup(group)}
      className="flex items-center gap-1 text-left text-slate-700 hover:text-primary hover:underline"
      title={`Open the cluster report for ${group}`}
    >
      <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: rgbCss(colorOf(group)) }} />
      <span className="break-all">{group}</span>
    </button>
  );
}


function ExpressionBody({ d, column, onGroup }: { d: Expression; column: string; onGroup: (g: string) => void }) {
  const top = Math.max(0, ...d.groups.map((g) => g.mean ?? 0));
  return (
    <ReportTable
      label="Expression per group"
      rows={d.groups}
      rowKey={(r) => r.group}
      columns={[
        { label: "Group", title: "Group of the column (click for its cluster report)", cell: (r) => <GroupLink group={r.group} column={column} onGroup={onGroup} /> },
        { label: "Cells", title: "Cells in the group", numeric: true, cell: (r) => count(r.n_cells) },
        { label: "Mean", title: `Mean ${d.layer_label} value over all the group's cells`, numeric: true, cell: (r) => num(r.mean) },
        { label: "", title: "Mean, relative to the highest group", cell: (r) => <Bar value={r.mean} max={top} /> },
        { label: "Mean expr.", title: "Mean over the cells expressing it (above 0)", numeric: true, cell: (r) => num(r.mean_expressing) },
        { label: "% expr.", title: "Share of the group's cells above 0", numeric: true, cell: (r) => pct(r.fraction) },
      ]}
    />
  );
}

function MarkerBody({ d, column, onGroup }: { d: GeneMarkers; column: string; onGroup: (g: string) => void }) {
  return (
    <ReportTable
      label="Groups where it is a marker"
      rows={d.groups ?? []}
      rowKey={(r) => r.group}
      columns={[
        { label: "Group", title: "Group where the gene passes the marker filters", cell: (r) => <GroupLink group={r.group} column={column} onGroup={onGroup} /> },
        { label: "log2FC", title: "Log2 fold change vs the rest", numeric: true, cell: (r) => num(r.logfc) },
        { label: "% in", title: "Share of the group's cells expressing it", numeric: true, cell: (r) => pct(r.pct_in, 0) },
        { label: "% rest", title: "Share of the other cells expressing it", numeric: true, cell: (r) => pct(r.pct_out, 0) },
        { label: "Adj. p", title: "Adjusted p-value of the stored test", numeric: true, cell: (r) => pval(r.padj) },
        { label: "Rank", title: "Place among the group's markers by log fold change", numeric: true, cell: (r) => `${r.marker_rank} of ${count(r.n_markers)}` },
      ]}
    />
  );
}

function DatabasesBody({ d }: { d: Databases }) {
  return (
    <>
      <ReportTable
        label="Database cell types"
        rows={d.types ?? []}
        rowKey={(r, i) => `${r.cell_type}|${i}`}
        columns={[
          { label: "Cell type", title: "Database cell type listing the gene", cell: (r) => r.cell_type },
          { label: "Rank", title: "The gene's place in the type's marker list (1 = best evidenced)", numeric: true, cell: (r) => count(r.rank) },
          { label: "List", title: "Genes the type lists", numeric: true, cell: (r) => count(r.n_genes) },
          { label: "Sources", title: "Databases naming the type", cell: (r) => <span className="text-slate-500">{r.sources.join(", ")}</span> },
        ]}
      />
      {d.n_types > (d.types?.length ?? 0) && (
        <p className="text-[11px] text-slate-500">First {d.types.length} of {count(d.n_types)} cell types, by rank.</p>
      )}
    </>
  );
}

function PerDonorBody({ d, column, onGroup }: { d: PerDonor; column: string; onGroup: (g: string) => void }) {
  const max = Math.max(1e-6, ...d.points.map((p) => p.value));
  const tests = Object.entries(d.paired_tests ?? {});
  return (
    <div className="space-y-2">
      <ReportTable
        label="Per-donor values"
        rows={d.summary ?? []}
        rowKey={(r) => `${r.group}|${r.condition}`}
        columns={[
          { label: "Group", title: "Group of the column", cell: (r) => <GroupLink group={r.group} column={column} onGroup={onGroup} /> },
          ...(d.conditions.some((c) => c) ? [{ label: "Condition", title: "Condition group", cell: (r: PerDonor["summary"][number]) => r.condition }] : []),
          { label: "Donors", title: "Donor samples with enough cells", numeric: true, cell: (r) => count(r.n_donors) },
          {
            label: "Per donor",
            title: "One point per donor (hover for its name and value)",
            cell: (r) => (
              <DotStrip
                points={d.points.filter((p) => p.group === r.group && p.condition === r.condition)
                  .map((p) => ({ value: p.value, label: `${p.sample} (${p.n_cells} cells)` }))}
                max={max}
                format={(v) => v.toFixed(3)}
              />
            ),
          },
          { label: "Median", title: "Median over the donors", numeric: true, cell: (r) => num(r.median, 3) },
        ]}
      />
      {d.n_points > d.points.length && (
        <p className="text-[11px] text-slate-500">Points for the first {d.points.length} of {count(d.n_points)} samples.</p>
      )}
      {tests.length > 0 && (
        <ReportTable
          label="Paired tests"
          rows={tests}
          rowKey={([g]) => g}
          columns={[
            { label: "Group", title: "Group of the column", cell: ([g]) => g },
            { label: "Pairs", title: "Donors with a value in both conditions", numeric: true, cell: ([, t]) => count(t.n_pairs) },
            { label: "Change", title: "Mean of second minus first condition, over the pairs", numeric: true, cell: ([, t]) => num(t.mean_diff, 3) },
            { label: "Wilcoxon p", title: "Wilcoxon signed-rank test on the pairs", numeric: true, cell: ([, t]) => pval(t.wilcoxon_p) },
            { label: "Paired t p", title: "Paired t-test on the pairs", numeric: true, cell: ([, t]) => pval(t.ttest_p) },
          ]}
        />
      )}
    </div>
  );
}

function CoexpressionBody({ d, onGene }: { d: Coexpression; onGene: (gene: string) => void }) {
  const cols = [
    { label: "Gene", title: "Correlated gene (click for its report)", cell: (r: Coexpression["positive"][number]) => <GeneLink gene={r.gene} onGene={onGene} /> },
    { label: "r", title: "Pearson correlation over cells", numeric: true, cell: (r: Coexpression["positive"][number]) => num(r.r) },
    { label: "% expr.", title: "Share of cells expressing it", numeric: true, cell: (r: Coexpression["positive"][number]) => `${r.pct_expressing.toFixed(0)}%` },
  ];
  return (
    <div className="grid grid-cols-2 gap-2">
      <div>
        <p className="text-[10px] font-medium text-slate-500">Most positive</p>
        <ReportTable label="Most positively correlated" rows={d.positive ?? []} rowKey={(r) => r.gene} columns={cols} />
      </div>
      <div>
        <p className="text-[10px] font-medium text-slate-500">Most negative</p>
        <ReportTable label="Most negatively correlated" rows={d.negative ?? []} rowKey={(r) => r.gene} columns={cols} />
      </div>
    </div>
  );
}

function HvgBody({ d }: { d: Hvg }) {
  if (!d.stats?.length) return null;
  return (
    <p className="text-[11px] text-slate-600">
      {d.stats.map((s) => (
        <span key={s.field} className="mr-3" title={`var['${s.field}']`}>
          {s.label} <span className="font-medium tabular-nums text-slate-800">{num(s.value, 3)}</span>
        </span>
      ))}
    </p>
  );
}

function GeneSetsBody({ d }: { d: GeneSets }) {
  return (
    <div className="space-y-1 text-[11px]">
      <p className="text-slate-600">
        {d.by_collection.map((c) => (
          <span key={c.collection} className="mr-3" title={`Sets in ${c.collection_name} holding the gene`}>
            {c.collection_name} <span className="font-medium text-slate-800">{count(c.n_sets)}</span>
          </span>
        ))}
      </p>
      {d.sets.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-4 text-slate-700">
          {d.sets.map((s) => (
            <li key={`${s.collection}|${s.name}`} className="break-all">
              {s.name} <span className="text-slate-400">({s.collection_name})</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ProteinBody({ d, column, onGroup }: { d: Protein; column: string; onGroup: (g: string) => void }) {
  return (
    <div className="space-y-2">
      {d.proteins.map((p) => {
        const top = Math.max(1e-6, ...p.groups.map((g) => Math.abs(g.mean_clr ?? 0)));
        return (
          <div key={p.protein} className="space-y-1">
            <p className="text-[11px] text-slate-700">
              <span className="font-semibold">{p.protein}</span>
              {p.control && <span className="text-slate-400"> (isotype control)</span>}
              <span className="ml-2" title="Spearman rank correlation of protein (CLR) and RNA over cells">
                Spearman with RNA {num(p.spearman_rna)}
              </span>
            </p>
            <ReportTable
              label={`${p.protein} per group`}
              rows={p.groups}
              rowKey={(r) => r.group}
              columns={[
                { label: "Group", title: "Group of the column", cell: (r) => <GroupLink group={r.group} column={column} onGroup={onGroup} /> },
                { label: "Mean CLR", title: "Mean centred log-ratio of the antibody counts", numeric: true, cell: (r) => num(r.mean_clr) },
                { label: "", title: "Mean CLR relative to the largest", cell: (r) => <Bar value={r.mean_clr} max={top} /> },
              ]}
            />
          </div>
        );
      })}
    </div>
  );
}

/** The body of one gene-report section (the card is ReportCard). */
export function GeneSectionBody({ section, report, onGene, onGroup }: {
  section: ReportSection;
  report: GeneReport;
  onGene: (gene: string) => void;
  onGroup: (group: string) => void;
}): ReactNode {
  if (section.status !== "ok") return null;
  const d = section.data as never;
  const column = report.groupby;
  switch (section.id) {
    case "expression":
      return <ExpressionBody d={d} column={column} onGroup={onGroup} />;
    case "markers":
      return <MarkerBody d={d} column={column} onGroup={onGroup} />;
    case "databases":
      return (
        <>
          <DatabasesBody d={d} />
          <GeneLinkOuts gene={report.gene} className="mt-1.5" />
        </>
      );
    case "per_donor":
      return <PerDonorBody d={d} column={column} onGroup={onGroup} />;
    case "coexpression":
      return <CoexpressionBody d={d} onGene={onGene} />;
    case "hvg":
      return <HvgBody d={d} />;
    case "gene_sets":
      return <GeneSetsBody d={d} />;
    case "protein":
      return <ProteinBody d={d} column={column} onGroup={onGroup} />;
    case "methods":
      return <MethodsBody d={d} generatedAt={report.generated_at} version={report.scview_version} />;
    default:
      return null;
  }
}
