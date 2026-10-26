import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { Loader2, ShieldCheck } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import type { DEResponse } from "@/api/de";
import { runDEAudit, type DEAuditProgress, type DEAuditResult, type DEAuditStart } from "@/api/deAudit";
import { Caption, CapNote } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { errorMessage } from "@/lib/errorMessage";
import { formatPValue } from "@/lib/formatting";
import {
  auditBlocker,
  auditSummary,
  concordanceText,
  fewDonorsNote,
  nullBins,
  progressText,
  schemeLabel,
  shareText,
} from "@/lib/deAudit";

interface Props {
  datasetId: string;
  result: DEResponse;
  /** The result compares two groups of a column (not a selection vs the rest). */
  betweenGroups: boolean;
  /** Donor-like obs columns on offer (lib/de.ts replicateColumns). */
  donorColumns: number;
  /** The donor column picked for pseudobulk, used when the result names none. */
  replicateKey?: string;
  onGeneClick: (gene: string) => void;
}

const STYLE = { width: "100%" } as const;
const OBSERVED = "#dc2626";
const SHUFFLED = "#94a3b8";

/**
 * "Audit this result": the same cell-level test after shuffling the condition
 * across donors, and pseudobulk on the same cells (backend core/de_audit.py).
 * Says whether the cell-level genes survive donor-aware checks.
 */
export function DEAudit({ datasetId, result, betweenGroups, donorColumns, replicateKey, onGeneClick }: Props) {
  const [running, setRunning] = useState(false);
  const [start, setStart] = useState<DEAuditStart | null>(null);
  const [progress, setProgress] = useState<DEAuditProgress | null>(null);
  const [audit, setAudit] = useState<DEAuditResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  // A new result (or dataset) drops the old audit and stops a running one.
  useEffect(() => {
    setAudit(null);
    setError(null);
    return () => abort.current?.abort();
  }, [result.result_id, datasetId]);

  const blocker = auditBlocker(result, betweenGroups, donorColumns);
  const fewDonors = fewDonorsNote(result);

  const run = useCallback(async () => {
    if (!result.result_id) return;
    const ctrl = new AbortController();
    abort.current = ctrl;
    setRunning(true);
    setError(null);
    setAudit(null);
    setStart(null);
    setProgress(null);
    try {
      const res = await runDEAudit(
        datasetId,
        {
          result_id: result.result_id,
          ...(!result.replicate_key && replicateKey ? { replicate_key: replicateKey } : {}),
        },
        { onStart: setStart, onProgress: setProgress, signal: ctrl.signal },
      );
      setAudit(res);
    } catch (e) {
      if (!ctrl.signal.aborted) setError(errorMessage(e));
    } finally {
      if (abort.current === ctrl) abort.current = null;
      setRunning(false);
    }
  }, [datasetId, result.result_id, result.replicate_key, replicateKey]);

  const cancel = () => {
    abort.current?.abort();
    setRunning(false);
    setProgress(null);
  };

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-slate-200 p-2.5 text-xs" data-testid="de-audit">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={run}
          disabled={running || blocker != null}
          title={blocker ?? "Shuffle the condition across donors and rerun this test, and compare with pseudobulk"}
          className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
        >
          {running ? <Loader2 className="h-3 w-3 animate-spin" /> : <ShieldCheck className="h-3 w-3" />}
          Audit this result
        </button>
        <InfoTip label="About the DE audit" width={300}>
          A cell-level test counts every cell as a replicate, so differences between donors look like
          differences between the groups (Squair et al. 2021; Zimmerman et al. 2021). The audit reruns the
          same test after shuffling the group labels across whole donors (each shuffle keeps the donors
          and loses the comparison), and runs pseudobulk, which tests donors, on the same cells. If the
          shuffles find about as many genes, or pseudobulk finds few of them, read the pseudobulk list.
        </InfoTip>
        {running && (
          <>
            <span className="text-slate-500" role="status">
              {progressText(start, progress)}
            </span>
            <button
              type="button"
              onClick={cancel}
              className="rounded-md border border-slate-300 bg-white px-2 py-0.5 text-slate-600 hover:bg-slate-50"
            >
              Cancel
            </button>
          </>
        )}
      </div>
      {blocker && <p className="text-[11px] text-slate-400">{blocker}</p>}
      {!blocker && !audit && !running && fewDonors && <p className="text-[11px] text-slate-500">{fewDonors}</p>}
      <RequestError message={error} />
      {audit && <AuditResultView audit={audit} onGeneClick={onGeneClick} />}
    </div>
  );
}

function AuditResultView({ audit, onGeneClick }: { audit: DEAuditResult; onGeneClick: (g: string) => void }) {
  const summary = auditSummary(audit);
  const bins = useMemo(() => nullBins(audit.null_counts, audit.observed), [audit]);

  const histData = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        x: bins.x,
        y: bins.y,
        width: bins.x.map(() => bins.width * 0.9),
        marker: { color: SHUFFLED },
        name: "Donor shuffles",
        hovertemplate: "%{y} shuffles near %{x:.0f} genes<extra></extra>",
      },
    ],
    [bins],
  );
  const histLayout = useMemo<Partial<Layout>>(
    () => ({
      autosize: true,
      height: 170,
      margin: { l: 36, r: 8, t: 8, b: 34 },
      showlegend: false,
      xaxis: { title: { text: `Genes at adj. p < ${audit.alpha}`, font: { size: 10 } }, range: bins.range, tickfont: { size: 9 }, fixedrange: true },
      yaxis: { title: { text: "Shuffles", font: { size: 10 } }, tickfont: { size: 9 }, fixedrange: true, rangemode: "tozero" },
      shapes: [
        { type: "line", xref: "x", yref: "paper", x0: audit.observed, x1: audit.observed, y0: 0, y1: 1, line: { color: OBSERVED, width: 2 } },
      ],
      annotations: [
        { xref: "x", yref: "paper", x: audit.observed, y: 1, text: "observed", showarrow: false, xanchor: "right", yanchor: "top", font: { size: 9, color: OBSERVED } },
      ],
    }),
    [audit, bins],
  );

  const qqData = useMemo<Data[]>(() => {
    const out: Data[] = [
      {
        type: "scatter",
        mode: "lines+markers",
        x: audit.qq.expected,
        y: audit.qq.observed,
        name: "Observed",
        marker: { size: 3, color: OBSERVED },
        line: { color: OBSERVED, width: 1 },
      },
    ];
    if (audit.qq.permuted)
      out.push({
        type: "scatter",
        mode: "lines+markers",
        x: audit.qq.expected,
        y: audit.qq.permuted,
        name: "Donor shuffles",
        marker: { size: 3, color: SHUFFLED },
        line: { color: SHUFFLED, width: 1 },
      });
    return out;
  }, [audit]);
  const qqLayout = useMemo<Partial<Layout>>(() => {
    const top = Math.max(1, ...audit.qq.expected);
    return {
      autosize: true,
      height: 170,
      margin: { l: 36, r: 8, t: 8, b: 34 },
      legend: { orientation: "h", x: 0, y: 1.12, font: { size: 9 } },
      xaxis: { title: { text: "Expected -log10 p (uniform)", font: { size: 10 } }, tickfont: { size: 9 } },
      yaxis: { title: { text: "-log10 p", font: { size: 10 } }, tickfont: { size: 9 } },
      shapes: [{ type: "line", x0: 0, y0: 0, x1: top, y1: top, line: { color: "#cbd5e1", width: 1, dash: "dot" } }],
    };
  }, [audit]);

  const pb = audit.pseudobulk;
  return (
    <div className="flex flex-col gap-2">
      <p role="status" className="rounded border border-slate-200 bg-slate-50 px-2 py-1.5 text-slate-700" data-testid="de-audit-verdict">
        {audit.verdict}
      </p>
      {audit.permutation_note && <p className="text-[11px] text-amber-700">{audit.permutation_note}</p>}
      {audit.null_counts.length > 0 && (
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="min-w-0">
            <Plot data={histData} layout={histLayout} config={plotConfig("de_audit_shuffles")} useResizeHandler style={STYLE} />
            <Caption n={[{ value: audit.null_counts.length, unit: "donor shuffles" }, { value: audit.n_genes_tested, unit: "genes tested" }]}>
              Genes the same test calls after each shuffle ({schemeLabel(audit.scheme)}): {summary.shuffles}; red line: the
              real labels ({audit.observed.toLocaleString()}){summary.empirical ? `, ${summary.empirical}` : ""}.{" "}
              <InfoTip label="About the shuffles">
                Shuffles are balanced: each side gets the same mix of real {String(audit.comparison.group_a ?? "A")} and{" "}
                {String(audit.comparison.group_b ?? "B")} samples, so a real effect cancels and what is left is donor
                differences. With an odd number of donors a small part of a real effect stays. The verdict calls the
                shuffles &ldquo;as many&rdquo; when their median reaches {Math.round(audit.null_share * 100)}% of the
                observed count.
              </InfoTip>
            </Caption>
          </div>
          <div className="min-w-0">
            <Plot data={qqData} layout={qqLayout} config={plotConfig("de_audit_qq")} useResizeHandler style={STYLE} />
            <Caption n={{ value: audit.n_genes_tested, unit: "genes" }}>
              Quantile-quantile plot of p-values: observed and all shuffles pooled, against what no effect would give
              (dotted). Shuffles above the dotted line show how far the cell-level test overstates evidence here.
            </Caption>
          </div>
        </div>
      )}
      {audit.stopped && <p className="text-[11px] text-amber-700">Shuffles {audit.stopped}.</p>}
      {pb ? (
        <p className="text-[11px] text-slate-600">
          {concordanceText(pb, audit.alpha)}{" "}
          <InfoTip label="About the concordance">
            Jaccard: genes significant in both over genes significant in either (1 = the same list). Spearman: rank
            correlation of the two fold changes over the genes both tested. Pseudobulk sums each donor&apos;s counts and
            tests donors, not cells.
          </InfoTip>
        </p>
      ) : (
        audit.pseudobulk_note && <p className="text-[11px] text-amber-700">No pseudobulk: {audit.pseudobulk_note}</p>
      )}
      {audit.notes.map((n) => (
        <p key={n} className="text-[11px] text-slate-500">
          {n}
        </p>
      ))}
      {pb && (
        <>
          <Caption n={{ value: audit.n_cell_only, unit: "genes" }}>
            Significant at cell level, not in pseudobulk: the likely false positives. Top donor: the share of the
            difference lost when that donor is left out.{" "}
            <InfoTip label="About the top-donor share">
              Near 0%: the donors carry the difference evenly, so the gene is more likely a small effect that only
              thousands of cells make significant (or a normalisation shift). Near 100%: one donor carries it.
            </InfoTip>
          </Caption>
          <CapNote shown={audit.cell_only.length} total={audit.n_cell_only} by="cell-level p" rest="the results export has these" />
          {audit.cell_only.length > 0 && (
            <div className="max-h-56 overflow-auto rounded border border-slate-200">
              <table className="w-full text-[11px]">
                <thead className="sticky top-0 bg-slate-50">
                  <tr className="border-b border-slate-200 text-slate-600">
                    <th className="px-1.5 py-1 text-left font-medium" title="Gene symbol; click a row to colour the scatter by it">Gene</th>
                    <th className="px-1.5 py-1 text-right font-medium" title="Cell-level log2 fold change">Cell log2FC</th>
                    <th className="px-1.5 py-1 text-right font-medium" title="Cell-level adjusted p">Cell adj.P</th>
                    <th className="px-1.5 py-1 text-right font-medium" title="Pseudobulk log2 fold change (– : pseudobulk gave no p-value: filtered out, or an outlier sample)">PB log2FC</th>
                    <th className="px-1.5 py-1 text-right font-medium" title="Pseudobulk adjusted p">PB adj.P</th>
                    <th className="px-1.5 py-1 text-right font-medium" title="Share of the difference lost when this donor is left out">Top donor</th>
                  </tr>
                </thead>
                <tbody>
                  {audit.cell_only.map((g) => (
                    <tr
                      key={g.gene}
                      onClick={() => onGeneClick(g.gene)}
                      className="cursor-pointer border-b border-slate-50 hover:bg-blue-50"
                    >
                      <td className="px-1.5 py-1 font-mono text-slate-800">{g.gene}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{g.cell_lfc.toFixed(2)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{formatPValue(g.cell_padj)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{g.pb_lfc == null ? "–" : g.pb_lfc.toFixed(2)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{g.pb_padj == null ? "–" : formatPValue(g.pb_padj)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{shareText(g.top_donor_share, g.top_donor)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
      <p className="text-[10px] text-slate-400">
        {audit.label}; donors from <code>{audit.donor_key}</code> ({audit.n_donors_a} vs {audit.n_donors_b}); {audit.seconds} s. Kept in
        History &gt; Results.
      </p>
    </div>
  );
}
