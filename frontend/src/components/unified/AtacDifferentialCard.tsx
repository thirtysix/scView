import { useEffect, useMemo, useState } from "react";
import { daCsvUrl, getDaPage, getLatestDa, type AtacDaRow, type AtacState } from "@/api/atac";
import { CapNote, Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { formatDistance, formatP } from "@/lib/atac";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { AtacRunStatus } from "./AtacRunStatus";

const PAGE = 100;

/** Differential peaks per group (Wilcoxon on TF-IDF), with the nearest gene of each. */
export function AtacDifferentialCard({
  datasetId,
  state,
  columns,
  defaultColumn,
}: {
  datasetId: string;
  state: AtacState;
  columns: string[];
  defaultColumn: string;
}) {
  const run = useAtacRunStore();
  const setDa = useAtacRunStore.setState;
  const live = run.datasetId === datasetId && run.running;
  const elsewhere = !live && !!state.running;
  const [column, setColumn] = useState(defaultColumn);
  const [group, setGroup] = useState("");
  const [rows, setRows] = useState<AtacDaRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const da = run.da;

  useEffect(() => {
    if (defaultColumn) setColumn(defaultColumn);
  }, [defaultColumn]);

  // The last result on these cells, when this tab opens with none in hand.
  useEffect(() => {
    let cancelled = false;
    getLatestDa(datasetId)
      .then((d) => {
        if (!cancelled) setDa({ da: d });
      })
      .catch(() => {
        if (!cancelled) setDa({ da: null });
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, setDa]);

  const shownGroup = da && da.groups.includes(group) ? group : da?.groups[0] ?? "";
  const first = useMemo(() => da?.rows[shownGroup] ?? [], [da, shownGroup]);
  useEffect(() => setRows(null), [da?.id, shownGroup]);
  const table = rows ?? first;
  const kept = da ? Math.min(da.kept_per_group, da.n_tested[shownGroup] ?? 0) : 0;

  const more = async () => {
    if (!da) return;
    try {
      const page = await getDaPage(datasetId, da.id, shownGroup, Math.min(1000, table.length + PAGE * 4));
      setRows(page.rows[shownGroup] ?? []);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Differential peaks">
      <h4 className="flex items-center gap-1 text-xs font-semibold text-slate-600">
        <span>Differential peaks</span>
        <InfoTip label="About differential peaks" width={300}>
          Wilcoxon rank-sum on TF-IDF, each group against the rest, for peaks open in at least 5% of either; FDR within each
          group. Signac recommends logistic regression with the cell&rsquo;s total as a covariate, about 144k fits per group; this is
          the fast default.
        </InfoTip>
      </h4>
      <div className="flex flex-wrap items-center gap-1.5">
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
          Groups from
          <select value={column} onChange={(e) => setColumn(e.target.value)}
                  className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
            <option value="">Choose a column</option>
            {columns.map((c) => <option key={c} value={c}>{prettyObsLabel(c)}</option>)}
          </select>
        </label>
        {!live && (
          <button type="button" disabled={!column || run.running || elsewhere} onClick={() => void run.runDa(datasetId, { column })}
                  className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50">
            Find differential peaks
          </button>
        )}
      </div>
      <AtacRunStatus datasetId={datasetId} kind="differential" />
      <RequestError message={error} />
      {da && (
        <div className="space-y-1.5" data-testid="atac-da">
          <div className="flex flex-wrap items-center gap-1.5">
            <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
              Group
              <select value={shownGroup} onChange={(e) => setGroup(e.target.value)}
                      className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
                {da.groups.map((g) => (
                  <option key={g} value={g}>{g} ({(da.n_significant[g] ?? 0).toLocaleString()} at FDR 5%)</option>
                ))}
              </select>
            </label>
            <a href={daCsvUrl(datasetId, da.id)} download className="text-[11px] text-primary hover:underline">CSV</a>
          </div>
          {da.note && <p className="text-[11px] leading-snug text-amber-700">{da.note}</p>}
          <div className="max-h-80 overflow-auto">
            <table className="w-full text-[11px] text-slate-600">
              <thead className="sticky top-0 bg-white text-left text-slate-500">
                <tr>
                  <th className="font-medium">Peak</th>
                  <th className="font-medium" title="The gene whose TSS is nearest; pseudogenes and small RNAs left out">Nearest gene</th>
                  <th className="text-right font-medium" title="To that gene's TSS: + downstream, - upstream on its strand">To TSS</th>
                  <th className="font-medium" title="Promoter: within 2 kb of a TSS; gene body; distal">Class</th>
                  <th className="text-right font-medium" title="log2 fold change of mean TF-IDF (expm1) + 1">log2FC</th>
                  <th className="text-right font-medium" title="Share of the group's cells with the peak">% in</th>
                  <th className="text-right font-medium" title="Share of the other cells with the peak">% rest</th>
                  <th className="text-right font-medium" title="Benjamini-Hochberg within this group">FDR</th>
                </tr>
              </thead>
              <tbody>
                {table.map((r) => (
                  <tr key={r.peak}>
                    <td className="whitespace-nowrap font-mono text-[10px]">{r.peak}</td>
                    <td>{r.nearest_gene || "–"}</td>
                    <td className="whitespace-nowrap text-right tabular-nums">{formatDistance(r.distance_to_tss)}</td>
                    <td className="whitespace-nowrap">{r.peak_class || "–"}</td>
                    <td className="text-right tabular-nums">{r.log2fc.toFixed(2)}</td>
                    <td className="text-right tabular-nums">{(100 * r.pct_in).toFixed(0)}</td>
                    <td className="text-right tabular-nums">{(100 * r.pct_rest).toFixed(0)}</td>
                    <td className="text-right tabular-nums">{formatP(r.pval_adj)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <CapNote shown={table.length} total={da.n_tested[shownGroup] ?? table.length} by="p" rest={`the CSV has the first ${kept.toLocaleString()}`} />
          {table.length < kept && table.length < 1000 && (
            <button type="button" onClick={() => void more()} className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary">
              Show more
            </button>
          )}
          <Caption n={{ value: da.n_tested[shownGroup] ?? 0, unit: "peaks tested" }}>
            {shownGroup} against the rest, by {prettyObsLabel(da.column)}. {da.method}
          </Caption>
        </div>
      )}
    </section>
  );
}
