import { useEffect, useMemo, useState } from "react";
import { daCsvUrl, getDaEstimate, getDaPage, getLatestDa, type AtacDaEstimate, type AtacDaRow, type AtacDaTest, type AtacState } from "@/api/atac";
import { CapNote, Caption } from "@/components/common/Caption";
import { Help } from "@/components/common/Help";
import { HoverTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { daFitLabel, daFitText, daMostPass, daTestName, formatDistance, formatP } from "@/lib/atac";
import { formatDuration } from "@/lib/hvgSensitivity";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { AtacRunStatus } from "./AtacRunStatus";

const PAGE = 100;

/**
 * Differential peaks per group, each against the rest or one group against another (Signac's FindMarkers ident.1 / ident.2),
 * by Wilcoxon on TF-IDF or logistic regression with a depth covariate, with the nearest gene of each.
 */
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
  const [test, setTest] = useState<AtacDaTest>("lr");
  const [pair, setPair] = useState(false);
  const [groupA, setGroupA] = useState("");
  const [groupB, setGroupB] = useState("");
  const [groups, setGroups] = useState<string[]>([]);
  const [estimate, setEstimate] = useState<AtacDaEstimate | null>(null);
  const [group, setGroup] = useState("");
  const [rows, setRows] = useState<AtacDaRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const da = run.da;

  // The selects describe the result in view, so a reloaded table never sits under another column or test.
  const daId = da?.id;
  const daColumn = da?.column;
  const daTest = da?.test;
  const daVs = da?.vs ?? null;
  const daFirst = da?.groups[0] ?? "";
  const daColumnListed = daColumn != null && columns.includes(daColumn);
  useEffect(() => {
    if (daColumnListed) setColumn(daColumn);
    else if (!daId && defaultColumn) setColumn(defaultColumn);
  }, [daId, daColumn, daColumnListed, defaultColumn]);
  useEffect(() => {
    setTest(daId ? daTest ?? "wilcoxon" : "lr");
  }, [daId, daTest]);
  useEffect(() => {
    setPair(daVs != null);
    if (daVs != null) {
      setGroupA(daFirst);
      setGroupB(daVs);
    }
  }, [daId, daVs, daFirst]);

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

  // What the chosen test would need on this column, before it runs.
  useEffect(() => {
    let cancelled = false;
    setEstimate(null);
    if (!column) return;
    const ready = pair && groupA && groupB && groupA !== groupB;
    (ready ? getDaEstimate(datasetId, column, test, groupA, groupB) : getDaEstimate(datasetId, column, test))
      .then((e) => {
        if (cancelled) return;
        setEstimate(e);
        setGroups(e.groups ?? []);
      })
      .catch(() => {
        if (!cancelled) setEstimate(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, column, test, pair, groupA, groupB]);

  // A pair names groups of this column: the first two until the user picks.
  useEffect(() => {
    if (!groups.length) return;
    setGroupA((a) => (groups.includes(a) ? a : groups[0] ?? ""));
    setGroupB((b) => (groups.includes(b) ? b : groups[1] ?? ""));
  }, [groups]);
  const pairReady = pair && !!groupA && !!groupB && groupA !== groupB;
  const body = pairReady ? { column, test, groups: [groupA], vs: groupB } : { column, test };

  const shownGroup = da && da.groups.includes(group) ? group : da?.groups[0] ?? "";
  const first = useMemo(() => da?.rows[shownGroup] ?? [], [da, shownGroup]);
  useEffect(() => setRows(null), [da?.id, shownGroup]);
  const table = rows ?? first;
  const kept = da ? Math.min(da.kept_per_group, da.n_tested[shownGroup] ?? 0) : 0;
  const isLr = da?.test === "lr";
  const fitText = daFitText(da?.n_fit?.[shownGroup]);
  const against = da?.vs ?? null;
  const mostPass = da ? daMostPass(da.n_significant[shownGroup] ?? 0, da.n_tested[shownGroup] ?? 0) : false;

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
      <h4 className="text-xs font-semibold text-slate-600">
        <HoverTip
          width={320}
          tip={<>
            Each group against the rest, or one group against another (the cells of every other group left out, as Signac&rsquo;s
          FindMarkers with ident.1 and ident.2), for peaks open in at least 5% of either; FDR within each group. Logistic regression, the default:
            what Signac recommends (Seurat&rsquo;s LR test, group ~ peak + the cell&rsquo;s total counts against group ~ total counts).
            Every peak gets a score test against one fit per group, and the 10,000 highest of each group a true likelihood ratio; it costs
            minutes per column, and the line below says how many. Wilcoxon: rank-sum on TF-IDF, no covariate, faster, but in a low-depth
            group its calls follow depth.
          </>}
        >
          Differential peaks
        </HoverTip>
      </h4>
      <div className="flex flex-wrap items-center gap-1.5">
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
          <Help id="atac.da-groups">Groups from</Help>
          <select value={column} onChange={(e) => setColumn(e.target.value)}
                  className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
            <option value="">Choose a column</option>
            {columns.map((c) => <option key={c} value={c}>{prettyObsLabel(c)}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
          Test
          <select value={test} onChange={(e) => setTest(e.target.value as AtacDaTest)}
                  className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
            <option value="lr">Logistic regression (depth covariate)</option>
            <option value="wilcoxon">Wilcoxon (fast, no covariate)</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
          <Help id="atac.da-compare">Compare</Help>
          <select value={pair ? "pair" : "rest"} onChange={(e) => setPair(e.target.value === "pair")} aria-label="Compare"
                  className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
            <option value="rest">Each group vs the rest</option>
            <option value="pair" disabled={groups.length < 2}>One group vs another</option>
          </select>
        </label>
        {pair && (
          <span className="flex items-center gap-1 text-[11px] text-slate-600">
            <select value={groupA} onChange={(e) => setGroupA(e.target.value)} aria-label="Group"
                    className="max-w-[10rem] rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
              {groups.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
            vs
            <select value={groupB} onChange={(e) => setGroupB(e.target.value)} aria-label="Against"
                    className="max-w-[10rem] rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
              {groups.map((g) => <option key={g} value={g} disabled={g === groupA}>{g}</option>)}
            </select>
          </span>
        )}
        {!live && (
          <button type="button" disabled={!column || run.running || elsewhere || (pair && !pairReady)}
                  onClick={() => void run.runDa(datasetId, body)}
                  className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50">
            Find differential peaks
          </button>
        )}
      </div>
      {!live && estimate && (
        <p className="text-[11px] leading-snug text-slate-500" data-testid="atac-da-estimate">
          {estimate.test === "lr" ? "Logistic regression" : "Wilcoxon"}: about {formatDuration(estimate.estimate_seconds)} and{" "}
          {(estimate.estimate_bytes / 1024 ** 3).toFixed(1)} GB beyond the open dataset{" "}
          {estimate.vs ? `for ${estimate.group} vs ${estimate.vs} (${estimate.n_cells.toLocaleString()} cells).`
            : `for the ${estimate.n_groups} groups of this column.`}
        </p>
      )}
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
            <span className="text-[11px] text-slate-500">{daTestName(da.test)}</span>
          </div>
          {da.note && <p className="text-[11px] leading-snug text-amber-700">{da.note}</p>}
          <div className="max-h-80 overflow-auto">
            <table className="w-full min-w-max border-separate border-spacing-x-1 text-[11px] text-slate-600">
              <thead className="sticky top-0 bg-white text-left text-slate-500">
                <tr>
                  <th className="font-medium" title="The peak's place on the genome: chromosome, start and end">Peak</th>
                  <th className="font-medium" title="The gene whose TSS is nearest; pseudogenes and small RNAs left out">Nearest gene</th>
                  <th className="text-right font-medium" title="To that gene's TSS: + downstream, - upstream on its strand">To TSS</th>
                  <th className="pl-2 font-medium" title="Promoter: within 2 kb of a TSS; gene body; distal">Class</th>
                  <th className="text-right font-medium" title="log2 fold change of mean TF-IDF (expm1) + 1">log2FC</th>
                  <th className="text-right font-medium" title="Share of the group's cells with the peak">% in</th>
                  <th className="text-right font-medium" title={against ? `Share of the ${against} cells with the peak` : "Share of the other cells with the peak"}>
                    {against ? `% ${against.length > 10 ? "other" : against}` : "% rest"}
                  </th>
                  <th className="text-right font-medium" title="Benjamini-Hochberg within this group">FDR</th>
                  {isLr && <th className="pl-2 font-medium" title="LRT: a true likelihood ratio. score: the score test, for peaks below the fitted ones. Separated: the peak splits the group, p from the deviance's limit. No fit: the fit did not converge, so the score test.">p from</th>}
                </tr>
              </thead>
              <tbody>
                {table.map((r) => (
                  <tr key={r.peak}>
                    <td className="whitespace-nowrap font-mono text-[10px]">{r.peak}</td>
                    <td>{r.nearest_gene || "–"}</td>
                    <td className="whitespace-nowrap text-right tabular-nums">{formatDistance(r.distance_to_tss)}</td>
                    <td className="whitespace-nowrap pl-2">{r.peak_class || "–"}</td>
                    <td className="text-right tabular-nums">{r.log2fc.toFixed(2)}</td>
                    <td className="text-right tabular-nums">{(100 * r.pct_in).toFixed(0)}</td>
                    <td className="text-right tabular-nums">{(100 * r.pct_rest).toFixed(0)}</td>
                    <td className="whitespace-nowrap pl-1 text-right tabular-nums">{formatP(r.pval_adj)}</td>
                    {isLr && <td className="whitespace-nowrap pl-2">{daFitLabel(r.fit)}</td>}
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
            {shownGroup} against {against ?? "the rest"}, by {prettyObsLabel(da.column)}. {da.method}
            {fitText && ` ${fitText}`}
            {mostPass && " Most tested peaks pass here: with this many cells small differences reach significance, so read the top of the table, not the count."}
          </Caption>
        </div>
      )}
    </section>
  );
}
