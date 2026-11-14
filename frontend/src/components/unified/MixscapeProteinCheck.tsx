import { useMemo, useState } from "react";
import Plot from "@/components/plots/Plot";
import { Caption } from "@/components/common/Caption";
import { Help } from "@/components/common/Help";
import type { ProteinCheck } from "@/api/perturbation";
import { formatPValue } from "@/lib/formatting";
import { pctText, proteinRowsFor, proteinShiftTraces, replicateText, shiftText } from "@/lib/perturbation";
import { plotConfig } from "@/lib/plotConfig";

const chip = (on: boolean) =>
  `rounded border px-1.5 py-0.5 text-[11px] ${on ? "border-slate-600 bg-slate-600 text-white" : "border-slate-300 text-slate-600 hover:bg-slate-50"}`;

/**
 * A Mixscape run's protein check (backend core/perturbation_protein.py): each protein's
 * shift per target in all, KO and NP cells against the controls, and the targets that
 * move a protein with no RNA KO class (Papalexi 2021: CMTM6 on PD-L1).
 */
export function MixscapeProteinCheck({
  check,
  onPickTarget,
}: {
  check: ProteinCheck;
  onPickTarget?: (target: string) => void;
}) {
  const [chosen, setChosen] = useState<string | null>(null);
  const protein = chosen && check.proteins.includes(chosen) ? chosen : (check.default_protein ?? check.proteins[0] ?? "");
  const rows = useMemo(() => proteinRowsFor(check, protein), [check, protein]);
  const traces = useMemo(() => proteinShiftTraces(rows, protein), [rows, protein]);
  const withReps = rows.some((r) => r.n_replicates);
  return (
    <section aria-label="Protein by target" className="space-y-1.5 border-t border-slate-200 pt-2" data-testid="mixscape-protein">
      <div className="font-medium text-slate-600">
        <Help id="perturbation.protein">Protein by target</Help>
      </div>
      {check.flags.length > 0 && (
        <div className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] text-amber-800" data-testid="mixscape-protein-flags">
          <div className="font-medium">
            <Help id="perturbation.protein_flag">Protein effect without an RNA KO class</Help>
          </div>
          <ul className="list-disc pl-4">
            {check.flags.map((f) => (
              <li key={`${f.target}|${f.protein}`}>{f.text}</li>
            ))}
          </ul>
        </div>
      )}
      {check.proteins.length > 1 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-[11px] text-slate-500">Protein</span>
          {check.proteins.map((p) => (
            <button key={p} aria-pressed={p === protein} onClick={() => setChosen(p)} className={chip(p === protein)}>
              {p}
            </button>
          ))}
        </div>
      )}
      <div style={{ height: 240 }}>
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { l: 44, r: 6, t: 6, b: 70 },
            showlegend: true,
            legend: { orientation: "h", y: 1.12, font: { size: 9 } },
            xaxis: { type: "category", categoryorder: "array", categoryarray: rows.map((r) => r.target), tickangle: -45, tickfont: { size: 9 } },
            yaxis: {
              title: { text: `${protein} shift vs ${check.control} (CLR)`, font: { size: 10 } },
              tickfont: { size: 9 },
              zeroline: true,
              zerolinecolor: "#94a3b8",
            },
          }}
          config={plotConfig(`mixscape_protein_${protein}`)}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>
      <div className="max-h-64 overflow-auto rounded-lg border border-slate-200">
        <table className="w-full text-[11px]" data-testid="mixscape-protein-table">
          <thead className="sticky top-0 bg-slate-50 text-slate-600">
            <tr className="border-b border-slate-200">
              <th className="px-1.5 py-1 text-left font-medium" title="The targeted gene">Target</th>
              <th className="px-1.5 py-1 text-right font-medium" title="Share of the target's cells Mixscape called KO (RNA)">KO</th>
              <th className="px-1.5 py-1 text-right font-medium" title={`Mean ${protein} (CLR) of all the target's guide cells minus the controls'`}>
                All cells
              </th>
              <th className="px-1.5 py-1 text-right font-medium" title="Wilcoxon rank-sum p of all guide cells vs the controls, Benjamini-Hochberg over the targets and proteins">
                Adj. p
              </th>
              <th className="px-1.5 py-1 text-right font-medium" title="Shift in the target's KO cells">KO cells</th>
              <th className="px-1.5 py-1 text-right font-medium" title="Shift in the target's NP (not perturbed) cells">NP cells</th>
              {withReps && (
                <th className="px-1.5 py-1 text-right font-medium" title="Replicates whose shift (all guide cells vs their own controls) has the same sign; paired t p over them">
                  Replicates
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.target}
                onClick={() => onPickTarget?.(r.target)}
                className={`cursor-pointer border-b border-slate-50 hover:bg-blue-50 ${r.flag ? "bg-amber-50" : ""}`}
                data-testid={`mixscape-protein-row-${r.target}`}
              >
                <td className="px-1.5 py-0.5 font-medium text-slate-800" title={r.note}>
                  {r.target}
                  {r.flag && <span className="ml-1 text-[10px] font-normal text-amber-700">protein only</span>}
                </td>
                <td className="px-1.5 py-0.5 text-right tabular-nums text-slate-600">{pctText(r.ko_fraction)}</td>
                <td
                  className={`px-1.5 py-0.5 text-right tabular-nums ${r.significant ? "font-medium text-slate-800" : "text-slate-500"}`}
                  title={r.all ? `${r.all.n.toLocaleString()} cells` : `fewer than ${check.min_cells} cells`}
                >
                  {shiftText(r.all?.diff)}
                </td>
                <td className="px-1.5 py-0.5 text-right tabular-nums text-slate-500">
                  {r.all?.padj != null ? formatPValue(r.all.padj) : "–"}
                </td>
                <td className="px-1.5 py-0.5 text-right tabular-nums text-slate-600" title={r.ko ? `${r.ko.n.toLocaleString()} cells` : "no KO cells"}>
                  {shiftText(r.ko?.diff)}
                </td>
                <td className="px-1.5 py-0.5 text-right tabular-nums text-slate-600" title={r.np ? `${r.np.n.toLocaleString()} cells` : "no NP cells"}>
                  {shiftText(r.np?.diff)}
                </td>
                {withReps && (
                  <td
                    className="px-1.5 py-0.5 text-right tabular-nums text-slate-600"
                    title={(r.replicates ?? []).map((p) => `${p.replicate}: ${shiftText(p.diff)} (${p.n} cells)`).join("\n")}
                  >
                    {replicateText(r)}
                    {r.p_replicate != null && <span className="ml-1 text-slate-400">p {formatPValue(r.p_replicate)}</span>}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Caption n={[{ value: check.n_control, unit: `'${check.control}' cells` }, { value: rows.length, unit: "targets" }]}>
        {protein} per target against the controls. {check.notes.join(" ")}
      </Caption>
    </section>
  );
}
