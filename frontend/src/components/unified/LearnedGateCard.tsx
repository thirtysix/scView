import { useState } from "react";
import { Loader2 } from "lucide-react";
import { saveLearnedGate, type GateGene, type TrainResult } from "@/api/learnedGates";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { errorMessage } from "@/lib/errorMessage";
import { decodeSelection } from "@/lib/session";
import { useSelectionStore } from "@/stores/selectionStore";

const VERDICT_STYLE: Record<string, string> = {
  good: "bg-emerald-50 text-emerald-800",
  check: "bg-amber-50 text-amber-800",
  warn: "bg-red-50 text-red-800",
};

const pct = (x: number) => `${(100 * x).toFixed(0)}%`;

/**
 * A trained learned gate before it is kept: the verdict, the out-of-fold metrics with
 * their n, the genes for and against, the cells the model and the outline disagree on
 * (as a selection, to review them on the plot), and "Save to library".
 */
export function LearnedGateCard({
  result,
  nCells,
  onSaved,
}: {
  result: TrainResult;
  nCells: number;
  onSaved?: (name: string) => void;
}) {
  const { gate, report, overlap } = result;
  const cv = gate.cv;
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(result.saved);
  const [error, setError] = useState<string | null>(null);
  const nDisagree = report.disagree.n_missed + report.disagree.n_extra;

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await saveLearnedGate(gate.id);
      setSaved(true);
      onSaved?.(gate.name);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setSaving(false);
    }
  };

  const selectDisagreements = () => {
    const sel = report.disagree.cells;
    if ("filter" in sel) return;
    const { cells } = decodeSelection(sel, nCells);
    useSelectionStore.getState().setSelection(cells.size > 0 ? cells : null);
  };

  return (
    <div className="space-y-1.5 rounded border border-slate-200 bg-white p-2 text-[11px]" data-testid="learned-gate-card">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-semibold text-slate-700">Learned gate &ldquo;{gate.name}&rdquo;</span>
        <span className={`rounded px-1.5 py-0.5 ${VERDICT_STYLE[report.verdict.level] ?? ""}`} data-testid="learned-gate-verdict">
          {report.verdict.text}
        </span>
      </div>
      <table className="text-[11px]">
        <thead>
          <tr className="text-left text-[10px] text-slate-500">
            <th className="pr-3 font-medium" title="Area under the ROC curve of the out-of-fold gate scores: 1 = every selected cell scores above every other">
              AUROC
            </th>
            <th className="pr-3 font-medium" title="Of the cells the gate calls in, the share that were selected">
              Precision
            </th>
            <th className="pr-3 font-medium" title="Of the selected cells, the share the gate calls in">
              Recall
            </th>
            <th className="pr-3 font-medium" title="The balance of precision and recall; the in/out threshold maximises it">
              F1
            </th>
          </tr>
        </thead>
        <tbody>
          <tr className="tabular-nums text-slate-700">
            <td className="pr-3">{cv.auroc.toFixed(3)}</td>
            <td className="pr-3">{cv.precision.toFixed(2)}</td>
            <td className="pr-3">{cv.recall.toFixed(2)}</td>
            <td className="pr-3">{cv.f1.toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
      <Caption n={[{ value: cv.n_pos, unit: "selected" }, { value: cv.n_neg, unit: "other cells" }]}>
        Out-of-fold: each cell scored by a model that did not see it ({cv.folds} folds
        {cv.grouped ? ", grouped by sample" : ""}). Other cells come from{" "}
        {report.negatives === "all"
          ? "the whole dataset"
          : report.samples.length > 0
            ? `the ${report.samples.length} sample${report.samples.length === 1 ? "" : "s"} holding the selection`
            : "the rest of the dataset (no sample column)"}
        ; {gate.genes.length} genes kept of {report.n_candidates} candidates, threshold {gate.threshold.toFixed(2)}.
      </Caption>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <GeneList title="Genes for" genes={gate.top_for} testId="learned-gate-for" />
        <GeneList title="Genes against" genes={gate.top_against} testId="learned-gate-against" />
      </div>
      {overlap?.is_cluster && (
        <p className="text-slate-600" data-testid="learned-gate-cluster">
          {overlap.note}
        </p>
      )}
      {report.warnings.map((w) => (
        <p key={w} className="text-amber-700">
          {w}
        </p>
      ))}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={selectDisagreements}
          disabled={nDisagree === 0}
          className="rounded border border-slate-300 px-2 py-0.5 text-slate-700 hover:bg-slate-50 disabled:opacity-40"
          title="Make the cells where the model and the outline disagree the selection, to look at them on the plot"
        >
          Select disagreements ({nDisagree.toLocaleString()})
        </button>
        <span className="text-slate-500">
          {report.disagree.n_missed.toLocaleString()} selected but scored out, {report.disagree.n_extra.toLocaleString()} not
          selected but scored in (of {report.disagree.n_training.toLocaleString()} training cells)
        </span>
        <div className="flex-1" />
        {saved ? (
          <span className="text-emerald-700" role="status">
            In the library
          </span>
        ) : (
          <button
            onClick={() => void save()}
            disabled={saving}
            className="flex items-center gap-1 rounded bg-blue-600 px-2 py-0.5 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {saving && <Loader2 className="h-3 w-3 animate-spin" />}
            Save to library
          </button>
        )}
      </div>
      {error && (
        <p role="alert" className="text-red-600">
          {error}
        </p>
      )}
      <p className="flex items-center gap-1 text-[10px] text-slate-500">
        Gate score
        <InfoTip label="About the gate score" width={300}>
          {report.score_note} A learned gate is a lead for finding these cells elsewhere; check it on each dataset.
        </InfoTip>
        <span>· L1 logistic regression after CellTypist&rsquo;s recipe (Domínguez Conde et al. 2022, Science).</span>
      </p>
    </div>
  );
}

function GeneList({ title, genes, testId }: { title: string; genes: GateGene[]; testId: string }) {
  return (
    <div data-testid={testId}>
      <p className="text-[10px] font-medium text-slate-500">{title}</p>
      {genes.length === 0 ? (
        <p className="text-slate-400">none</p>
      ) : (
        <ul className="space-y-0.5">
          {genes.map((g) => (
            <li key={g.gene} className="flex gap-2 tabular-nums">
              <span className="w-20 truncate font-medium text-slate-700" title={g.gene}>
                {g.gene}
              </span>
              <span className="text-slate-500" title="Weight on the scaled gene">
                {g.weight > 0 ? "+" : ""}
                {g.weight.toFixed(2)}
              </span>
              <span className="text-slate-500" title="Share of selected / other training cells detecting the gene">
                {pct(g.pct_in)} / {pct(g.pct_out)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
