import { useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { trainLearnedGate, type TrainResult } from "@/api/learnedGates";
import type { DatasetInfo } from "@/api/types";
import { InfoTip } from "@/components/common/InfoTip";
import { LearnedGateCard } from "@/components/unified/LearnedGateCard";
import { replicateColumns } from "@/lib/de";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { encodeSelection } from "@/lib/session";
import { useSelectionStore } from "@/stores/selectionStore";

const AUTO = "__auto__";

/**
 * "Learn gate…": train a learned gate on the current selection (a small model of the
 * genes that set these cells apart from the others in their samples), then review it
 * on the card and keep it in the library.
 */
export function LearnedGateForm({
  dataset,
  onCancel,
  onSaved,
}: {
  dataset: DatasetInfo;
  onCancel: () => void;
  onSaved?: (name: string) => void;
}) {
  const selected = useSelectionStore((s) => s.selectedCellIndices);
  const nCells = dataset.n_cells ?? 0;
  const options = useMemo(() => replicateColumns(dataset.obs_columns, []), [dataset.obs_columns]);
  const [name, setName] = useState("");
  const [sample, setSample] = useState(AUTO);
  const [excludeCc, setExcludeCc] = useState(true);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TrainResult | null>(null);
  const abort = useRef<AbortController | null>(null);
  const nSel = selected?.size ?? 0;
  const nameProblem = name.trim() ? null : "Give the gate a name";

  const run = async () => {
    if (!selected || nameProblem) return;
    setRunning(true);
    setError(null);
    setResult(null);
    setProgress("Reading the cells' values");
    abort.current = new AbortController();
    try {
      const r = await trainLearnedGate(
        dataset.id,
        {
          name: name.trim(),
          cells: encodeSelection(selected, nCells),
          sample_column: sample === AUTO ? null : sample,
          exclude_cc_stress: excludeCc,
        },
        {
          signal: abort.current.signal,
          onStart: (s) =>
            setProgress(
              `${s.n_pos.toLocaleString()} selected vs ${s.n_neg.toLocaleString()} other cells, ${s.n_candidates} candidate genes`,
            ),
          onProgress: (p) => setProgress(`Fitting ${p.done} of ${p.total} (${p.elapsed_s.toFixed(0)} s)`),
        },
      );
      setResult(r);
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(errorMessage(e));
    } finally {
      setRunning(false);
      setProgress("");
      abort.current = null;
    }
  };

  return (
    <div className="space-y-1.5 rounded border border-slate-200 bg-slate-50 p-2 text-[11px]" data-testid="learn-gate-form">
      <p className="flex items-center gap-1 text-slate-600">
        Learn a gate from the {nSel.toLocaleString()} selected cells
        <InfoTip label="About learned gates" width={320}>
          A learned gate is a small model (an L1 logistic regression on up to 50 genes) of what sets the selected cells
          apart from the other cells in the same samples. Unlike a saved gate, it carries to other datasets: apply it
          from Learned gates. Other cells come only from the samples (and conditions) that hold the selection, so a
          lasso on control cells does not learn the stimulation.
        </InfoTip>
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1 text-slate-600">
          Name
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={60}
            placeholder="e.g. B cells"
            className="w-36 rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] focus:border-blue-400 focus:outline-none"
          />
        </label>
        <label className="flex items-center gap-1 text-slate-600">
          Samples
          <select
            value={sample}
            onChange={(e) => setSample(e.target.value)}
            className="max-w-[160px] rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700 focus:border-blue-400 focus:outline-none"
            title="The column naming samples: other cells come from the samples holding the selection"
          >
            <option value={AUTO}>pick for me</option>
            <option value="">none (all other cells)</option>
            {options.map((c) => (
              <option key={c.name} value={c.name}>
                {prettyObsLabel(c.name)}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1 text-slate-600" title="Leave the Tirosh 2016 cell-cycle genes and the dissociation-stress genes out of the model">
          <input type="checkbox" checked={excludeCc} onChange={(e) => setExcludeCc(e.target.checked)} />
          Leave out cell-cycle and stress genes
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {running ? (
          <>
            <Loader2 className="h-3 w-3 animate-spin text-slate-500" />
            <span className="text-slate-600" role="status">
              {progress}
            </span>
            <button
              onClick={() => abort.current?.abort()}
              className="rounded border border-slate-300 px-2 py-0.5 text-slate-700 hover:bg-white"
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <button
              onClick={() => void run()}
              disabled={!selected || nSel === 0 || !!nameProblem}
              title={nameProblem ?? "Train the gate (about 10-30 s)"}
              className="rounded bg-blue-600 px-2 py-0.5 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              Train
            </button>
            <button onClick={onCancel} className="rounded px-2 py-0.5 text-slate-600 hover:bg-white">
              Close
            </button>
          </>
        )}
      </div>
      {error && (
        <p role="alert" className="text-red-600">
          {error}
        </p>
      )}
      {result && <LearnedGateCard result={result} nCells={nCells} onSaved={onSaved} />}
    </div>
  );
}
