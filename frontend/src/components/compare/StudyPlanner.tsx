import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, AlertTriangle, ChevronRight, Loader2, Play, Sparkles, Square } from "lucide-react";
import { getDataset } from "@/api/datasets";
import {
  planPilot,
  runCompositionPlan,
  runDEPlan,
  type CompositionPlan,
  type DEPlan,
  type PilotInfo,
  type PilotRequest,
  type PlanProgress,
} from "@/api/planner";
import type { DatasetInfo } from "@/api/types";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { PowerCurves } from "@/components/compare/StudyPlannerPlot";
import { conditionColumns, suggestConditionColumn } from "@/lib/composition";
import { replicateColumns } from "@/lib/de";
import { prettyObsLabel } from "@/lib/formatting";
import { rankGroupingColumns } from "@/lib/obsColumns";
import {
  DEFAULT_CELLS,
  DEFAULT_DONORS,
  DEFAULT_FOLD_CHANGES,
  DE_DEFAULT_CELLS,
  DE_DEFAULT_LOG2FC,
  compositionCurves,
  compositionSentence,
  deCurves,
  deSentence,
  donorsNeeded,
  foldText,
  groupName,
  headlineDEPoint,
  log2fcText,
  lostGenesNote,
  parseNumbers,
  pctText,
  plannerGroupOrder,
  plannerQuestion,
  shareText,
} from "@/lib/studyPlanner";
import { useViewStore } from "@/stores/viewStore";
import { errorMessage } from "@/lib/errorMessage";

interface Props {
  datasetId: string;
  info?: DatasetInfo | null;
}

type Mode = "composition" | "de";

const selectCls = "rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-700";
const inputCls = "w-40 rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-700";
const chipCls = (on: boolean) =>
  `rounded-full border px-2.5 py-0.5 text-xs transition ${
    on ? "border-primary bg-primary/10 text-primary" : "border-slate-200 text-slate-500 hover:text-slate-700"
  }`;
const TARGETS = [0.5, 0.8, 0.9];

/**
 * "How many donors would a follow-up need?" Simulated from the open dataset's own
 * donor-to-donor variability (POST /planner/*): power to detect a shift in a cell
 * type's share with the app's composition test, or pseudobulk DE in one type, by
 * donors per group and cells per donor, with the assumptions spelled out.
 */
export function StudyPlanner({ datasetId, info }: Props) {
  const askCopilot = useViewStore((s) => s.askCopilot);
  const [fetched, setFetched] = useState<DatasetInfo | null>(null);
  const dataset = info ?? fetched;
  const [mode, setMode] = useState<Mode>("composition");
  const [groupKey, setGroupKey] = useState("");
  const [conditionKey, setConditionKey] = useState(""); // "" = none (donors only)
  const [sampleKey, setSampleKey] = useState(""); // "" = detected by the backend
  const [reference, setReference] = useState("");
  const [pilot, setPilot] = useState<PilotInfo | null>(null);
  const [pilotError, setPilotError] = useState<string | null>(null);
  const [groups, setGroups] = useState<string[]>([]);
  const [deGroup, setDeGroup] = useState("");
  const [foldsText, setFoldsText] = useState(DEFAULT_FOLD_CHANGES.join(", "));
  const [lfcText, setLfcText] = useState(DE_DEFAULT_LOG2FC.join(", "));
  const [donorsText, setDonorsText] = useState(DEFAULT_DONORS.join(", "));
  const [cellsText, setCellsText] = useState(DEFAULT_CELLS.join(", "));
  const [deCellsText, setDeCellsText] = useState(DE_DEFAULT_CELLS.join(", "));
  const [alpha, setAlpha] = useState(0.05);
  const [target, setTarget] = useState(0.8);
  const [design, setDesign] = useState<"auto" | "paired" | "unpaired">("auto");
  const [assumedSd, setAssumedSd] = useState("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<PlanProgress | null>(null);
  const [comp, setComp] = useState<CompositionPlan | null>(null);
  const [de, setDe] = useState<DEPlan | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    if (info) return;
    let cancelled = false;
    getDataset(datasetId)
      .then((d) => !cancelled && setFetched(d))
      .catch(() => !cancelled && setFetched(null));
    return () => {
      cancelled = true;
    };
  }, [datasetId, info]);
  useEffect(() => () => abort.current?.abort(), []);

  const cols = useMemo(() => dataset?.obs_columns ?? [], [dataset]);
  const groupOptions = useMemo(
    () =>
      plannerGroupOrder(
        rankGroupingColumns(cols, {
          nCells: dataset?.n_cells,
          activeClustering: dataset?.active_clustering,
          userColumns: dataset?.user_annotation_columns,
        }).map((g) => g.name),
        cols,
      ),
    [cols, dataset],
  );
  const conditionOptions = useMemo(() => conditionColumns(cols, [groupKey]), [cols, groupKey]);
  const sampleOptions = useMemo(
    () => replicateColumns(cols, [conditionKey, groupKey]),
    [cols, conditionKey, groupKey],
  );
  const conditionValues = useMemo(
    () => cols.find((c) => c.name === conditionKey)?.values ?? [],
    [cols, conditionKey],
  );

  // Defaults once the columns are known.
  useEffect(() => {
    if (!cols.length) return;
    const g = groupOptions[0] ?? "";
    setGroupKey(g);
    setConditionKey(suggestConditionColumn(conditionColumns(cols, [g])));
  }, [cols, groupOptions]);
  useEffect(() => {
    setReference("");
    setSampleKey("");
    setDesign("auto");
  }, [conditionKey]);

  const pilotReq = useMemo<PilotRequest | null>(() => {
    if (!groupKey) return null;
    return {
      group_key: groupKey,
      ...(conditionKey ? { condition_key: conditionKey } : {}),
      ...(sampleKey ? { sample_key: sampleKey } : {}),
      ...(conditionKey && reference ? { reference } : {}),
    };
  }, [groupKey, conditionKey, sampleKey, reference]);

  // The pilot (design, samples, shares) follows every column change; obs only.
  useEffect(() => {
    if (!pilotReq) return;
    let cancelled = false;
    planPilot(datasetId, pilotReq)
      .then((p) => {
        if (cancelled) return;
        setPilot(p);
        setPilotError(null);
        setGroups(p.suggested);
        const top = [...p.groups].sort((a, b) => b.share - a.share)[0];
        setDeGroup(top?.group ?? "");
        setAssumedSd(p.replicated ? "" : String(p.assumed_sd_default));
      })
      .catch((e) => {
        if (cancelled) return;
        setPilot(null);
        setPilotError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, pilotReq]);

  const donors = parseNumbers(donorsText, { integer: true, min: 2, max: 100 });
  const cells = parseNumbers(cellsText, { integer: true, min: 20, max: 100000 });
  const deCells = parseNumbers(deCellsText, { integer: true, min: 10, max: 100000 });
  const folds = parseNumbers(foldsText, { min: 0.05, max: 20 })?.filter((f) => f !== 1) ?? null;
  const lfcs = parseNumbers(lfcText, { min: 0.1, max: 5 });
  const sd = assumedSd.trim() ? Number(assumedSd) : null;
  const sdBad = sd != null && !(sd >= 0 && sd <= 5);
  const needsSd = pilot != null && !pilot.replicated;
  const paired = pilot?.design === "paired";

  const inputError =
    !donors ? "Donors per group: whole numbers from 2 to 100." :
    mode === "composition" && !cells ? "Cells per donor: whole numbers from 20 to 100,000." :
    mode === "composition" && !folds?.length ? "Fold changes: numbers from 0.05 to 20 (not 1)." :
    mode === "composition" && !groups.length ? "Pick at least one cell type." :
    mode === "composition" && needsSd && (sd == null || sdBad) ? "Enter a spread (log-odds SD from 0 to 5)." :
    mode === "de" && !deCells ? "Cells per donor: whole numbers from 10 to 100,000." :
    mode === "de" && !lfcs ? "log2 fold changes: numbers from 0.1 to 5." :
    mode === "de" && needsSd ? null : null;
  const canRun = !!pilotReq && !!pilot && !running && !inputError && !(mode === "de" && needsSd);

  const run = useCallback(() => {
    if (!pilotReq || !pilot || !donors) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setRunning(true);
    setRunError(null);
    setProgress(null);
    const handlers = { onProgress: setProgress, signal: ctrl.signal };
    const common = {
      ...pilotReq,
      donors,
      alpha,
      target_power: target,
      ...(design !== "auto" ? { design } : {}),
    };
    const job =
      mode === "composition"
        ? runCompositionPlan(
            datasetId,
            {
              ...common,
              groups,
              fold_changes: folds ?? DEFAULT_FOLD_CHANGES,
              cells: cells ?? DEFAULT_CELLS,
              ...(needsSd && sd != null ? { assumed_sd: sd } : {}),
            },
            handlers,
          ).then((r) => !ctrl.signal.aborted && setComp(r))
        : runDEPlan(
            datasetId,
            { ...common, group: deGroup, log2fc: lfcs ?? DE_DEFAULT_LOG2FC, cells: deCells ?? DE_DEFAULT_CELLS },
            handlers,
          ).then((r) => !ctrl.signal.aborted && setDe(r));
    job
      .catch((e) => {
        if (ctrl.signal.aborted) return;
        setRunError(errorMessage(e));
      })
      .finally(() => {
        if (abort.current === ctrl) {
          setRunning(false);
          setProgress(null);
        }
      });
  }, [pilotReq, pilot, donors, alpha, target, design, mode, datasetId, groups, folds, cells, needsSd, sd, deGroup, lfcs, deCells]);

  const stop = useCallback(() => {
    abort.current?.abort();
    setRunning(false);
    setProgress(null);
  }, []);

  if (!dataset) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-sm text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading columns…
      </div>
    );
  }

  const result = mode === "composition" ? comp : de;

  return (
    <div className="space-y-4">
      <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex w-fit rounded-md border border-slate-200 p-0.5 text-xs">
            {(
              [
                ["composition", "Cell-type proportions"],
                ["de", "Pseudobulk DE"],
              ] as const
            ).map(([m, label]) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`rounded px-3 py-1 font-medium transition ${
                  mode === m ? "bg-primary text-white" : "text-slate-500 hover:text-slate-700"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <InfoTip width={320}>
            Each point simulates many studies of that size from this dataset&apos;s own donor-to-donor
            variability and runs on each the test the app itself would run (the composition test, or
            pseudobulk DESeq2). Power is the share of those studies that find the effect.
          </InfoTip>
        </div>

        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Cell types</span>
            <select aria-label="Planner cell type column" value={groupKey}
              onChange={(e) => setGroupKey(e.target.value)} className={selectCls}>
              {groupOptions.map((c) => <option key={c} value={c}>{prettyObsLabel(c)}</option>)}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Conditions</span>
            <select aria-label="Planner condition column" value={conditionKey}
              title="The pilot's conditions; without one, the donors of all cells are the pilot"
              onChange={(e) => setConditionKey(e.target.value)} className={selectCls}>
              <option value="">None (donors only)</option>
              {conditionOptions.map((c) => <option key={c.name} value={c.name}>{prettyObsLabel(c.name)}</option>)}
            </select>
          </label>
          {conditionValues.length > 0 && (
            <label className="text-xs font-medium text-slate-500">
              <span className="mb-1 block">Reference</span>
              <select aria-label="Planner reference" value={reference}
                title="The baseline condition: its shares are the starting point of the simulated studies"
                onChange={(e) => setReference(e.target.value)} className={selectCls}>
                <option value="">{pilot?.conditions[0] ? `Auto (${pilot.conditions[0]})` : "Auto"}</option>
                {conditionValues.map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
          )}
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Donors (replicates)</span>
            <select aria-label="Planner donor column" value={sampleKey}
              onChange={(e) => setSampleKey(e.target.value)} className={selectCls}>
              <option value="">
                {!sampleKey && pilot?.sample_key_detected && pilot.sample_key
                  ? `Auto (${prettyObsLabel(pilot.sample_key)})` : "Auto"}
              </option>
              {sampleOptions.map((c) => <option key={c.name} value={c.name}>{prettyObsLabel(c.name)}</option>)}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Design</span>
            <select aria-label="Planner design" value={design}
              title="Paired: every donor sampled in both conditions (only from a paired pilot). Unpaired: different donors per group."
              onChange={(e) => setDesign(e.target.value as "auto" | "paired" | "unpaired")} className={selectCls}>
              <option value="auto">{`Auto (${paired ? "paired" : "unpaired"})`}</option>
              <option value="paired" disabled={!paired}>Paired</option>
              <option value="unpaired">Unpaired</option>
            </select>
          </label>
        </div>

        {pilot && !pilot.sample_key && (
          <Caption testId="planner-pilot">
            Pilot: no donor or sample column, so all{" "}
            {pilot.groups.reduce((a, g) => a + g.n_cells, 0).toLocaleString()} cells
            count as one sample per condition.
          </Caption>
        )}
        {pilot && pilot.sample_key && (
          <Caption testId="planner-pilot">
            Pilot: {pilot.n_donors.toLocaleString()} donor{pilot.n_donors === 1 ? "" : "s"}
            {pilot.sample_key ? ` of ${prettyObsLabel(pilot.sample_key)}` : ""},{" "}
            {Object.entries(pilot.n_samples).map(([k, v]) => `${v} ${k}`).join(" and ")} samples
            {pilot.design === "paired" ? " (paired)" : pilot.design === "unpaired" ? " (unpaired)" : ""};{" "}
            {Math.round(pilot.cells_per_sample.median).toLocaleString()} cells per sample (median).
          </Caption>
        )}

        {mode === "composition" ? (
          <div className="space-y-2">
            {pilot && (
              <div className="flex flex-wrap items-center gap-1.5" aria-label="Planner cell types">
                <span className="text-xs text-slate-500">Plan for</span>
                {[...pilot.groups].sort((a, b) => b.share - a.share).map((g) => {
                  const on = groups.includes(g.group);
                  return (
                    <button key={g.group} className={chipCls(on)}
                      title={`${g.group}: ${shareText(g.share)} of the reference cells (${g.n_cells.toLocaleString()} cells in the pilot)`}
                      onClick={() => setGroups((gs) => (on ? gs.filter((x) => x !== g.group) : gs.length >= 6 ? gs : [...gs, g.group]))}>
                      {g.group} <span className="text-slate-400">{shareText(g.share)}</span>
                    </button>
                  );
                })}
              </div>
            )}
            <div className="flex flex-wrap items-end gap-3">
              <NumbersField label="Fold changes" value={foldsText} onChange={setFoldsText}
                title="The tested group's mean share of the type divided by the reference's (2 = doubles, 0.5 = halves); the other types make room" />
              <NumbersField label="Cells per donor" value={cellsText} onChange={setCellsText}
                title="Cells captured per sample (all types); one curve each" />
              <NumbersField label="Donors per group" value={donorsText} onChange={setDonorsText}
                title="The x axis: donors in each group (paired: each donor in both conditions)" />
              <Controls alpha={alpha} setAlpha={setAlpha} target={target} setTarget={setTarget} />
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-xs font-medium text-slate-500">
              <span className="mb-1 block">Cell type</span>
              <select aria-label="Planner DE cell type" value={deGroup}
                onChange={(e) => setDeGroup(e.target.value)} className={selectCls}>
                {[...(pilot?.groups ?? [])].sort((a, b) => b.share - a.share).map((g) => (
                  <option key={g.group} value={g.group}>{g.group} ({shareText(g.share)})</option>
                ))}
              </select>
            </label>
            <NumbersField label="log2 fold changes" value={lfcText} onChange={setLfcText}
              title="Size of the change in the changed genes (1 = 2-fold); 10% of the simulated genes change, half up, half down" />
            <NumbersField label="Cells of this type per donor" value={deCellsText} onChange={setDeCellsText}
              title="Cells of the chosen type in each sample (its pseudobulk); one curve each" />
            <NumbersField label="Donors per group" value={donorsText} onChange={setDonorsText}
              title="The x axis: donors in each group" />
            <Controls alpha={alpha} setAlpha={setAlpha} target={target} setTarget={setTarget} />
          </div>
        )}

        {needsSd && pilot && (
          <div role="alert" className="space-y-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5 text-xs text-amber-800" data-testid="planner-no-replicates">
            <p className="flex gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />{pilot.message}</p>
            {mode === "composition" ? (
              <label className="flex flex-wrap items-center gap-2">
                <span>Spread to plan with (donor-to-donor SD of the log-odds share), an ASSUMPTION:</span>
                <input aria-label="Assumed spread" value={assumedSd} onChange={(e) => setAssumedSd(e.target.value)}
                  className="w-20 rounded border border-amber-300 bg-white px-2 py-0.5 text-xs text-slate-700" />
                <span className="text-amber-700">Default {pilot.assumed_sd_default} (source above); a solid tissue usually varies more.</span>
              </label>
            ) : (
              <p>Pseudobulk DE power needs per-gene donor variability, which only a pilot with replicate donors has.</p>
            )}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <button onClick={run} disabled={!canRun}
            className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">
            {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Simulate
          </button>
          {running && (
            <button onClick={stop} className="flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-600 hover:bg-slate-50">
              <Square className="h-3 w-3" /> Stop
            </button>
          )}
          {running && progress && (
            <div className="flex items-center gap-2 text-xs text-slate-500" data-testid="planner-progress">
              <div className="h-1.5 w-40 overflow-hidden rounded bg-slate-100">
                <div className="h-full bg-primary" style={{ width: `${(100 * progress.done) / Math.max(progress.total, 1)}%` }} />
              </div>
              {progress.done} of {progress.total} grid points
            </div>
          )}
          {running && !progress && <span className="text-xs text-slate-500">Fitting the pilot…</span>}
          {inputError && <span className="text-xs text-red-600">{inputError}</span>}
        </div>
      </div>

      {pilotError && <ErrorBox text={pilotError} />}
      {runError && <ErrorBox text={runError} />}

      {result && !running && result.kind === "composition" && (
        <CompositionResult plan={result} target={target} onAsk={() => askCopilot(plannerQuestion(result))} />
      )}
      {result && !running && result.kind === "de" && (
        <DEResult plan={result} target={target} onAsk={() => askCopilot(plannerQuestion(result))} />
      )}
    </div>
  );
}

function NumbersField({ label, value, onChange, title }: { label: string; value: string; onChange: (v: string) => void; title: string }) {
  return (
    <label className="text-xs font-medium text-slate-500">
      <span className="mb-1 block" title={title}>{label}</span>
      <input aria-label={label} value={value} onChange={(e) => onChange(e.target.value)} className={inputCls} title={title} />
    </label>
  );
}

function Controls({ alpha, setAlpha, target, setTarget }: {
  alpha: number; setAlpha: (v: number) => void; target: number; setTarget: (v: number) => void;
}) {
  return (
    <>
      <label className="text-xs font-medium text-slate-500">
        <span className="mb-1 block" title="The adjusted p (FDR) a study must reach to count as finding the effect">Significance</span>
        <select aria-label="Planner alpha" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className={selectCls}>
          {[0.01, 0.05, 0.1].map((a) => <option key={a} value={a}>FDR &lt; {a}</option>)}
        </select>
      </label>
      <label className="text-xs font-medium text-slate-500">
        <span className="mb-1 block" title="The power the sentence answers aim at (the dotted line)">Target power</span>
        <select aria-label="Planner target power" value={target} onChange={(e) => setTarget(Number(e.target.value))} className={selectCls}>
          {TARGETS.map((t) => <option key={t} value={t}>{Math.round(t * 100)}%</option>)}
        </select>
      </label>
    </>
  );
}

function Header({ text, onAsk }: { text: string; onAsk: () => void }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <h4 className="text-sm font-semibold text-slate-700">{text}</h4>
      <button onClick={onAsk} title="Ask the co-pilot about this plan"
        className="ml-auto flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50 hover:text-primary">
        <Sparkles className="h-3 w-3" /> Ask
      </button>
    </div>
  );
}

function CompositionResult({ plan, target, onAsk }: { plan: CompositionPlan; target: number; onAsk: () => void }) {
  const [fc, setFc] = useState(plan.fold_changes.find((f) => f === 2) ?? plan.fold_changes[0]!);
  useEffect(() => {
    if (!plan.fold_changes.includes(fc)) setFc(plan.fold_changes[0]!);
  }, [plan, fc]);
  const types = plan.types.filter((t) => t.targeted);
  return (
    <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <Header onAsk={onAsk}
        text={`Power by donors per group (${plan.design} design${plan.source === "assumed" ? ", ENTERED spread" : ""})`} />
      {plan.notes.map((n) => <WarnBox key={n} text={n} />)}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs text-slate-500">Effect</span>
        {plan.fold_changes.map((f) => (
          <button key={f} className={chipCls(f === fc)} onClick={() => setFc(f)}>×{f}</button>
        ))}
      </div>
      <div className="grid gap-4 [grid-template-columns:repeat(auto-fit,minmax(320px,1fr))]">
        {types.map((t) => (
          <div key={t.group} className="space-y-1" data-testid="planner-type">
            <p className="text-sm text-slate-700" data-testid="planner-sentence">
              {compositionSentence(plan, t.group, fc, target)}
            </p>
            <PowerCurves curves={compositionCurves(plan, t.group, fc)} target={target}
              title={`${groupName(t.group)}: ${foldText(fc)}`} unit="cells per donor"
              filename={`study_plan_${t.group}_x${fc}`} />
            <Caption n={{ value: plan.reps, unit: "simulated studies per point" }}>
              Power to detect {foldText(fc)} in {groupName(t.group)}&apos;s share ({shareText(t.share)}
              {plan.reference ? ` in ${plan.reference}` : ""}) at adjusted p &lt; {plan.alpha}; one line per cells per
              donor{plan.source === "pilot" ? `; dashed: the spread at the upper end of its 95% interval (SD ${t.sd_upper_used?.toFixed(2)} instead of ${t.sd_used.toFixed(2)})` : ""}; dotted: {pctText(target)}.
            </Caption>
          </div>
        ))}
      </div>
      <NeededTable plan={plan} target={target} />
      <Assumptions items={plan.assumptions} elapsed={plan.elapsed_s} />
    </div>
  );
}

function NeededTable({ plan, target }: { plan: CompositionPlan; target: number }) {
  const types = plan.types.filter((t) => t.targeted);
  const maxN = Math.max(...plan.donors);
  const cell = (pts: { donors: number; power: number | null }[], feasible: boolean) => {
    if (!feasible) return "not possible";
    const n = donorsNeeded(pts, target);
    return n == null ? `> ${maxN}` : String(n);
  };
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs" data-testid="planner-needed">
        <thead>
          <tr className="border-b border-slate-100 bg-slate-50 text-left text-slate-600">
            <th className="px-3 py-1.5 font-medium" title="The planned cell type or cluster">{prettyObsLabel(plan.group_key)}</th>
            <th className="px-3 py-1.5 font-medium" title="The tested group's mean share divided by the reference's">Effect</th>
            <th className="px-3 py-1.5 text-right font-medium"
              title="Donor-to-donor SD of the type's log-odds share the simulation uses (within donor for a paired design, total for unpaired), estimated from the pilot with counting noise removed">
              Spread (SD)
            </th>
            {plan.cells.map((m) => (
              <th key={m} className="px-3 py-1.5 text-right font-medium"
                title={`Donors per group for ${pctText(target)} power at ${m.toLocaleString()} cells per donor (interpolated between grid values, rounded up)`}>
                {m.toLocaleString()} cells
              </th>
            ))}
            {plan.source === "pilot" && (
              <th className="px-3 py-1.5 text-right font-medium"
                title={`Donors per group at ${plan.middle_cells.toLocaleString()} cells if the spread is at the upper end of its 95% interval`}>
                Upper spread
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {types.flatMap((t) =>
            plan.fold_changes.map((f) => {
              const pts = (m: number, pess: boolean) =>
                plan.points.filter((p) => p.group === t.group && p.fold_change === f && p.pessimistic === pess && (pess || p.cells === m));
              return (
                <tr key={`${t.group}-${f}`} className="border-b border-slate-50">
                  <td className="px-3 py-1 text-slate-800">{t.group}</td>
                  <td className="px-3 py-1 text-slate-600">×{f}</td>
                  <td className="px-3 py-1 text-right tabular-nums text-slate-600">{t.sd_used.toFixed(2)}</td>
                  {plan.cells.map((m) => {
                    const p = pts(m, false);
                    return <td key={m} className="px-3 py-1 text-right tabular-nums text-slate-700">{cell(p, p.some((x) => x.power != null))}</td>;
                  })}
                  {plan.source === "pilot" && (() => {
                    const p = pts(0, true);
                    return <td className="px-3 py-1 text-right tabular-nums text-slate-500">{cell(p, p.some((x) => x.power != null))}</td>;
                  })()}
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
      <Caption>Donors per group needed for {pctText(target)} power; &gt; {maxN}: not reached on this grid.</Caption>
    </div>
  );
}

function DEResult({ plan, target, onAsk }: { plan: DEPlan; target: number; onAsk: () => void }) {
  const [L, setL] = useState(plan.log2fc.includes(1) ? 1 : plan.log2fc[0]!);
  useEffect(() => {
    if (!plan.log2fc.includes(L)) setL(plan.log2fc[0]!);
  }, [plan, L]);
  const lost = lostGenesNote(plan, L);
  const head = headlineDEPoint(plan, L, target);
  const maxN = Math.max(...plan.donors);
  return (
    <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <Header onAsk={onAsk} text={`Pseudobulk DE in ${plan.group}: share of changed genes found (${plan.design} design)`} />
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs text-slate-500">Effect</span>
        {plan.log2fc.map((x) => (
          <button key={x} className={chipCls(x === L)} onClick={() => setL(x)}>{log2fcText(x)}</button>
        ))}
      </div>
      <p className="text-sm text-slate-700" data-testid="planner-sentence">{deSentence(plan, L, target)}</p>
      {lost && <p className="text-xs text-slate-600">{lost}</p>}
      <PowerCurves curves={deCurves(plan, L)} target={target} title={`${plan.group}: ${log2fcText(L)}`}
        unit={`${plan.group} cells per donor`} filename={`study_plan_de_${plan.group}_L${L}`} />
      <Caption n={{ value: plan.reps, unit: "simulated studies per point" }}>
        Share of the changed genes ({Math.round(plan.de_share * 100)}% of {plan.pilot.n_simulated.toLocaleString()} simulated
        genes, {log2fcText(L)}) found at FDR &lt; {plan.alpha} with the right sign; one line per {plan.group} cells per
        donor; dotted: {pctText(target)}.
        {head && head.power_by_expression.length === 3 &&
          ` At ${head.donors} donors and ${head.cells.toLocaleString()} cells: ${pctText(head.power_by_expression[0])} of the lowest-expressed third, ${pctText(head.power_by_expression[1])} of the middle, ${pctText(head.power_by_expression[2])} of the highest.`}
      </Caption>
      <div className="overflow-x-auto">
        <table className="w-full text-xs" data-testid="planner-needed">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50 text-left text-slate-600">
              <th className="px-3 py-1.5 font-medium" title="log2 fold change of the changed genes">Effect</th>
              {plan.cells.map((m) => (
                <th key={m} className="px-3 py-1.5 text-right font-medium"
                  title={`Donors per group to find ${pctText(target)} of the changed genes at ${m.toLocaleString()} ${plan.group} cells per donor`}>
                  {m.toLocaleString()} cells
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {plan.log2fc.map((x) => (
              <tr key={x} className="border-b border-slate-50">
                <td className="px-3 py-1 text-slate-700">{log2fcText(x)}</td>
                {plan.cells.map((m) => {
                  const n = donorsNeeded(plan.points.filter((p) => p.log2fc === x && p.cells === m), target);
                  return <td key={m} className="px-3 py-1 text-right tabular-nums text-slate-700">{n == null ? `> ${maxN}` : n}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <Caption>Donors per group needed to find {pctText(target)} of the changed genes; &gt; {maxN}: not reached on this grid.</Caption>
      </div>
      <Assumptions items={plan.assumptions} elapsed={plan.elapsed_s} />
    </div>
  );
}

function Assumptions({ items, elapsed }: { items: string[]; elapsed: number }) {
  return (
    <details className="group rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600" data-testid="planner-assumptions">
      <summary className="flex cursor-pointer list-none items-center gap-1 font-medium text-slate-700">
        <ChevronRight className="h-3.5 w-3.5 transition group-open:rotate-90" />
        What this assumes ({items.length}) · simulated in {elapsed.toFixed(0)} s
      </summary>
      <ul className="mt-2 list-disc space-y-1 pl-5">
        {items.map((a) => <li key={a}>{a}</li>)}
      </ul>
    </details>
  );
}

function ErrorBox({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <AlertCircle className="h-4 w-4 flex-shrink-0" />
      {text}
    </div>
  );
}

function WarnBox({ text }: { text: string }) {
  return (
    <div role="alert" className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
      <span>{text}</span>
    </div>
  );
}
