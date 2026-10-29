import { useCallback, useEffect, useRef, useState } from "react";
import { Download, Loader2, Trash2, Upload } from "lucide-react";
import {
  applyLearnedGate,
  checkLearnedGate,
  deleteLearnedGate,
  importLearnedGate,
  learnedGateFile,
  listLearnedGates,
  type GateApplyResult,
  type GateCheck,
  type LearnedGateSummary,
} from "@/api/learnedGates";
import type { DatasetInfo } from "@/api/types";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { errorMessage } from "@/lib/errorMessage";
import { downloadText } from "@/lib/geneLists";
import { useSettingsStore } from "@/stores/settingsStore";

const pct = (x: number, digits = 0) => `${(100 * x).toFixed(digits)}%`;
const VERDICT_TEXT: Record<string, string> = { good: "good fit", check: "check", warn: "poor fit" };
const VERDICT_STYLE: Record<string, string> = {
  good: "text-emerald-700",
  check: "text-amber-700",
  warn: "text-red-600",
};

/**
 * "Learned gates": the library of gates trained here or imported. Each can be exported
 * as a file, deleted, or applied to the open dataset, after a check of its genes,
 * species and warnings; applying writes an in/out column and a score.
 */
export function LearnedGatesLibrary({ dataset, onClose }: { dataset: DatasetInfo; onClose: () => void }) {
  const [gates, setGates] = useState<LearnedGateSummary[] | null>(null);
  const [scoreNote, setScoreNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [checking, setChecking] = useState<string | null>(null);
  const [check, setCheck] = useState<{ id: string; report: GateCheck } | null>(null);
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState<GateApplyResult | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const refresh = useDatasetRefresh();

  const load = useCallback(async () => {
    try {
      const r = await listLearnedGates();
      setGates(r.gates);
      setScoreNote(r.score_note);
    } catch (e) {
      setError(errorMessage(e));
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);

  const doCheck = async (g: LearnedGateSummary) => {
    setChecking(g.id);
    setError(null);
    setApplied(null);
    setMessage(null);
    try {
      setCheck({ id: g.id, report: await checkLearnedGate(dataset.id, g.id) });
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setChecking(null);
    }
  };

  const doApply = async () => {
    if (!check) return;
    setApplying(true);
    setError(null);
    try {
      const r = await applyLearnedGate(dataset.id, check.id, check.report.sample_column ?? null);
      setApplied(r);
      setCheck(null);
      await refresh(dataset.id, r.column);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setApplying(false);
    }
  };

  const doExport = async (g: LearnedGateSummary) => {
    try {
      const text = await learnedGateFile(g.id);
      const safe = g.name.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_|_$/g, "") || "gate";
      downloadText(`learned_gate_${safe}.json`, text, "application/json");
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const doDelete = async (g: LearnedGateSummary) => {
    try {
      await deleteLearnedGate(g.id);
      setMessage(`Deleted “${g.name}” from the library (columns it wrote stay).`);
      if (check?.id === g.id) setCheck(null);
      await load();
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const doImport = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    try {
      const g = await importLearnedGate(await file.text());
      setMessage(`Imported “${g.name}”.`);
      await load();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  return (
    <div className="space-y-1.5 rounded border border-slate-200 bg-slate-50 p-2 text-[11px]" data-testid="learned-gates-library">
      <div className="flex flex-wrap items-center gap-2">
        <span className="flex items-center gap-1 font-semibold text-slate-700">
          Learned gates
          <InfoTip label="About learned gates" width={320}>
            Gates learned from a selection (&ldquo;Learn gate&hellip;&rdquo;) or imported from a file. Applying one
            scores every cell of the open dataset and writes two columns: in / out, and the gate score. {scoreNote}
          </InfoTip>
        </span>
        <label className="flex cursor-pointer items-center gap-1 rounded px-2 py-0.5 text-slate-600 hover:bg-white">
          <Upload className="h-3 w-3" />
          Import file…
          <input
            ref={fileInput}
            type="file"
            accept=".json,application/json"
            className="sr-only"
            aria-label="Import a learned gate file"
            onChange={(e) => void doImport(e.target.files?.[0])}
          />
        </label>
        <div className="flex-1" />
        <button onClick={onClose} className="rounded px-2 py-0.5 text-slate-600 hover:bg-white">
          Close
        </button>
      </div>
      {gates === null && !error && (
        <p className="flex items-center gap-1 text-slate-500">
          <Loader2 className="h-3 w-3 animate-spin" /> Loading
        </p>
      )}
      {gates && gates.length === 0 && (
        <p className="text-slate-500">No learned gates yet. Select cells, then &ldquo;Learn gate&hellip;&rdquo;, or import a file.</p>
      )}
      {gates && gates.length > 0 && (
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-left text-[10px] text-slate-500">
              <th className="font-medium">Gate</th>
              <th className="font-medium" title="Out-of-fold AUROC and F1 on the training cells">
                Fit
              </th>
              <th className="font-medium" title="Genes the model uses">
                Genes
              </th>
              <th className="font-medium">Trained on</th>
              <th>
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {gates.map((g) => (
              <tr key={g.id} className="border-t border-slate-200" data-testid={`learned-gate-row-${g.name}`}>
                <td className="py-0.5 font-medium text-slate-700">{g.name}</td>
                {g.error ? (
                  <td colSpan={3} className="text-red-600">
                    {g.error}
                  </td>
                ) : (
                  <>
                    <td className="tabular-nums">
                      <span className={VERDICT_STYLE[g.verdict.level]}>{VERDICT_TEXT[g.verdict.level]}</span>{" "}
                      <span className="text-slate-500">
                        AUROC {g.cv.auroc.toFixed(3)}, F1 {g.cv.f1.toFixed(2)}
                      </span>
                    </td>
                    <td className="tabular-nums text-slate-600" title={g.top_for.join(", ")}>
                      {g.n_genes}
                    </td>
                    <td className="text-slate-600">
                      {g.source.name || "imported"} ({g.species}, {g.source.n_selected.toLocaleString()} cells)
                    </td>
                  </>
                )}
                <td className="py-0.5">
                  <div className="flex items-center justify-end gap-1">
                    {!g.error && (
                      <button
                        onClick={() => void doCheck(g)}
                        disabled={checking !== null || applying}
                        className="rounded border border-slate-300 bg-white px-1.5 py-0 text-slate-700 hover:bg-slate-100 disabled:opacity-40"
                      >
                        {checking === g.id ? "Checking…" : "Apply…"}
                      </button>
                    )}
                    {!g.error && (
                      <button
                        onClick={() => void doExport(g)}
                        aria-label={`Export ${g.name}`}
                        title="Download the gate file (JSON) to apply it in another scView"
                        className="rounded p-0.5 text-slate-400 hover:text-slate-600"
                      >
                        <Download className="h-3 w-3" />
                      </button>
                    )}
                    <button
                      onClick={() => void doDelete(g)}
                      aria-label={`Delete ${g.name}`}
                      title="Delete from the library"
                      className="rounded p-0.5 text-slate-400 hover:text-red-600"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {check && (
        <CheckPanel
          report={check.report}
          name={gates?.find((g) => g.id === check.id)?.name ?? ""}
          applying={applying}
          onApply={() => void doApply()}
          onCancel={() => setCheck(null)}
        />
      )}
      {applied && <ApplyResult result={applied} />}
      {message && (
        <p role="status" className="text-slate-600">
          {message}
        </p>
      )}
      {error && (
        <p role="alert" className="text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}

function CheckPanel({
  report,
  name,
  applying,
  onApply,
  onCancel,
}: {
  report: GateCheck;
  name: string;
  applying: boolean;
  onApply: () => void;
  onCancel: () => void;
}) {
  const cov = report.coverage;
  return (
    <div className="space-y-1 rounded border border-slate-200 bg-white p-2" data-testid="learned-gate-check" role="region" aria-label={`Check of ${name}`}>
      <p className="text-slate-700">
        <span className="font-medium">&ldquo;{name}&rdquo; on this dataset</span>
        {cov && (
          <>
            : {cov.n_found} of {cov.n_genes} genes found ({pct(cov.gene_share)}), {pct(cov.weight_share)} of the
            gate&rsquo;s weight
          </>
        )}
        . Gate {report.gate_species}, this dataset {report.species}
        {report.via ? `; genes matched by ${report.via}` : ""}.
      </p>
      {cov && cov.missing_top.length > 0 && <p className="text-slate-500">Missing: {cov.missing_top.join(", ")}</p>}
      {report.warnings.map((w) => (
        <p key={w} className="text-amber-700">
          {w}
        </p>
      ))}
      {report.refusal && (
        <p role="alert" className="text-red-600">
          {report.refusal}
        </p>
      )}
      {report.columns && report.ok && (
        <p className="text-slate-500">
          Writes {report.columns.column} (in / out) and {report.columns.score_column}
          {report.replaces ? ", replacing the columns this gate wrote before" : ""}.
        </p>
      )}
      <div className="flex gap-2">
        <button
          onClick={onApply}
          disabled={!report.ok || applying}
          className="flex items-center gap-1 rounded bg-blue-600 px-2 py-0.5 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {applying && <Loader2 className="h-3 w-3 animate-spin" />}
          Apply to this dataset
        </button>
        <button onClick={onCancel} className="rounded px-2 py-0.5 text-slate-600 hover:bg-slate-100">
          Cancel
        </button>
      </div>
    </div>
  );
}

function ApplyResult({ result }: { result: GateApplyResult }) {
  const colourBy = (c: string) => useSettingsStore.getState().setColorBy(c);
  return (
    <div className="space-y-1 rounded border border-slate-200 bg-white p-2" data-testid="learned-gate-applied">
      <Caption n={{ value: result.n_in, unit: "cells in the gate", of: result.n_cells }}>
        &ldquo;{result.name}&rdquo; applied ({pct(result.share, 1)} in; {result.coverage.n_found} of{" "}
        {result.coverage.n_genes} genes found).
      </Caption>
      {result.warnings.map((w) => (
        <p key={w} className="text-amber-700">
          {w}
        </p>
      ))}
      <div className="flex flex-wrap gap-2">
        <button className="rounded border border-slate-300 px-2 py-0.5 hover:bg-slate-50" onClick={() => colourBy(result.column)}>
          Colour by {result.column}
        </button>
        <button className="rounded border border-slate-300 px-2 py-0.5 hover:bg-slate-50" onClick={() => colourBy(result.score_column)}>
          Colour by gate score
        </button>
        <InfoTip label="About the gate score" width={300}>
          {result.score_note}
        </InfoTip>
      </div>
      {result.per_sample.length > 0 && (
        <>
          <table className="text-[11px]" data-testid="learned-gate-per-sample">
            <thead>
              <tr className="text-left text-[10px] text-slate-500">
                <th className="pr-3 font-medium">{result.sample_column}</th>
                <th className="pr-3 text-right font-medium">Cells</th>
                <th className="pr-3 text-right font-medium">In gate</th>
                <th className="pr-3 text-right font-medium">% in</th>
              </tr>
            </thead>
            <tbody>
              {result.per_sample.map((s) => (
                <tr key={s.sample} className="tabular-nums text-slate-700">
                  <td className="pr-3">{s.sample}</td>
                  <td className="pr-3 text-right">{s.n_cells.toLocaleString()}</td>
                  <td className="pr-3 text-right">{s.n_in.toLocaleString()}</td>
                  <td className="pr-3 text-right">{pct(s.share, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Caption n={{ value: result.per_sample.length, unit: "samples" }}>
            Cells in the gate per {result.sample_column}.
          </Caption>
        </>
      )}
    </div>
  );
}
