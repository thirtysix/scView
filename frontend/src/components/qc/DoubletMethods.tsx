import { useState } from "react";
import type { ClassifierCalls, DetectorAgreement } from "@/api/qc";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { ThresholdHistogram } from "@/components/plots/ThresholdHistogram";
import { classifierText, detectorAgreementText } from "@/lib/doubletMethods";
import { pct } from "@/lib/qcThresholds";

const CHANNEL_ROWS = 8; // more behind "show all"

const n = (v: number) => v.toLocaleString();

/**
 * The doublet step's second detector on the doublet card: the scDblFinder-style
 * classifier's calls and, when Scrublet ran too, where the two agree, overall and per
 * capture channel (both / Scrublet only / classifier only). With the classifier alone
 * (no Scrublet scores), its score histogram and per-channel calls.
 */
export function DoubletMethods({
  classifier,
  methods,
  scrubletAt,
  showHistogram = false,
}: {
  classifier: ClassifierCalls;
  methods?: DetectorAgreement | null;
  /** "Scrublet at 0.25", "Scrublet's saved call". */
  scrubletAt: string;
  showHistogram?: boolean;
}) {
  const [all, setAll] = useState(false);
  const [logY, setLogY] = useState(true);
  const perChannel = methods ? methods.channels : [];
  const rows = all ? perChannel : perChannel.slice(0, CHANNEL_ROWS);
  const ownRows = all ? classifier.channels : classifier.channels.slice(0, CHANNEL_ROWS);
  const more = (methods ? perChannel.length : classifier.channels.length) > CHANNEL_ROWS;
  const cell = "px-2 text-right tabular-nums";
  return (
    <div className="space-y-1.5 border-t border-slate-100 pt-2" data-testid="doublet-methods">
      <h5 className="flex items-center gap-1 text-[11px] font-semibold text-slate-700">
        {methods ? "Scrublet vs the scDblFinder-style classifier" : "scDblFinder-style classifier"}
        <InfoTip width={300} label="About the classifier">
          A second doublet detector after Germain et al. 2021 (scDblFinder): it adds pairs of cells of different
          clusters together as artificial doublets, trains a model to tell them from the real cells, and scores
          each cell with a model that did not see it. Its threshold is its own, set in each channel; the threshold
          above is Scrublet&apos;s.
        </InfoTip>
      </h5>
      {showHistogram && (
        <div className="space-y-1">
          <label className="flex items-center gap-1 text-[10px] text-slate-500">
            <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} className="h-3 w-3" />
            log scale
          </label>
          <div className="h-[160px]">
            <ThresholdHistogram
              edges={classifier.hist.bin_edges}
              counts={classifier.hist.counts}
              logY={logY}
              barColor="#60a5fa"
              plotName="qc_doublet_score_classifier"
            />
          </div>
        </div>
      )}
      <Caption n={{ value: classifier.n_scored, unit: "cells scored" }} testId="doublet-classifier-text">
        {classifierText(classifier)}
      </Caption>
      {methods && (
        <>
          <table className="text-[11px]" aria-label="Scrublet x classifier doublet calls">
            <thead>
              <tr className="text-slate-400">
                <th />
                <th className="px-2 text-right font-medium" title="Cells the classifier calls doublets">
                  Classifier doublet
                </th>
                <th className="px-2 text-right font-medium" title="Cells the classifier does not call doublets">
                  Classifier singlet
                </th>
              </tr>
            </thead>
            <tbody className="text-slate-600">
              <tr>
                <th className="pr-2 text-left font-medium" title={`Cells ${scrubletAt} calls doublets`}>
                  Scrublet doublet
                </th>
                <td className={`${cell} font-semibold text-slate-800`}>{n(methods.both)}</td>
                <td className={cell}>{n(methods.scrublet_only)}</td>
              </tr>
              <tr>
                <th className="pr-2 text-left font-medium" title={`Cells ${scrubletAt} does not call doublets`}>
                  Scrublet singlet
                </th>
                <td className={cell}>{n(methods.classifier_only)}</td>
                <td className={cell}>{n(methods.neither)}</td>
              </tr>
            </tbody>
          </table>
          <Caption n={{ value: methods.n_cells, unit: "cells" }} testId="doublet-methods-text">
            {detectorAgreementText(methods, scrubletAt)}
          </Caption>
        </>
      )}
      {methods && perChannel.length > 0 && (
        <table className="w-full text-[11px]" aria-label="Detector agreement per capture channel">
          <thead>
            <tr className="text-left text-slate-400">
              <th className="font-medium">Channel</th>
              <th className="text-right font-medium" title="Cells in the channel">Cells</th>
              <th className="text-right font-medium" title="Cells both detectors call doublets">Both</th>
              <th className="text-right font-medium" title={`Cells only ${scrubletAt} calls doublets`}>
                Only Scrublet
              </th>
              <th className="text-right font-medium" title="Cells only the classifier calls doublets">
                Only classifier
              </th>
            </tr>
          </thead>
          <tbody className="text-slate-600">
            {rows.map((r) => (
              <tr key={r.name}>
                <td className="truncate pr-2">{r.name}</td>
                <td className="text-right tabular-nums">{n(r.n_cells)}</td>
                <td className="text-right tabular-nums font-semibold text-slate-800">{n(r.both)}</td>
                <td className="text-right tabular-nums">{n(r.scrublet_only)}</td>
                <td className="text-right tabular-nums">{n(r.classifier_only)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {!methods && classifier.channels.length > 0 && (
        <table className="w-full text-[11px]" aria-label="Classifier calls per capture channel">
          <thead>
            <tr className="text-left text-slate-400">
              <th className="font-medium">Channel</th>
              <th className="text-right font-medium" title="Cells in the channel">Cells</th>
              <th className="text-right font-medium" title="Cells the classifier calls doublets">Called</th>
              <th className="text-right font-medium" title="Called doublets as a share of the channel's cells">Share</th>
              <th className="text-right font-medium" title="The threshold the classifier set in this channel">
                Threshold
              </th>
            </tr>
          </thead>
          <tbody className="text-slate-600">
            {ownRows.map((r) => (
              <tr key={r.name} className={r.status === "scored" ? "" : "text-slate-400"}>
                <td className="truncate pr-2">{r.name}</td>
                <td className="text-right tabular-nums">{n(r.n_cells)}</td>
                {r.status === "scored" ? (
                  <>
                    <td className="text-right tabular-nums">{n(r.n_called)}</td>
                    <td className="text-right tabular-nums">{pct(r.fraction)}</td>
                  </>
                ) : (
                  <td colSpan={2} className="text-right">{r.status === "too_small" ? "too small" : "not scored"}</td>
                )}
                <td className="text-right tabular-nums text-slate-400">
                  {r.threshold != null ? String(Number(r.threshold.toFixed(3))) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {more && (
        <button type="button" onClick={() => setAll((v) => !v)} className="text-[11px] text-primary hover:underline">
          {all ? "Show fewer" : `Show all ${methods ? perChannel.length : classifier.channels.length}`}
        </button>
      )}
    </div>
  );
}
