import { useState } from "react";
import type { AgreementCounts, DoubletAgreement as Agreement } from "@/api/doubletSource";
import { Caption } from "@/components/common/Caption";
import { agreementText } from "@/lib/doubletSource";

const SAMPLE_ROWS = 8; // more behind "show all"

const n = (v: number) => v.toLocaleString();

function SampleRow({ name, c, title }: { name: string; c: AgreementCounts; title?: string }) {
  return (
    <tr>
      <td className="truncate pr-2" title={title}>{name}</td>
      <td className="text-right tabular-nums">{n(c.n_cells)}</td>
      <td className="text-right tabular-nums">{n(c.hashing)}</td>
      <td className="text-right tabular-nums">{n(c.scrublet)}</td>
      <td className="text-right tabular-nums">{n(c.both)}</td>
    </tr>
  );
}

/**
 * Cell hashing x Scrublet on the doublet card, when the dataset has both: the 2 x 2
 * counts of their doublet calls (Scrublet at the threshold shown) and the same per
 * sample. Hashing is the doublet views' primary call; Scrublet the second opinion,
 * the only one that sees two cells of one sample in a droplet.
 */
export function DoubletAgreement({ agreement: a, scrubletAt }: { agreement: Agreement; scrubletAt: string }) {
  const [all, setAll] = useState(false);
  const rows = all ? a.samples : a.samples.slice(0, SAMPLE_ROWS);
  const cell = "px-2 text-right tabular-nums";
  return (
    <div className="space-y-1.5 border-t border-slate-100 pt-2" data-testid="doublet-agreement">
      <h5 className="text-[11px] font-semibold text-slate-700">Cell hashing vs Scrublet</h5>
      <table className="text-[11px]" aria-label="Cell hashing x Scrublet doublet calls">
        <thead>
          <tr className="text-slate-400">
            <th />
            <th className="px-2 text-right font-medium" title={`Cells ${scrubletAt} calls doublets`}>
              Scrublet doublet
            </th>
            <th className="px-2 text-right font-medium" title={`Cells ${scrubletAt} does not call doublets`}>
              Scrublet singlet
            </th>
            <th className="px-2 text-right font-medium" title="Row total">Total</th>
          </tr>
        </thead>
        <tbody className="text-slate-600">
          <tr>
            <th className="pr-2 text-left font-medium" title="hash_class = Doublet: tags of two or more samples">
              Hashing doublet
            </th>
            <td className={`${cell} font-semibold text-slate-800`}>{n(a.both)}</td>
            <td className={cell}>{n(a.hashing_only)}</td>
            <td className={cell}>{n(a.hashing)}</td>
          </tr>
          <tr>
            <th
              className="pr-2 text-left font-medium"
              title="hash_class = Singlet or Negative (or not called): not a doublet between samples"
            >
              Hashing not doublet
            </th>
            <td className={cell}>{n(a.scrublet_only)}</td>
            <td className={cell}>{n(a.neither)}</td>
            <td className={cell}>{n(a.n_cells - a.hashing)}</td>
          </tr>
          <tr className="text-slate-400">
            <th className="pr-2 text-left font-medium">Total</th>
            <td className={cell}>{n(a.scrublet)}</td>
            <td className={cell}>{n(a.n_cells - a.scrublet)}</td>
            <td className={cell}>{n(a.n_cells)}</td>
          </tr>
        </tbody>
      </table>
      <Caption n={{ value: a.n_cells, unit: "cells" }} testId="doublet-agreement-text">
        {agreementText(a, scrubletAt)}
      </Caption>
      {a.samples.length > 0 && (
        <>
          <table className="w-full text-[11px]" aria-label="Cell hashing x Scrublet per sample">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="font-medium">{a.column}</th>
                <th className="text-right font-medium" title="Cells in the sample">Cells</th>
                <th className="text-right font-medium" title="Cells hashing calls doublets">Hashing</th>
                <th className="text-right font-medium" title={`Cells ${scrubletAt} calls doublets`}>Scrublet</th>
                <th className="text-right font-medium" title="Cells both call doublets">Both</th>
              </tr>
            </thead>
            <tbody className="text-slate-600">
              {rows.map((r) => (
                <SampleRow key={r.name} name={r.name} c={r} />
              ))}
              {a.unassigned && (
                <SampleRow
                  name="(no value)"
                  c={a.unassigned}
                  title={`Cells without a ${a.column} value: a sample column from hashing leaves doublets and negatives empty`}
                />
              )}
            </tbody>
          </table>
          {a.samples.length > SAMPLE_ROWS && (
            <button type="button" onClick={() => setAll((v) => !v)} className="text-[11px] text-primary hover:underline">
              {all ? "Show fewer" : `Show all ${a.samples.length}`}
            </button>
          )}
        </>
      )}
    </div>
  );
}
