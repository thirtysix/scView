import { useCallback, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { getVdj, redefineClonotypes, removeVdj, type VdjSummary } from "@/api/vdj";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { ClonotypeAttach } from "@/components/unified/ClonotypeAttach";
import { ClonotypeRepertoire } from "@/components/unified/ClonotypeRepertoire";
import { ClonotypeTable } from "@/components/unified/ClonotypeTable";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { rankGroupingColumns } from "@/lib/obsColumns";
import { errorText } from "@/lib/programs";
import { droppedText, fileSentence, matchSentences, STATUS_INFO } from "@/lib/vdj";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const COLOUR_BY: { column: string; label: string; title: string }[] = [
  { column: "clone_expansion", label: "Expansion", title: "Colour cells by clone size bin (1, 2-5, 6-20, >20 cells), in size order" },
  { column: "clone_size", label: "Clone size", title: "Colour cells by the number of cells in their clonotype; cells without one are grey" },
  { column: "vdj_status", label: "Chain status", title: "Colour cells by chain status (paired, single chain, multichain ...)" },
];

/**
 * Unified View > Clonotypes: TCR/BCR contigs joined to the cells (backend
 * core/vdj.py). Says how to attach a contig table when there is none; otherwise
 * what matched both ways, the clonotype definition, colour buttons, the largest
 * clonotypes (a row selects its cells) and the repertoire views per group.
 */
export function ClonotypeSubtab({ groupByColumn }: { groupByColumn: string }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const queryClient = useQueryClient();
  const refresh = useDatasetRefresh();
  const { data, error, isLoading } = useQuery({
    queryKey: ["vdj", datasetId],
    queryFn: () => getVdj(datasetId!),
    enabled: !!datasetId,
    retry: false,
  });

  const changed = useCallback(
    async (summary: VdjSummary) => {
      if (!datasetId) return;
      queryClient.setQueryData(["vdj", datasetId], summary);
      await queryClient.invalidateQueries({ queryKey: ["vdj-view", datasetId] });
      await refresh(datasetId, "clonotype");
    },
    [datasetId, queryClient, refresh],
  );

  const groupColumns = useMemo(() => {
    if (!dataset) return [];
    const own = new Set(data?.columns ?? []);
    return rankGroupingColumns(dataset.obs_columns, { nCells: dataset.n_cells ?? 0 })
      .map((g) => g.name)
      .filter((name) => !own.has(name));
  }, [dataset, data?.columns]);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  return (
    <div className="space-y-3 text-xs">
      <div className="flex items-center gap-1.5 font-medium text-slate-700">
        TCR / BCR clonotypes
        <InfoTip label="About clonotypes" width={320}>
          10x 5&apos; immune profiling reads each cell&apos;s T- or B-cell receptor chains beside its expression.
          The contig table is joined to the cells by barcode. Per cell, productive high-confidence chains with a
          CDR3 are kept, the top two of each arm by UMIs. A clonotype is the cells with identical CDR3 amino
          acids on the primary VJ and VDJ chains (scirpy&apos;s definition, Sturm et al. 2020): cells of one
          clone, descended from one T or B cell. Clone sizes count the cells in this dataset, a small sample of
          the person&apos;s repertoire.
        </InfoTip>
        {isLoading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
          {errorText(error)}
        </p>
      )}
      {data && !data.has_vdj && (
        <div className="space-y-2" data-testid="vdj-empty">
          <p className="text-[11px] leading-snug text-slate-600">{data.how_to}</p>
          <ClonotypeAttach datasetId={datasetId} summary={data} onAttached={changed} />
        </div>
      )}
      {data && data.has_vdj && (
        <>
          <VdjSummaryCard datasetId={datasetId} summary={data} onChanged={changed} />
          <ClonotypeTable datasetId={datasetId} groupColumns={groupColumns} />
          <ClonotypeRepertoire datasetId={datasetId} groupColumns={groupColumns} groupByColumn={groupByColumn} />
          <details className="rounded border border-slate-200 px-2 py-1.5">
            <summary className="cursor-pointer text-[11px] font-medium text-slate-600">Add another contig table</summary>
            <div className="pt-2">
              <ClonotypeAttach datasetId={datasetId} summary={data} onAttached={changed} />
            </div>
          </details>
        </>
      )}
    </div>
  );
}

function VdjSummaryCard({
  datasetId,
  summary,
  onChanged,
}: {
  datasetId: string;
  summary: VdjSummary;
  onChanged: (s: VdjSummary) => Promise<void>;
}) {
  const [busy, setBusy] = useState<"define" | "remove" | null>(null);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const colorBy = useSettingsStore((s) => s.colorBy);

  const run = async (kind: "define" | "remove", fn: () => Promise<{ summary: VdjSummary }>) => {
    setBusy(kind);
    setError(null);
    try {
      await onChanged((await fn()).summary);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
      setConfirmRemove(false);
    }
  };

  const status = summary.status ?? {};
  const expansion = summary.expansion ?? {};
  return (
    <section aria-label="V(D)J summary" className="space-y-1.5">
      {matchSentences(summary).map((s) => (
        <p key={s} className="text-[11px] leading-snug text-slate-600">
          {s}
        </p>
      ))}
      <table className="w-full text-[11px]" data-testid="vdj-status">
        <thead>
          <tr className="text-left text-slate-400">
            <th className="font-normal" title="Chain status of each cell (obs column vdj_status)">Chain status</th>
            <th className="text-right font-normal" title="Number of cells">Cells</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(status).map(([label, count]) => (
            <tr key={label} className="text-slate-600">
              <td title={STATUS_INFO[label] ?? label}>{label}</td>
              <td className="text-right tabular-nums">{count.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] leading-snug text-slate-600" data-testid="vdj-clonotype-counts">
        {(summary.n_clonotypes ?? 0).toLocaleString()} clonotypes over {(summary.n_cells_in_clonotypes ?? 0).toLocaleString()}{" "}
        cells; {(summary.n_expanded ?? 0).toLocaleString()} expanded (2 or more cells) holding{" "}
        {(summary.n_cells_expanded ?? 0).toLocaleString()} cells
        {Object.entries(expansion)
          .filter(([, v]) => v.clonotypes > 0)
          .map(([label, v]) => ` · ${label}: ${v.clonotypes.toLocaleString()}`)
          .join("")}
        .
      </p>
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-[11px] text-slate-500">Colour by</span>
        {COLOUR_BY.map((c) => (
          <button
            key={c.column}
            title={c.title}
            aria-pressed={colorBy === c.column}
            onClick={() => useSettingsStore.getState().setColorBy(c.column)}
            className={`rounded border px-1.5 py-0.5 text-[11px] ${
              colorBy === c.column ? "border-slate-600 bg-slate-600 text-white" : "border-slate-300 text-slate-600 hover:bg-slate-50"
            }`}
          >
            {c.label}
          </button>
        ))}
      </div>
      {summary.stale && (
        <div role="status" className="rounded-md bg-amber-50 px-2 py-1.5 text-[11px] text-amber-800">
          {summary.stale_note}{" "}
          <button
            onClick={() => run("define", () => redefineClonotypes(datasetId, !!summary.same_v_gene))}
            disabled={busy !== null}
            className="font-medium underline disabled:opacity-50"
          >
            Recompute
          </button>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
        <label className="flex items-center gap-1" title="Also require the same V genes on both chains (scirpy's same_v_gene)">
          <input
            type="checkbox"
            checked={!!summary.same_v_gene}
            disabled={busy !== null}
            onChange={(e) => run("define", () => redefineClonotypes(datasetId, e.target.checked))}
          />
          Same V genes too
        </label>
        <InfoTip label="The clonotype definition" width={300}>
          {summary.definition}
        </InfoTip>
        {busy === "define" && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
        <span className="ml-auto">
          {confirmRemove ? (
            <>
              Remove every V(D)J column?{" "}
              <button onClick={() => run("remove", () => removeVdj(datasetId))} className="font-medium text-red-700 underline">
                Remove
              </button>{" "}
              <button onClick={() => setConfirmRemove(false)} className="underline">
                Keep
              </button>
            </>
          ) : (
            <button
              onClick={() => setConfirmRemove(true)}
              disabled={busy !== null}
              title="Remove the V(D)J columns, chain table and record (the stored contig files stay)"
              className="text-slate-400 underline hover:text-slate-600"
            >
              Remove contigs
            </button>
          )}
        </span>
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
          {error}
        </p>
      )}
      <details className="text-[11px] text-slate-500">
        <summary className="cursor-pointer">Attached contig tables ({(summary.files ?? []).length})</summary>
        <ul className="mt-1 space-y-1">
          {(summary.files ?? []).map((f) => (
            <li key={`${f.sha256}-${f.sample ?? ""}`}>
              {fileSentence(f)}
              {droppedText(f.dropped) && <> Left out: {droppedText(f.dropped)}.</>}
              {f.unmatched_examples.length > 0 && (
                <> Unmatched, e.g. {f.unmatched_examples.slice(0, 3).join(", ")}.</>
              )}
            </li>
          ))}
        </ul>
      </details>
      <Caption n={{ value: summary.match?.cells_with_chain ?? 0, unit: "cells with a chain", of: summary.n_cells }} />
    </section>
  );
}
