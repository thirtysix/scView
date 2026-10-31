import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { getAtac, type AtacActivity, type AtacState } from "@/api/atac";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { AtacActivityCard } from "./AtacActivityCard";
import { AtacDifferentialCard } from "./AtacDifferentialCard";
import { AtacLsiCard } from "./AtacLsiCard";
import { AtacQcCard } from "./AtacQcCard";
import { AtacWnnCard } from "./AtacWnnCard";

/**
 * "Accessibility" (backend core/atac.py): a 10x Multiome dataset's peaks, which stay on
 * disk. QC, LSI and clusters, WNN with RNA, gene activity, differential peaks. The tab
 * shows only for a dataset with ATAC (UnifiedViewPanel's tab list).
 */
export function AccessibilitySubtab({
  datasetId,
  groupByColumn,
  setGroupByColumn,
  activeGene,
  onActivity,
}: {
  datasetId: string;
  groupByColumn: string;
  setGroupByColumn: (column: string) => void;
  activeGene: string | null;
  onActivity: (a: AtacActivity) => void;
}) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const version = useAtacRunStore((s) => s.version);
  const refreshDataset = useDatasetRefresh();
  const [state, setState] = useState<AtacState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const groupColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter(
      (c) =>
        (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
        c.n_unique >= 2 &&
        c.n_unique <= 100 &&
        !(nCells > 0 && c.n_unique / nCells >= 0.9),
    );
  }, [dataset]);
  const groupby = groupColumns.some((c) => c.name === groupByColumn) ? groupByColumn : "";

  const reload = useCallback(() => {
    setError(null);
    return getAtac(datasetId, groupby || undefined)
      .then(setState)
      .catch((e) => setError(errorMessage(e)));
  }, [datasetId, groupby]);

  useEffect(() => {
    void reload();
  }, [reload, version]);

  // A run that just saved: the embedding menu and colour-by pickers list what it added.
  useEffect(() => {
    if (version > 0) void refreshDataset(datasetId).catch(() => undefined);
  }, [version, datasetId, refreshDataset]);

  if (error) return <RequestError message={error} />;
  if (!state) {
    return (
      <p className="flex items-center gap-1.5 text-[11px] text-slate-400">
        <Loader2 className="h-3 w-3 animate-spin" /> Loading…
      </p>
    );
  }
  if (!state.available) return <p className="text-xs leading-snug text-slate-500">{state.reason}</p>;

  const info = state.info ?? {};
  return (
    <div className="space-y-3" data-testid="accessibility-subtab">
      <div className="space-y-1">
        <h3 className="flex items-center gap-1 text-xs font-semibold text-slate-700">
          <span>ATAC peaks</span>
          <InfoTip label="About the ATAC peaks" width={300}>
            A 10x Multiome file holds genes and peaks in one matrix. The peaks stay in their own file on disk, read a block
            of cells at a time; they are never in the gene list, and the RNA pipeline, QC and filtering run as for RNA alone.
          </InfoTip>
        </h3>
        <Caption n={{ value: state.n_cells ?? 0, unit: "cells" }}>
          {(info.n_peaks ?? 0).toLocaleString()} peaks, {Math.round((info.nnz ?? 0) / 1e6).toLocaleString()}M values, on{" "}
          {info.genome ?? "an unnamed genome"}
          {info.genome_assumed ? " (read from the gene ids; the file does not name one)" : ""}, from {info.source ?? "the upload"}.
        </Caption>
        {groupColumns.length > 0 && (
          <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
            Group by
            <select
              value={groupby}
              onChange={(e) => setGroupByColumn(e.target.value)}
              className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700"
            >
              <option value="">None</option>
              {groupColumns.map((c) => (
                <option key={c.name} value={c.name}>{prettyObsLabel(c.name)}</option>
              ))}
            </select>
          </label>
        )}
      </div>

      <AtacQcCard datasetId={datasetId} groupby={groupby} qcColumns={state.qc_columns ?? []} metricsFiles={info.metrics_files ?? []}
                  onAttached={() => void reload().then(() => refreshDataset(datasetId).catch(() => undefined))} />
      {/* FRAGMENTS SLOT: the fragments agent's AtacFragmentQc and AtacCoverage cards are mounted here by the lead. */}
      <AtacLsiCard datasetId={datasetId} state={state} />
      <AtacWnnCard datasetId={datasetId} state={state} groupby={groupby} />
      <AtacActivityCard datasetId={datasetId} state={state} groupby={groupby} activeGene={activeGene} onActivity={onActivity} />
      <AtacDifferentialCard datasetId={datasetId} state={state} columns={groupColumns.map((c) => c.name)} defaultColumn={groupby} />
    </div>
  );
}
