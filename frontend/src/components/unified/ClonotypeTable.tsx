import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { getClonotypes } from "@/api/vdj";
import { Caption, CapNote } from "@/components/common/Caption";
import { prettyObsLabel } from "@/lib/formatting";
import { errorText } from "@/lib/programs";
import { chainText, genesText } from "@/lib/vdj";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";

const N_ROWS = 25;

/**
 * The largest clonotypes, overall or within one group (top per sample), with the
 * group of a column most of their cells sit in. Clicking a row selects its cells
 * on the plot.
 */
export function ClonotypeTable({ datasetId, groupColumns }: { datasetId: string; groupColumns: string[] }) {
  const [groupBy, setGroupBy] = useState("");
  const [within, setWithin] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const dataset = useDatasetStore((s) => s.currentDataset);
  const { data, error, isLoading } = useQuery({
    queryKey: ["vdj-view", datasetId, "clonotypes", groupBy, within],
    queryFn: () => getClonotypes(datasetId, { groupBy: groupBy || undefined, group: within || null, n: N_ROWS }),
    retry: false,
  });
  const column = data?.group_by ?? groupBy;
  const values = dataset?.obs_columns.find((c) => c.name === column)?.values ?? [];

  const select = (id: string, indices: number[]) => {
    const sel = useSelectionStore.getState();
    if (picked === id) {
      setPicked(null);
      sel.clearSelection();
    } else {
      setPicked(id);
      sel.setSelection(new Set(indices));
    }
  };

  return (
    <section aria-label="Top clonotypes" className="space-y-1">
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-slate-600">
        <span className="font-medium">Largest clonotypes</span>
        <label className="flex items-center gap-1">
          by
          <select
            aria-label="Group cells by"
            value={column ?? ""}
            onChange={(e) => {
              setGroupBy(e.target.value);
              setWithin("");
            }}
            className="max-w-[9rem] rounded border border-slate-300 px-1 py-0.5 text-xs"
          >
            {column && !groupColumns.includes(column) && <option value={column}>{prettyObsLabel(column)}</option>}
            {groupColumns.map((c) => (
              <option key={c} value={c}>
                {prettyObsLabel(c)}
              </option>
            ))}
          </select>
        </label>
        {values.length > 0 && (
          <label className="flex items-center gap-1">
            in
            <select
              aria-label="Within group"
              value={within}
              onChange={(e) => setWithin(e.target.value)}
              className="max-w-[8rem] rounded border border-slate-300 px-1 py-0.5 text-xs"
            >
              <option value="">all cells</option>
              {values.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        )}
        {isLoading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
          {errorText(error)}
        </p>
      )}
      {data && (
        <>
          <div className="max-h-72 overflow-auto rounded border border-slate-200">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-slate-50 text-left text-slate-500">
                <tr>
                  <th className="px-1.5 py-1 font-medium" title="Clonotype id: numbered by size within TCR and BCR">Clonotype</th>
                  <th className="px-1.5 py-1 text-right font-medium" title="Cells of the dataset in this clonotype">Cells</th>
                  {within && (
                    <th className="px-1.5 py-1 text-right font-medium" title={`Cells of this clonotype in ${within}`}>
                      In {within}
                    </th>
                  )}
                  <th className="px-1.5 py-1 font-medium" title="CDR3 amino acids of the primary VJ / VDJ chains (hover for V and J genes)">
                    CDR3
                  </th>
                  <th
                    className="px-1.5 py-1 font-medium"
                    title={`The group of ${column ? prettyObsLabel(column) : "the column"} holding most of the clonotype's cells, and its share`}
                  >
                    Mostly in
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((r) => (
                  <tr
                    key={r.id}
                    onClick={() => select(r.id, r.indices)}
                    aria-selected={picked === r.id}
                    className={`cursor-pointer border-t border-slate-100 ${picked === r.id ? "bg-blue-50" : "hover:bg-slate-50"}`}
                    title="Click to select these cells on the plot"
                  >
                    <td className="px-1.5 py-0.5 font-medium text-slate-700">{r.id}</td>
                    <td className="px-1.5 py-0.5 text-right tabular-nums" title={r.expansion}>
                      {r.n_cells.toLocaleString()}
                    </td>
                    {within && <td className="px-1.5 py-0.5 text-right tabular-nums">{r.n_in_group ?? ""}</td>}
                    <td className="max-w-[12rem] truncate px-1.5 py-0.5 font-mono text-[10px]" title={genesText(r.chains)}>
                      {chainText(r.chains)}
                    </td>
                    <td
                      className="max-w-[8rem] truncate px-1.5 py-0.5"
                      title={Object.entries(r.by_group)
                        .sort((a, b) => b[1] - a[1])
                        .map(([g, c]) => `${g}: ${c}`)
                        .join(", ")}
                    >
                      {r.top_group !== null ? `${r.top_group} (${Math.round((r.top_share ?? 0) * 100)}%)` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <CapNote shown={data.n_shown} total={data.n_clonotypes} by="cells" />
          <Caption n={{ value: data.n_cells_in_clonotypes, unit: within ? `cells with a clonotype in ${within}` : "cells with a clonotype" }}>
            Clonotypes by number of cells{within ? ` in ${within}` : ""}; sizes count the cells in this dataset.
          </Caption>
        </>
      )}
    </section>
  );
}
