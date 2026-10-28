import { useCallback, useEffect, useMemo, useState } from "react";
import { Download } from "lucide-react";
import type { GseaResponse, GseaTerm } from "@/api/enrichment";
import { GseaTermsTable } from "@/components/unified/GseaTermsTable";
import { downloadCsv } from "@/lib/csv";
import {
  GSEA_CSV_HEADERS,
  describeGseaRun,
  gseaCsvRows,
  nextGseaSort,
  sortGseaResults,
  type GseaSortDir,
  type GseaSortField,
} from "@/lib/gsea";

type Side = "up" | "down";

interface Props {
  data: GseaResponse;
  /** What a positive and a negative NES mean here, e.g. "Higher in STIM". */
  labels: { up: string; down: string };
  /** CSV file name without extension. */
  csvName: string;
  onTermClick: (r: GseaTerm) => void;
  onAsk: (r: GseaTerm, side: Side) => void;
  scoringTerm: string | null;
  /** A Reactome row's "Diagram" button. */
  onDiagram?: (r: GseaTerm) => void;
}

/**
 * A GSEA run: how it was made, the two directions (positive / negative NES) one at
 * a time, each as a terms table, and both as CSV.
 */
export function GseaResults({ data, labels, csvName, onTermClick, onAsk, scoringTerm, onDiagram }: Props) {
  const [side, setSide] = useState<Side>("up");
  const [sort, setSort] = useState<{ field: GseaSortField; dir: GseaSortDir }>({ field: "fdr", dir: "asc" });

  // A new run opens on the direction with something significant.
  useEffect(() => {
    setSide(data.up.n_significant > 0 || data.down.n_significant === 0 ? "up" : "down");
  }, [data]);

  const sorted = useMemo(
    () => sortGseaResults(data[side].results, sort.field, sort.dir),
    [data, side, sort],
  );
  const onSort = useCallback((field: GseaSortField) => setSort((s) => nextGseaSort(s, field)), []);
  const exportCsv = useCallback(
    () => downloadCsv(`${csvName}.csv`, GSEA_CSV_HEADERS, gseaCsvRows(data, labels)),
    [data, labels, csvName],
  );
  const fmt = (n: number) => n.toLocaleString();

  return (
    <div className="flex flex-col gap-2">
      <p className="text-[10px] leading-snug text-slate-500">{describeGseaRun(data)}</p>
      {data.warning && (
        <p role="note" className="rounded border border-amber-200 bg-amber-50 px-1.5 py-1 text-amber-800">
          {data.warning}
        </p>
      )}
      <div className="flex items-center gap-2">
        <div className="flex min-w-0 flex-1 rounded-md border border-slate-200 p-0.5">
          {(["up", "down"] as const).map((s) => {
            const n = data[s].n_significant;
            return (
              <button
                key={s}
                onClick={() => setSide(s)}
                title={s === "up" ? "Positive NES, FDR < 0.05" : "Negative NES, FDR < 0.05"}
                className={`flex-1 truncate rounded px-2 py-1 transition ${
                  side === s ? "bg-primary font-medium text-white" : "text-slate-500 hover:text-slate-700"
                }`}
              >
                {labels[s]} · {n} set{n === 1 ? "" : "s"}
              </button>
            );
          })}
        </div>
        <button
          onClick={exportCsv}
          title="Export both directions as CSV"
          className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50"
        >
          <Download className="h-3 w-3" />
          CSV
        </button>
      </div>
      {sorted.length === 0 ? (
        <p className="py-2 text-center text-slate-400">No gene set scores in this direction.</p>
      ) : (
        <>
          {data[side].n_terms > data[side].results.length && (
            <p className="text-[10px] text-slate-400">
              The {fmt(data[side].results.length)} most significant of {fmt(data[side].n_terms)} sets in
              this direction.
            </p>
          )}
          <GseaTermsTable
            results={sorted}
            permutations={data.permutations}
            sortField={sort.field}
            sortDir={sort.dir}
            onSort={onSort}
            onTermClick={onTermClick}
            onAsk={(r) => onAsk(r, side)}
            scoringTerm={scoringTerm}
            onDiagram={onDiagram}
          />
        </>
      )}
    </div>
  );
}
