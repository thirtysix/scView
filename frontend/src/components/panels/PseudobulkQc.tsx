import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { getPseudobulkQc, type PcAssociationRow, type PseudobulkQc as Result } from "@/api/sampleQc";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { plotlyColorscale } from "@/lib/colors";
import { plotConfig } from "@/lib/plotConfig";
import { defaultColourBy, fmtP, pcaTraces, pcLabel, r2Shade } from "@/lib/pseudobulkQc";
import { useSettingsStore } from "@/stores/settingsStore";

const FILL_STYLE = { width: "100%", height: "100%" } as const;

/** Rows = covariates, columns = PCs, shaded by R^2; a driver's cell is outlined. */
export function AssociationTable({ data }: { data: Result }) {
  const nPcs = data.explained_variance.length;
  const isDriver = (row: PcAssociationRow, pc: number) => data.drivers[`PC${pc}`]?.covariate === row.covariate;
  return (
    <table className="w-full text-[11px]" aria-label="PC-covariate association">
      <thead>
        <tr className="text-left text-slate-400">
          <th className="font-medium">Covariate ({data.profiles.length} profiles)</th>
          {Array.from({ length: nPcs }, (_, i) => (
            <th key={i} className="px-1 text-right font-medium">
              {pcLabel(i + 1, data.explained_variance)}
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="text-slate-700">
        {data.association.map((row) => (
          <tr key={row.covariate} className="border-t border-white">
            <td className="max-w-[10rem] truncate pr-2" title={`${row.covariate}: ${row.test}`}>
              {row.covariate}
              {row.same_as.length > 0 && <span className="text-slate-400"> (= {row.same_as.join(", ")})</span>}
            </td>
            {row.pcs.map((t) => (
              <td
                key={t.pc}
                className={`px-1 text-right ${isDriver(row, t.pc) ? "font-semibold ring-1 ring-inset ring-blue-500" : ""}`}
                style={{ backgroundColor: r2Shade(t.r2) }}
                title={
                  t.r2 === null
                    ? "Not tested: too few profiles per level"
                    : `R² ${t.r2.toFixed(2)}, p ${fmtP(t.p)}${t.p_rank !== null ? `, Kruskal-Wallis p ${fmtP(t.p_rank)}` : ""}${
                        t.rho !== undefined ? `, rho ${t.rho.toFixed(2)}` : ""
                      }`
                }
              >
                {t.r2 === null ? "–" : t.r2.toFixed(2)}
                {t.p !== null && t.p < 0.05 ? "*" : ""}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function driverText(data: Result, pc: number): string {
  const d = data.drivers[`PC${pc}`];
  const label = pcLabel(pc, data.explained_variance);
  if (!d) return `${label}: no covariate tracks it (p < 0.05)`;
  return `${label}: ${d.covariate} (R² ${d.r2.toFixed(2)}, p ${fmtP(d.p)})`;
}

/**
 * Sample-level QC on the Data Assessment tab: each sample's (x condition's) summed
 * counts as one profile, a PCA of the profiles coloured by any sample-level
 * covariate, which covariate drives each PC, and outlier profiles. Asked for on
 * demand (it sums every cell's counts), optionally within one cell type.
 */
export function PseudobulkQc({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
  const [open, setOpen] = useState(false);
  // One cell type's cells only ("" = all), from the column the backend offers.
  const [cells, setCells] = useState<{ column: string | null; value: string }>({ column: null, value: "" });
  const celltype = cells.value;
  const [split, setSplit] = useState(true);
  const [splitName, setSplitName] = useState<string | null>(null);
  const [data, setData] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [colourBy, setColourBy] = useState<string | null>(null);
  const colormapName = useSettingsStore((s) => s.continuousColormap);

  useEffect(() => {
    setData(null);
    setCells({ column: null, value: "" });
    setSplit(true);
    setSplitName(null);
    setColourBy(null);
  }, [datasetId]);

  useEffect(() => {
    if (!open || !datasetId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPseudobulkQc(datasetId, {
      splitBy: split ? undefined : "",
      celltypeColumn: cells.value ? cells.column : null,
      celltype: cells.value || null,
    })
      .then((d) => {
        if (cancelled) return;
        setData(d);
        if (d.split_by) setSplitName(d.split_by); // kept while unticked, to tick it back
        setColourBy((c) => (c && d.covariates.some((x) => x.name === c) ? c : defaultColourBy(d)));
      })
      .catch((e) => {
        if (!cancelled) setError(String(e?.message ?? e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, datasetId, cells, split, refreshKey]);

  const colorOf = useCategoryColors(colourBy);
  const traces = useMemo(
    () => (data && data.profiles.length ? pcaTraces(data, colourBy, colorOf, plotlyColorscale(colormapName)) : []),
    [data, colourBy, colorOf, colormapName],
  );

  const header = (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h4 className="text-xs font-semibold text-slate-700">Sample-level (pseudobulk) QC</h4>
      <span className="flex items-center gap-2">
        {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="rounded border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
        >
          {open ? "Hide" : "Show"}
        </button>
      </span>
    </div>
  );

  if (!open) {
    return (
      <div className="space-y-1 rounded-xl border border-slate-200 bg-white p-3" data-testid="pseudobulk-qc">
        {header}
        <p className="text-[11px] text-slate-500">
          Each sample&apos;s summed counts as one profile: a PCA of the samples, which sample-level covariate
          (condition, batch, donor, sex, depth) drives each PC, and samples that stand apart.
        </p>
      </div>
    );
  }

  const nProfiles = data?.profiles.length ?? 0;
  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3" data-testid="pseudobulk-qc">
      {header}
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Pseudobulk QC unavailable: {error}
        </p>
      )}
      {data && (
        <>
          {data.sample_column && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-500">
              {data.celltype_column && data.celltype_levels.length > 0 && (
                <label className="flex items-center gap-1">
                  Cells
                  <select
                    aria-label="Cell type"
                    value={celltype}
                    onChange={(e) => setCells({ column: data.celltype_column, value: e.target.value })}
                    className="max-w-[12rem] rounded border border-slate-200 bg-white px-1 py-0.5 text-xs text-slate-600"
                  >
                    <option value="">All cells</option>
                    {data.celltype_levels.map((l) => (
                      <option key={l.name} value={l.name}>
                        {data.celltype_column}: {l.name} ({l.n_cells.toLocaleString()})
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {splitName && (
                <label className="flex items-center gap-1">
                  <input
                    type="checkbox"
                    checked={split}
                    onChange={(e) => setSplit(e.target.checked)}
                    className="h-3 w-3"
                  />
                  One profile per {data.sample_column} x {splitName}
                </label>
              )}
              {data.covariates.length > 0 && (
                <label className="flex items-center gap-1">
                  Colour by
                  <select
                    aria-label="Colour by"
                    value={colourBy ?? ""}
                    onChange={(e) => setColourBy(e.target.value || null)}
                    className="max-w-[12rem] rounded border border-slate-200 bg-white px-1 py-0.5 text-xs text-slate-600"
                  >
                    {data.covariates.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
          )}
          {data.message ? (
            <p className="rounded-md bg-slate-50 px-3 py-2 text-[11px] text-slate-600">{data.message}</p>
          ) : (
            <>
              <p className="text-[11px] text-slate-600" data-testid="pseudobulk-caption">
                {nProfiles} profiles ({data.sample_column}
                {data.split_by ? ` x ${data.split_by}` : ""}
                {data.celltype ? `, ${data.celltype} cells only` : ""}) from{" "}
                {data.n_cells_used.toLocaleString()} of {data.n_cells.toLocaleString()} cells; log2(CPM + 1) of the{" "}
                {data.n_genes.toLocaleString()} most variable genes.
                {data.dropped.length > 0 &&
                  ` ${data.dropped.length} profile${data.dropped.length > 1 ? "s" : ""} under ${data.min_cells} cells left out.`}
              </p>
              <div className="h-[300px]">
                <Plot
                  data={traces}
                  layout={{
                    autosize: true,
                    margin: { l: 50, r: 10, t: 8, b: 40 },
                    xaxis: { title: { text: pcLabel(1, data.explained_variance), font: { size: 10 } }, tickfont: { size: 9 }, zeroline: false },
                    yaxis: { title: { text: pcLabel(2, data.explained_variance), font: { size: 10 } }, tickfont: { size: 9 }, zeroline: false },
                    showlegend: true,
                    legend: { font: { size: 9 }, itemsizing: "constant" },
                    font: { size: 9 },
                  }}
                  config={plotConfig("pseudobulk_pca")}
                  style={FILL_STYLE}
                  useResizeHandler
                />
              </div>
              <ul className="space-y-0.5 text-[11px] text-slate-600" aria-label="PC drivers">
                {[1, 2].filter((pc) => pc <= data.explained_variance.length).map((pc) => (
                  <li key={pc}>
                    <span className="font-medium">{driverText(data, pc)}</span>
                    {data.loadings[`PC${pc}`] && (
                      <span className="text-slate-400">
                        {" "}
                        · + {data.loadings[`PC${pc}`]!.positive.slice(0, 4).join(", ")} / −{" "}
                        {data.loadings[`PC${pc}`]!.negative.slice(0, 4).join(", ")}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
              {data.association.length > 0 && (
                <div className="max-h-64 overflow-auto">
                  <AssociationTable data={data} />
                </div>
              )}
              <p className="text-[10px] text-slate-400">
                R² of each covariate against each PC's scores (ANOVA for categories, Spearman for numbers); * p &lt;
                0.05; outlined: the PC&apos;s driver (highest R² at p &lt; 0.05). Hover a cell for p-values.
              </p>
              {data.outliers.length > 0 ? (
                <ul className="space-y-1" aria-label="Outlier profiles">
                  {data.outliers.map((o) => (
                    <li
                      key={o.id}
                      className="flex items-start gap-1.5 rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800"
                    >
                      <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
                      <span>{o.reason}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                data.outlier_rule && (
                  <p className="text-[11px] text-slate-600">No outlier among the {nProfiles} profiles.</p>
                )
              )}
              {data.outlier_rule && <p className="text-[10px] text-slate-400">{data.outlier_rule}</p>}
            </>
          )}
          {data.notes.map((n) => (
            <p key={n} className="text-[10px] text-slate-400">
              {n}
            </p>
          ))}
        </>
      )}
    </div>
  );
}
