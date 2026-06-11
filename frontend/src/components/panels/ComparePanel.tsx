import { useCallback, useEffect, useMemo, useState } from "react";
import { GitCompare, Loader2, AlertCircle, Sparkles } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { listDatasets, getDataset } from "@/api/datasets";
import { apiFetch } from "@/api/client";
import { CATEGORICAL_COLORS } from "@/lib/colors";
import { prettyObsLabel } from "@/lib/formatting";
import type { DatasetInfo } from "@/api/types";

interface SummaryRaw {
  groupby: string | null;
  counts: Record<string, number>;
}

/** Categorical obs columns usable as a comparison grouping (cell type, cluster…). */
function categoricalCols(ds: DatasetInfo | null): string[] {
  if (!ds) return [];
  return ds.obs_columns
    .filter(
      (c) =>
        (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
        c.n_unique >= 2 &&
        c.n_unique <= 100,
    )
    .map((c) => c.name);
}

async function fetchProportions(datasetId: string, column: string): Promise<Record<string, number>> {
  const raw = await apiFetch<SummaryRaw>(
    `/datasets/${datasetId}/metadata/summary?groupby=${encodeURIComponent(column)}`,
  );
  const counts = raw.counts ?? {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const props: Record<string, number> = {};
  for (const [k, v] of Object.entries(counts)) props[k] = (v / total) * 100;
  return props;
}

/**
 * Compare two datasets by the composition of a shared categorical column
 * (e.g. cell-type proportions in dataset A vs B). Proportions are normalized so
 * datasets of different sizes are comparable.
 */
export function ComparePanel() {
  const datasetA = useDatasetStore((s) => s.currentDataset);
  const askCopilot = useViewStore((s) => s.askCopilot);

  const [allDatasets, setAllDatasets] = useState<DatasetInfo[]>([]);
  const [datasetB, setDatasetB] = useState<DatasetInfo | null>(null);
  const [datasetBId, setDatasetBId] = useState<string>("");
  const [column, setColumn] = useState<string>("");
  const [propsA, setPropsA] = useState<Record<string, number> | null>(null);
  const [propsB, setPropsB] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load the dataset list (so B can be picked even if Data wasn't visited).
  useEffect(() => {
    listDatasets()
      .then(setAllDatasets)
      .catch(() => setAllDatasets([]));
  }, []);

  // Fetch B's full info (for its obs columns) when chosen.
  useEffect(() => {
    if (!datasetBId) {
      setDatasetB(null);
      return;
    }
    let cancelled = false;
    getDataset(datasetBId)
      .then((d) => !cancelled && setDatasetB(d))
      .catch(() => !cancelled && setDatasetB(null));
    return () => {
      cancelled = true;
    };
  }, [datasetBId]);

  // Columns categorical in BOTH datasets.
  const sharedColumns = useMemo(() => {
    const a = new Set(categoricalCols(datasetA));
    return categoricalCols(datasetB).filter((c) => a.has(c));
  }, [datasetA, datasetB]);

  // Default the comparison column once both datasets are known.
  useEffect(() => {
    if (column && sharedColumns.includes(column)) return;
    const preferred = sharedColumns.find((c) => /cell.?type/i.test(c)) ?? sharedColumns[0] ?? "";
    setColumn(preferred);
  }, [sharedColumns, column]);

  // Fetch both compositions whenever the pair + column are set.
  useEffect(() => {
    if (!datasetA?.id || !datasetBId || !column) {
      setPropsA(null);
      setPropsB(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([fetchProportions(datasetA.id, column), fetchProportions(datasetBId, column)])
      .then(([a, b]) => {
        if (cancelled) return;
        setPropsA(a);
        setPropsB(b);
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetA?.id, datasetBId, column]);

  // Union of categories, ordered by combined abundance.
  const categories = useMemo(() => {
    if (!propsA || !propsB) return [];
    const keys = new Set([...Object.keys(propsA), ...Object.keys(propsB)]);
    return [...keys].sort((x, y) => (propsB[y] ?? 0) + (propsA[y] ?? 0) - (propsA[x] ?? 0) - (propsB[x] ?? 0));
  }, [propsA, propsB]);

  const askCompare = useCallback(() => {
    if (!categories.length || !datasetA || !datasetB) return;
    const top = categories
      .slice(0, 8)
      .map((c) => `${c}: ${(propsA?.[c] ?? 0).toFixed(0)}% vs ${(propsB?.[c] ?? 0).toFixed(0)}%`)
      .join("; ");
    askCopilot(
      `Compare the ${column} composition of "${datasetA.name}" (A) vs "${datasetB.name}" (B): ${top}. ` +
        `What are the most notable differences and what might explain them?`,
    );
  }, [categories, datasetA, datasetB, propsA, propsB, column, askCopilot]);

  const otherDatasets = allDatasets.filter((d) => d.id !== datasetA?.id);

  if (!datasetA) {
    return (
      <div className="space-y-6">
        <Header />
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
          Load a dataset first, then pick a second one to compare against.
        </div>
      </div>
    );
  }

  const aColor = `rgb(${CATEGORICAL_COLORS[0]!.join(",")})`;
  const bColor = `rgb(${CATEGORICAL_COLORS[3]!.join(",")})`;

  return (
    <div className="space-y-5">
      <Header />

      {/* Dataset + column pickers */}
      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-500">Dataset A</label>
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm font-medium text-slate-700">
            {datasetA.name}
          </div>
        </div>
        <div className="text-slate-400">vs</div>
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-500">Dataset B</label>
          <select
            value={datasetBId}
            onChange={(e) => setDatasetBId(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700"
          >
            <option value="">Select a dataset…</option>
            {otherDatasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
        {datasetBId && (
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-500">Compare by</label>
            <select
              value={column}
              onChange={(e) => setColumn(e.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700"
            >
              {sharedColumns.length === 0 && <option value="">(no shared categorical column)</option>}
              {sharedColumns.map((c) => (
                <option key={c} value={c}>
                  {prettyObsLabel(c)}
                </option>
              ))}
            </select>
          </div>
        )}
        {categories.length > 0 && (
          <button
            onClick={askCompare}
            className="ml-auto inline-flex items-center gap-1.5 rounded-lg border border-primary/30 px-3 py-1.5 text-sm font-medium text-primary hover:bg-primary/10"
          >
            <Sparkles className="h-4 w-4" />
            Ask co-pilot to compare
          </button>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}
      {loading && (
        <div className="flex items-center justify-center gap-2 py-12 text-sm text-slate-400">
          <Loader2 className="h-5 w-5 animate-spin" />
          Comparing compositions…
        </div>
      )}

      {!loading && datasetB && sharedColumns.length === 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-6 text-center text-sm text-amber-700">
          These two datasets share no categorical column (e.g. <code>cell_type</code>). Annotate cell
          types in both to compare them.
        </div>
      )}

      {!loading && categories.length > 0 && propsA && propsB && datasetB && (
        <>
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <Plot
              data={[
                {
                  type: "bar",
                  name: datasetA.name,
                  x: categories,
                  y: categories.map((c) => propsA[c] ?? 0),
                  marker: { color: aColor },
                  hovertemplate: "%{x}: %{y:.1f}%<extra>A</extra>",
                },
                {
                  type: "bar",
                  name: datasetB.name,
                  x: categories,
                  y: categories.map((c) => propsB[c] ?? 0),
                  marker: { color: bColor },
                  hovertemplate: "%{x}: %{y:.1f}%<extra>B</extra>",
                },
              ]}
              layout={{
                barmode: "group",
                autosize: true,
                height: 420,
                margin: { t: 30, r: 16, b: 110, l: 56 },
                title: { text: `${prettyObsLabel(column)} composition (% of cells)`, font: { size: 14, color: "#334155" } },
                xaxis: { tickangle: -45, tickfont: { size: 10, color: "#64748b" } },
                yaxis: { title: { text: "% of cells", font: { size: 11, color: "#64748b" } }, gridcolor: "#f1f5f9", tickfont: { size: 10 } },
                legend: { orientation: "h", y: 1.12, font: { size: 11 } },
                paper_bgcolor: "white",
                plot_bgcolor: "white",
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>

          {/* Δ table */}
          <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50 text-left text-slate-600">
                  <th className="px-4 py-2 font-medium">{prettyObsLabel(column)}</th>
                  <th className="px-4 py-2 text-right font-medium">A %</th>
                  <th className="px-4 py-2 text-right font-medium">B %</th>
                  <th className="px-4 py-2 text-right font-medium">Δ (B − A)</th>
                </tr>
              </thead>
              <tbody>
                {categories.map((c) => {
                  const a = propsA[c] ?? 0;
                  const bv = propsB[c] ?? 0;
                  const d = bv - a;
                  return (
                    <tr key={c} className="border-b border-slate-50">
                      <td className="px-4 py-1.5 font-medium text-slate-800">{c}</td>
                      <td className="px-4 py-1.5 text-right tabular-nums text-slate-600">{a.toFixed(1)}</td>
                      <td className="px-4 py-1.5 text-right tabular-nums text-slate-600">{bv.toFixed(1)}</td>
                      <td
                        className={`px-4 py-1.5 text-right tabular-nums font-medium ${
                          d > 1 ? "text-emerald-600" : d < -1 ? "text-red-600" : "text-slate-400"
                        }`}
                      >
                        {d > 0 ? "+" : ""}
                        {d.toFixed(1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function Header() {
  return (
    <div className="flex items-center gap-3">
      <GitCompare className="h-6 w-6 text-primary" />
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Compare datasets</h2>
        <p className="text-sm text-slate-500">
          Compare the cell-type / cluster composition of two datasets side by side.
        </p>
      </div>
    </div>
  );
}
