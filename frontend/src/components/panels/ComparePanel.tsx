import { useCallback, useEffect, useMemo, useState } from "react";
import { GitCompare, Loader2, AlertCircle, Sparkles } from "lucide-react";
import { CompositionTest } from "@/components/compare/CompositionTest";
import { JointEmbedding } from "@/components/compare/JointEmbedding";
import { MiloTest } from "@/components/compare/MiloTest";
import { MiloDETest } from "@/components/compare/MiloDETest";
import { StudyPlanner } from "@/components/compare/StudyPlanner";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { useOnScreenStore } from "@/stores/onScreenStore";
import { useComparePrefillStore, type ComparePrefill, type CompareTarget } from "@/stores/comparePrefillStore";
import { listDatasets, getDataset } from "@/api/datasets";
import { apiFetch } from "@/api/client";
import { errorMessage } from "@/lib/errorMessage";
import { getPalette } from "@/lib/colors";
import { useSettingsStore } from "@/stores/settingsStore";
import { prettyObsLabel } from "@/lib/formatting";
import type { DatasetInfo } from "@/api/types";
import { TabList, TabPanel } from "@/components/common/TabList";

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

type CompareTab = "joint" | "composition" | "planner";

const TABS: { id: CompareTab; label: string }[] = [
  { id: "joint", label: "Joint embedding" },
  { id: "composition", label: "Composition" },
  { id: "planner", label: "Plan a study" },
];

/** The tab a co-pilot target opens (Milo and miloDE are modes of Composition). */
const TAB_OF: Record<CompareTarget, CompareTab> = {
  composition: "composition", milo: "composition", milode: "composition", joint: "joint", planner: "planner",
};
const MODE_OF: Partial<Record<CompareTarget, CompositionMode>> = { milo: "milo", milode: "milode" };

/** A co-pilot request not applied yet opens its tab, once (a later visit opens as usual). */
function useFreshPrefill(): ComparePrefill | null {
  const prefill = useComparePrefillStore((s) => s.prefill);
  const applied = useComparePrefillStore((s) => s.applied);
  return prefill && prefill.nonce !== applied ? prefill : null;
}

/** Multi-dataset analysis: a joint embedding across datasets, or a side-by-side
 * composition comparison of two. */
export function ComparePanel() {
  const fresh = useFreshPrefill();
  const [tab, setTab] = useState<CompareTab>(fresh ? TAB_OF[fresh.tab] : "joint");
  // The columns the co-pilot asked for, handed to the tab it opened (remounted per request).
  const [prefill, setPrefill] = useState<ComparePrefill | null>(fresh);
  const markApplied = useComparePrefillStore((s) => s.markApplied);
  useEffect(() => {
    if (!fresh) return;
    setTab(TAB_OF[fresh.tab]);
    setPrefill(fresh);
    markApplied(fresh.nonce);
  }, [fresh, markApplied]);
  // The co-pilot's view context names the open tab (stores/onScreenStore.ts)
  const setOnScreenTab = useOnScreenStore((s) => s.setCompareTab);
  useEffect(() => {
    setOnScreenTab(tab);
  }, [tab, setOnScreenTab]);
  useEffect(() => () => setOnScreenTab(null), [setOnScreenTab]);
  return (
    <div className="space-y-5">
      <Header />
      <TabList
        tabs={TABS}
        active={tab}
        onChange={setTab}
        label="Compare"
        idPrefix="compare"
        className="flex gap-1 border-b border-slate-200"
        tabClassName={(selected) =>
          `-mb-px border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
            selected ? "border-primary text-primary" : "border-transparent text-slate-500 hover:text-slate-700"
          }`
        }
      />
      <TabPanel idPrefix="compare" id={tab}>
        {tab === "joint" ? (
          <JointEmbedding />
        ) : tab === "planner" ? (
          <PlannerTab key={prefill?.nonce ?? 0} prefill={prefill} />
        ) : (
          <CompositionTab key={prefill?.nonce ?? 0} prefill={prefill} />
        )}
      </TabPanel>
    </div>
  );
}

/** How many donors would a follow-up need, from the open dataset's own variability. */
function PlannerTab({ prefill }: { prefill: ComparePrefill | null }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  if (!dataset) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
        Load a dataset with replicate donors first: the planner simulates follow-up studies from its
        donor-to-donor variability.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-500">
        How many donors per group would a follow-up to <span className="font-medium">{dataset.name}</span>{" "}
        need? Power is estimated by simulating studies from this dataset&apos;s own donor-to-donor
        variability and running the app&apos;s own tests on them. More donors, not more cells, is what
        usually buys power.
      </p>
      <StudyPlanner
        key={dataset.id}
        datasetId={dataset.id}
        info={dataset}
        defaultCondition={prefill?.condition}
        defaultGroup={prefill?.group}
        defaultSample={prefill?.sample}
      />
    </div>
  );
}

type CompositionMode = "conditions" | "milo" | "milode" | "datasets";

/** When to use which test, shown with both. */
const WHICH_TEST =
  "Between conditions tests whether whole cell types or clusters change their share of the cells. " +
  "Neighbourhoods (Milo) tests small overlapping regions of the cell graph instead, so it finds " +
  "a shift inside part of a cluster or across cluster borders that a per-group test averages " +
  "away, and needs no clustering; a small, even shift of a whole type shows more clearly by group.";

/** Composition: a replicate-aware test between conditions of the open dataset
 *  (by group, or by neighbourhood with Milo), or two datasets side by side
 *  (descriptive). */
function CompositionTab({ prefill }: { prefill: ComparePrefill | null }) {
  const [mode, setMode] = useState<CompositionMode>((prefill && MODE_OF[prefill.tab]) || "conditions");
  const asked = {
    defaultCondition: prefill?.condition,
    defaultGroup: prefill?.group,
    defaultSample: prefill?.sample,
  };
  const dataset = useDatasetStore((s) => s.currentDataset);
  return (
    <div className="space-y-4">
      <div className="flex w-fit rounded-md border border-slate-200 p-0.5 text-sm">
        {(
          [
            ["conditions", "Between conditions"],
            ["milo", "Neighbourhoods (Milo)"],
            ["milode", "Neighbourhood DE (miloDE)"],
            ["datasets", "Two datasets"],
          ] as const
        ).map(([m, label]) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            aria-pressed={mode === m}
            className={`rounded px-3 py-1 font-medium transition ${
              mode === m ? "bg-primary text-white" : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {mode === "datasets" ? (
        <CompositionCompare />
      ) : dataset ? (
        <>
          <p className="text-sm text-slate-500">
            {mode === "milode" ? (
              <>
                Which genes change between two conditions, and where on <span className="font-medium">{dataset.name}</span>
                &apos;s cell graph? DE is tested inside each neighbourhood with each donor or sample as one
                replicate, so a change confined to part of a cell type, or crossing cluster borders, shows
                without clustering first.
              </>
            ) : mode === "milo" ? (
              <>
                Which regions of <span className="font-medium">{dataset.name}</span>&apos;s cell graph
                change in abundance between two conditions? Each donor or sample is one replicate.
              </>
            ) : (
              <>
                Do the cell-type or cluster proportions of <span className="font-medium">{dataset.name}</span>{" "}
                differ between conditions? Each donor or sample is one replicate.
              </>
            )}
          </p>
          {mode !== "milode" && <p className="text-xs text-slate-500">{WHICH_TEST}</p>}
          {mode === "milode" ? (
            <MiloDETest key={dataset.id} datasetId={dataset.id} info={dataset} />
          ) : mode === "milo" ? (
            <MiloTest key={dataset.id} datasetId={dataset.id} info={dataset} {...asked} />
          ) : (
            <CompositionTest key={dataset.id} datasetId={dataset.id} info={dataset} {...asked} />
          )}
        </>
      ) : (
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
          Load a dataset first to test its composition between conditions.
        </div>
      )}
    </div>
  );
}

/**
 * Compare two datasets by the composition of a shared categorical column
 * (e.g. cell-type proportions in dataset A vs B). Proportions are normalized so
 * datasets of different sizes are comparable.
 */
function CompositionCompare() {
  const palette = getPalette(useSettingsStore((s) => s.categoricalPalette));
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
      .catch((e) => !cancelled && setError(errorMessage(e)))
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
        `Each dataset is one sample here, so these differences are descriptive, not tested. ` +
        `What are the most notable differences and what might explain them?`,
    );
  }, [categories, datasetA, datasetB, propsA, propsB, column, askCopilot]);

  const otherDatasets = allDatasets.filter((d) => d.id !== datasetA?.id);

  if (!datasetA) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
        Load a dataset first, then pick a second one to compare against.
      </div>
    );
  }

  const aColor = `rgb(${palette[0]!.join(",")})`;
  const bColor = `rgb(${palette[3]!.join(",")})`;

  return (
    <div className="space-y-5">
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
            aria-label="Dataset B"
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
              aria-label="Compare by"
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
              config={plotConfig(`composition_${column}_vs_${datasetB.name}`, datasetA.name)}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>

          <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2.5 text-xs text-slate-600">
            Each dataset counts as one sample here, so these differences are descriptive: nothing can
            tell them apart from sample-to-sample variation. A chi-square or Fisher test on these cell
            counts would treat every cell as an independent replicate and overstate significance. To
            test, merge the datasets (Joint embedding) and use <span className="font-medium">Between
            conditions</span> with a donor or sample column.
          </p>

          {/* Δ table */}
          <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50 text-left text-slate-600">
                  <th className="px-4 py-2 font-medium">{prettyObsLabel(column)}</th>
                  <th className="px-4 py-2 text-right font-medium" title="Share of dataset A's cells with this value">A %</th>
                  <th className="px-4 py-2 text-right font-medium" title="Share of dataset B's cells with this value">B %</th>
                  <th className="px-4 py-2 text-right font-medium" title="Difference in percentage points, B minus A">Δ (B − A)</th>
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
          Embed several datasets together, or compare compositions: between conditions with
          samples as replicates (by group or by neighbourhood), or two datasets side by side.
        </p>
      </div>
    </div>
  );
}
