import { useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { InfoTip } from "@/components/common/InfoTip";
import { Caption } from "@/components/common/Caption";
import { RequestError } from "@/components/common/RequestError";
import { useDatasetStore } from "@/stores/datasetStore";
import { getCytokineActivity, type CytokineRequest, type CytokineResult } from "@/api/cytokines";
import { comparisonText, cytokineHeatmap } from "@/lib/cytokines";
import { plotConfig } from "@/lib/plotConfig";

interface Props {
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
}

// Diverging, white at 0, as the TF / pathway heatmap: red = the signature's response, blue = the opposite.
const Z_SCALE: [number, string][] = [
  [0, "#2166ac"],
  [0.5, "#f7f7f7"],
  [1, "#b2182b"],
];
const TOP_SHOWN = 3;

/**
 * Cytokine response (CytoSig): per group (vs the rest or all cells), or condition A vs B
 * within each group, the z-score of each of 43 cytokine-response signatures. Computed
 * on the server on demand; the first use downloads the signature.
 */
export function CytokineResponse({ groupByColumn, setGroupByColumn }: Props) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [mode, setMode] = useState<"groups" | "conditions">("groups");
  const [reference, setReference] = useState<"rest" | "all">("rest");
  const [condition, setCondition] = useState("");
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [result, setResult] = useState<CytokineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const groupColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter(
      (c) =>
        (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
        c.n_unique >= 1 &&
        c.n_unique <= 100 &&
        !(nCells > 0 && c.n_unique / nCells >= 0.9),
    );
  }, [dataset]);
  const conditionColumns = useMemo(
    () => groupColumns.filter((c) => c.n_unique >= 2 && c.n_unique <= 20 && (c.values?.length ?? 0) >= 2),
    [groupColumns],
  );
  const groupby = groupColumns.some((c) => c.name === groupByColumn) ? groupByColumn : "";
  const condValues = conditionColumns.find((c) => c.name === condition)?.values ?? [];

  // A new dataset: start over.
  useEffect(() => {
    setResult(null);
    setError(null);
    setCondition("");
  }, [datasetId]);
  // Default A / B to the condition's first two values.
  useEffect(() => {
    setA(condValues[1] ?? "");
    setB(condValues[0] ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [condition]);

  const ready = mode === "groups" ? !!groupby : !!condition && !!a && !!b && a !== b;

  const run = async () => {
    if (!datasetId || !ready) return;
    const body: CytokineRequest =
      mode === "groups"
        ? { mode, groupby, reference }
        : { mode, groupby: groupby === condition ? "" : groupby, condition, a, b };
    setLoading(true);
    setError(null);
    try {
      setResult(await getCytokineActivity(datasetId, body));
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const heat = useMemo(() => (result ? cytokineHeatmap(result) : null), [result]);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  const selectCls = "w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700";
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-medium text-slate-700">Cytokine response (CytoSig)</span>
        <InfoTip label="About cytokine response" width={300}>
          Which cytokines&apos; signalling the expression looks like. CytoSig (Jiang et al. 2021)
          collected the genes that change when cells are treated with each of 43 cytokines. A
          group&apos;s change in expression is fitted to those signatures (ridge regression), and
          each fit is compared with fits after shuffling the genes (z-score). A high z means the
          cells look like they <em>received</em> that signal; it says nothing about which cells
          make the cytokine. Human signatures: mouse genes are matched by symbol in upper case.
        </InfoTip>
      </div>

      <fieldset className="flex gap-3 text-xs text-slate-600">
        <legend className="sr-only">Compare</legend>
        <label className="flex items-center gap-1">
          <input type="radio" name="cytokine-mode" checked={mode === "groups"} onChange={() => setMode("groups")} />
          Each group vs the others
        </label>
        <label className="flex items-center gap-1">
          <input type="radio" name="cytokine-mode" checked={mode === "conditions"} onChange={() => setMode("conditions")} />
          Two conditions
        </label>
      </fieldset>

      <div className="grid grid-cols-2 gap-2">
        <label className="text-[11px] font-medium text-slate-500">
          {mode === "groups" ? "Groups" : "Within each group of (optional)"}
          <select value={groupby} onChange={(e) => setGroupByColumn(e.target.value)} className={`mt-1 ${selectCls}`}>
            <option value="">{mode === "groups" ? "Choose a column" : "All cells together"}</option>
            {groupColumns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.n_unique})
              </option>
            ))}
          </select>
        </label>
        {mode === "groups" ? (
          <label className="text-[11px] font-medium text-slate-500">
            Compared with
            <select value={reference} onChange={(e) => setReference(e.target.value as "rest" | "all")} className={`mt-1 ${selectCls}`}>
              <option value="rest">the rest of the cells</option>
              <option value="all">all cells</option>
            </select>
          </label>
        ) : (
          <label className="text-[11px] font-medium text-slate-500">
            Condition column
            <select value={condition} onChange={(e) => setCondition(e.target.value)} className={`mt-1 ${selectCls}`}>
              <option value="">Choose a column</option>
              {conditionColumns.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.n_unique})
                </option>
              ))}
            </select>
          </label>
        )}
        {mode === "conditions" && condition && (
          <>
            <label className="text-[11px] font-medium text-slate-500">
              Condition A
              <select value={a} onChange={(e) => setA(e.target.value)} className={`mt-1 ${selectCls}`}>
                {condValues.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </label>
            <label className="text-[11px] font-medium text-slate-500">
              vs condition B
              <select value={b} onChange={(e) => setB(e.target.value)} className={`mt-1 ${selectCls}`}>
                {condValues.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </label>
          </>
        )}
      </div>

      <button
        type="button"
        onClick={run}
        disabled={!ready || loading}
        className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
      >
        {loading ? (
          <span className="flex items-center gap-1.5">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Scoring… (the first use downloads the signature)
          </span>
        ) : (
          "Score cytokine response"
        )}
      </button>
      <RequestError message={error} />

      {result && !loading && (
        <div className="space-y-3">
          {heat && (
            <div style={{ height: Math.max(260, heat.y.length * 14 + 110) }}>
              <Plot
                data={[
                  {
                    type: "heatmap",
                    x: heat.x,
                    y: heat.y,
                    z: heat.z,
                    zmin: -heat.zmax, // symmetric: white is 0
                    zmax: heat.zmax,
                    colorscale: Z_SCALE,
                    colorbar: { title: { text: "z", font: { size: 9 } }, thickness: 8, tickfont: { size: 8 } },
                    hovertemplate: "%{y} in %{x}<br>z = %{z:.1f}<extra></extra>",
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 70, r: 6, t: 4, b: 90 },
                  xaxis: { type: "category", tickangle: -45, tickfont: { size: 9 }, automargin: true },
                  yaxis: { type: "category", tickfont: { size: 9 }, automargin: true },
                }}
                config={plotConfig("cytokine_response_heatmap")}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
              />
            </div>
          )}
          <Caption
            n={[
              { value: result.groups.length, unit: result.groups.length === 1 ? "group" : "groups" },
              { value: result.n_shared_genes, of: result.n_signature_genes, unit: "signature genes" },
            ]}
            testId="cytokine-caption"
          >
            CytoSig z-score per {result.mode === "conditions" ? "comparison" : "group"}: {comparisonText(result)}.
            Red: looks like a response to the cytokine; blue: the opposite.
          </Caption>

          <div>
            <h4 className="mb-1 text-[11px] font-medium text-slate-500">Top cytokines per group</h4>
            <ul className="space-y-0.5 text-xs" data-testid="cytokine-top">
              {result.top.map((t) => {
                const strong = t.cytokines.filter((c) => c.z >= 2).slice(0, TOP_SHOWN);
                return (
                  <li key={t.group} className="flex gap-2">
                    <span className="min-w-[6rem] truncate font-medium text-slate-700" title={t.group}>
                      {t.group}
                    </span>
                    <span className="text-slate-600">
                      {strong.length
                        ? strong.map((c) => `${c.cytokine} (z ${c.z.toFixed(1)})`).join(", ")
                        : <span className="text-slate-400">none with z ≥ 2</span>}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>

          {result.skipped.length > 0 && (
            <p className="text-[11px] text-amber-700">
              Left out: {result.skipped.map((s) => `${s.group} (${s.reason})`).join("; ")}.
            </p>
          )}
          {result.species === "mouse" && (
            <p className="text-[11px] text-amber-700">
              {result.mapped_by.includes("HCOP orthologs") && !result.mapped_by.includes("unavailable")
                ? `Mouse dataset: the human signatures were mapped to mouse genes through HCOP orthologs (${result.n_signature_genes.toLocaleString()} signature genes found).`
                : "Mouse dataset: the human signatures were matched by gene symbol in upper case (Ifng to IFNG); genes whose names differ between the species are missed."}
            </p>
          )}
          <p className="text-[10px] leading-snug text-slate-400">
            Ridge regression (lambda {result.lambda.toLocaleString()}) on {result.matrix_source},{" "}
            {result.n_permutations} gene shuffles; {result.elapsed_seconds}s{result.reused ? ", reused" : ""}.
            CytoSig: {result.citation}; signature data {result.licence}.
          </p>
        </div>
      )}
    </div>
  );
}
