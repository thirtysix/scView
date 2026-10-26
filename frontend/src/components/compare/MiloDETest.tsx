import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PlotMouseEvent } from "plotly.js";
import { AlertCircle, AlertTriangle, Download, Loader2, Play, Sparkles, Square } from "lucide-react";
import { getDataset } from "@/api/datasets";
import {
  miloDECells,
  miloDEGene,
  miloDENhood,
  planMiloDE,
  runMiloDE,
  type MiloDEGeneDetail,
  type MiloDEGenes,
  type MiloDENhoodDetail,
  type MiloDEPlan,
  type MiloDERequest,
  type MiloDEResult,
} from "@/api/milode";
import type { DatasetInfo } from "@/api/types";
import { Caption, CapNote } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { SortableTh, ariaSort } from "@/components/common/SortableTh";
import Plot from "@/components/plots/Plot";
import { DIVERGING_COLORSCALE } from "@/lib/cellHeatmap";
import { conditionColumns, suggestConditionColumn } from "@/lib/composition";
import { downloadCsv } from "@/lib/csv";
import { replicateColumns } from "@/lib/de";
import { formatPValue, prettyObsLabel } from "@/lib/formatting";
import {
  deNhoodRows,
  describeMiloDE,
  geneRange,
  genePoints,
  miloDEGeneCsv,
  miloDEQuestion,
  pickDELayout,
  sortGenes,
  type GeneSortField,
} from "@/lib/milode";
import { rankGroupingColumns } from "@/lib/obsColumns";
import { plotConfig } from "@/lib/plotConfig";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

interface Props {
  datasetId: string;
  info?: DatasetInfo | null;
}

const selectCls = "rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700";
const btnCls =
  "flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50 hover:text-primary";
const KS = [15, 20, 25, 30] as const;
const GENE_ROWS = 200;
const NHOOD_ROWS = 200;
const GREY = "#cbd5e1";

/**
 * Neighbourhood DE after miloDE (Missarova et al. 2024): DE between two
 * conditions inside each neighbourhood of the cell graph, donors as
 * replicates, no clusters needed. Plan, a streamed run with Cancel, a gene
 * table (how widespread each gene's DE is), one gene on the embedding, and a
 * neighbourhood table whose rows select their cells.
 */
export function MiloDETest({ datasetId, info }: Props) {
  const askCopilot = useViewStore((s) => s.askCopilot);
  const setPanel = useViewStore((s) => s.setPanel);
  const setSelection = useSelectionStore((s) => s.setSelection);
  const scatterEmbedding = useSettingsStore((s) => s.embedding);

  const [fetched, setFetched] = useState<DatasetInfo | null>(null);
  const dataset = info ?? fetched;
  const [conditionKey, setConditionKey] = useState("");
  const [reference, setReference] = useState("");
  const [tested, setTested] = useState("");
  const [sampleKey, setSampleKey] = useState("");
  const [labelKey, setLabelKey] = useState("");
  const [pairing, setPairing] = useState<"auto" | "paired" | "unpaired">("auto");
  const [genes, setGenes] = useState<MiloDEGenes>("hvg");
  const [k, setK] = useState<number>(25);
  const [plan, setPlan] = useState<MiloDEPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ message: string; done?: number; total?: number } | null>(null);
  const [result, setResult] = useState<MiloDEResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [layout, setLayout] = useState<string | null>(null);
  const [gene, setGene] = useState<string | null>(null);
  const [geneDetail, setGeneDetail] = useState<MiloDEGeneDetail | null>(null);
  const [geneError, setGeneError] = useState<string | null>(null);
  const [nhood, setNhood] = useState<MiloDENhoodDetail | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ field: GeneSortField; dir: "asc" | "desc" }>({ field: "n_de", dir: "desc" });
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    if (info) return;
    let cancelled = false;
    getDataset(datasetId)
      .then((d) => !cancelled && setFetched(d))
      .catch(() => !cancelled && setFetched(null));
    return () => {
      cancelled = true;
    };
  }, [datasetId, info]);
  useEffect(() => () => abort.current?.abort(), []);

  const cols = useMemo(() => dataset?.obs_columns ?? [], [dataset]);
  const labelOptions = useMemo(
    () =>
      rankGroupingColumns(cols, {
        nCells: dataset?.n_cells,
        activeClustering: dataset?.active_clustering,
        userColumns: dataset?.user_annotation_columns,
      }).map((g) => g.name),
    [cols, dataset],
  );
  const conditionOptions = useMemo(() => conditionColumns(cols, [labelKey]), [cols, labelKey]);
  const sampleOptions = useMemo(() => replicateColumns(cols, [conditionKey, labelKey]), [cols, conditionKey, labelKey]);
  const conditionValues = useMemo(() => cols.find((c) => c.name === conditionKey)?.values ?? [], [cols, conditionKey]);

  useEffect(() => {
    if (!cols.length) return;
    const label = labelOptions[0] ?? "";
    setLabelKey(label);
    setConditionKey(suggestConditionColumn(conditionColumns(cols, [label])));
  }, [cols, labelOptions]);

  useEffect(() => {
    setReference(conditionValues[0] ?? "");
    setTested(conditionValues[1] ?? "");
    setSampleKey("");
    setPairing("auto");
  }, [conditionKey, conditionValues]);

  const request = useMemo<MiloDERequest | null>(() => {
    if (!conditionKey) return null;
    return {
      condition_key: conditionKey,
      ...(sampleKey ? { sample_key: sampleKey } : {}),
      ...(reference && tested ? { conditions: [reference, tested] } : {}),
      ...(pairing !== "auto" ? { paired: pairing === "paired" } : {}),
      ...(labelKey ? { label_key: labelKey } : {}),
      k,
      genes,
    };
  }, [conditionKey, sampleKey, reference, tested, pairing, labelKey, k, genes]);

  // The plan builds the neighbourhoods (cached for the run): a second or two.
  useEffect(() => {
    if (!request) {
      setPlan(null);
      setPlanError(null);
      return;
    }
    if (conditionValues.length >= 2 && (!reference || !tested)) return;
    if (reference && reference === tested) {
      setPlan(null);
      setPlanError("Pick two different values to compare.");
      return;
    }
    let cancelled = false;
    setPlanning(true);
    planMiloDE(datasetId, request)
      .then((p) => {
        if (cancelled) return;
        setPlan(p);
        setPlanError(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setPlan(null);
        setPlanError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => !cancelled && setPlanning(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, request, reference, tested, conditionValues]);

  const run = useCallback(() => {
    if (!request) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setRunning(true);
    setRunError(null);
    setPicked(null);
    setNhood(null);
    setProgress({ message: "Starting…" });
    runMiloDE(datasetId, request, { onProgress: (p) => setProgress(p), signal: ctrl.signal })
      .then((r) => {
        if (ctrl.signal.aborted) return;
        setResult(r);
        setLayout((l) => (l && r.layouts[l] ? l : pickDELayout(r, scatterEmbedding)));
        setGene(r.genes.find((g) => g.n_de > 0)?.gene ?? r.genes[0]?.gene ?? null);
      })
      .catch((e) => {
        if (ctrl.signal.aborted) return;
        setRunError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (abort.current === ctrl) {
          setRunning(false);
          setProgress(null);
        }
      });
  }, [datasetId, request, scatterEmbedding]);

  const cancel = useCallback(() => {
    abort.current?.abort();
    abort.current = null;
    setRunning(false);
    setProgress(null);
    setRunError("Cancelled: the run stops at its next step.");
  }, []);

  // The picked gene in every neighbourhood.
  useEffect(() => {
    if (!result || !gene) {
      setGeneDetail(null);
      return;
    }
    let cancelled = false;
    setGeneError(null);
    miloDEGene(datasetId, gene)
      .then((d) => !cancelled && setGeneDetail(d))
      .catch((e) => {
        if (cancelled) return;
        setGeneDetail(null);
        setGeneError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, result, gene]);

  const rows = useMemo(() => (result ? deNhoodRows(result) : []), [result]);
  const nhoodOrder = useMemo(
    () => [...rows].sort((a, b) => Number(b.tested) - Number(a.tested) || b.nDe - a.nDe || a.id - b.id),
    [rows],
  );
  const geneRows = useMemo(
    () => (result ? sortGenes(result.genes, sort.field, sort.dir, query) : []),
    [result, sort, query],
  );

  const pickNhood = useCallback(
    (id: number) => {
      const row = rows[id];
      if (!row) return;
      miloDECells(datasetId, [id])
        .then(({ cells }) => {
          setSelection(new Set(cells));
          setPicked(`Selected ${cells.length.toLocaleString()} cells (neighbourhood ${id}${row.label ? `, ${row.label}` : ""}).`);
        })
        .catch((e) => setPicked(e instanceof Error ? e.message : String(e)));
      miloDENhood(datasetId, id)
        .then(setNhood)
        .catch(() => setNhood(null));
    },
    [datasetId, rows, setSelection],
  );

  const sortBy = (field: GeneSortField) =>
    setSort((s) => ({ field, dir: s.field === field ? (s.dir === "asc" ? "desc" : "asc") : field === "gene" ? "asc" : "desc" }));

  if (!dataset) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-sm text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading columns…
      </div>
    );
  }

  const stale =
    result &&
    plan &&
    (result.condition_key !== plan.condition_key || result.conditions.join() !== plan.conditions.join() ||
      result.sample_key !== plan.sample_key || result.formula !== plan.formula || result.k !== plan.k ||
      result.genes_mode !== plan.genes_mode);
  const nOneSided = rows.filter((r) => r.oneSided).length;

  return (
    <div className="space-y-4">
      <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Conditions</span>
            <select aria-label="Neighbourhood DE condition column" value={conditionKey}
              onChange={(e) => setConditionKey(e.target.value)} className={selectCls}>
              <option value="">Select a column…</option>
              {conditionOptions.map((c) => <option key={c.name} value={c.name}>{prettyObsLabel(c.name)}</option>)}
            </select>
          </label>
          {conditionValues.length > 0 && (
            <>
              <label className="text-xs font-medium text-slate-500">
                <span className="mb-1 block">Reference</span>
                <select aria-label="Neighbourhood DE reference" value={reference}
                  onChange={(e) => setReference(e.target.value)} className={selectCls}>
                  {conditionValues.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
              <label className="text-xs font-medium text-slate-500">
                <span className="mb-1 block">Tested</span>
                <select aria-label="Neighbourhood DE tested condition" value={tested}
                  onChange={(e) => setTested(e.target.value)} className={selectCls}>
                  {conditionValues.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            </>
          )}
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Samples (replicates)</span>
            <select aria-label="Neighbourhood DE sample column" value={sampleKey}
              onChange={(e) => setSampleKey(e.target.value)} className={selectCls}>
              <option value="">
                {!sampleKey && plan?.sample_key_detected ? `Auto (${prettyObsLabel(plan.sample_key)})` : "Auto"}
              </option>
              {sampleOptions.map((c) => <option key={c.name} value={c.name}>{prettyObsLabel(c.name)}</option>)}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Design</span>
            <select aria-label="Neighbourhood DE design" value={pairing}
              title="Paired adds the donor to the model: each donor is compared with itself"
              onChange={(e) => setPairing(e.target.value as "auto" | "paired" | "unpaired")} className={selectCls}>
              <option value="auto">{pairing === "auto" && plan ? `Auto (${plan.design})` : "Auto"}</option>
              <option value="paired">Paired by sample</option>
              <option value="unpaired">Unpaired</option>
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Genes</span>
            <select aria-label="Neighbourhood DE genes" value={genes}
              onChange={(e) => setGenes(e.target.value as MiloDEGenes)} className={selectCls}>
              <option value="hvg">Highly variable</option>
              <option value="expressed">All expressed</option>
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 flex items-center gap-1">
              Neighbours (k)
              <InfoTip label="About the neighbourhoods">
                As miloDE: a k-nearest-neighbour graph is built on the dataset&apos;s own embedding, and a
                neighbourhood is every cell within two steps of a sampled centre, a few hundred cells, so
                each holds enough cells per donor to sum. A larger k gives bigger, fewer neighbourhoods.
              </InfoTip>
            </span>
            <select aria-label="Neighbourhood DE k" value={String(k)}
              onChange={(e) => setK(Number(e.target.value))} className={selectCls}>
              {KS.map((v) => <option key={v} value={v}>{v === 25 ? "25 (miloDE)" : v}</option>)}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Label by</span>
            <select aria-label="Neighbourhood DE label column" value={labelKey}
              onChange={(e) => setLabelKey(e.target.value)} className={selectCls}>
              <option value="">(none)</option>
              {labelOptions.map((c) => <option key={c} value={c}>{prettyObsLabel(c)}</option>)}
            </select>
          </label>
          <div className="ml-auto flex gap-2">
            {running && (
              <button onClick={cancel}
                className="flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-600 hover:bg-slate-50">
                <Square className="h-3.5 w-3.5" /> Cancel
              </button>
            )}
            <button onClick={run} disabled={!plan || running || plan.n_testable === 0}
              className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Run neighbourhood DE
            </button>
          </div>
        </div>
        {planning && !plan && (
          <p className="flex items-center gap-1.5 text-xs text-slate-500">
            <Loader2 className="h-3 w-3 animate-spin" /> Building the neighbourhoods…
          </p>
        )}
        {plan && (
          <p className="text-xs text-slate-500" data-testid="milode-plan">
            {plan.n_cells.toLocaleString()} cells in{" "}
            {plan.conditions.map((c) => `${plan.n_samples[c]} ${c}`).join(" and ")} samples of{" "}
            {prettyObsLabel(plan.sample_key)}; {plan.formula} ({plan.design}). {plan.n_nhoods.toLocaleString()}{" "}
            neighbourhoods (cells within 2 steps on a {plan.k}-neighbour graph of {plan.representation}; median{" "}
            {Math.round(plan.median_nhood_size)} cells and {Math.round(plan.median_samples_per_nhood)} samples with{" "}
            {plan.min_cells_per_sample}+ cells), {plan.n_testable.toLocaleString()} testable;{" "}
            {plan.n_genes.toLocaleString()} {plan.genes_mode === "hvg" ? "highly variable" : "expressed"} genes; about{" "}
            {Math.max(1, Math.round(plan.estimate_seconds))} s.
            {plan.notes.map((n) => ` ${n}`)}
          </p>
        )}
      </div>

      {planError && <ErrorBox text={planError} />}
      {runError && <ErrorBox text={runError} />}
      {plan?.warnings.map((w) => !result && <WarnBox key={w} text={w} />)}
      {running && progress && (
        <div className="space-y-1.5 py-4 text-center text-sm text-slate-500" role="status">
          <div className="flex items-center justify-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin" /> {progress.message}
          </div>
          {progress.total ? (
            <progress className="h-1.5 w-64" value={progress.done ?? 0} max={progress.total} aria-label="Neighbourhoods tested" />
          ) : null}
        </div>
      )}

      {result && !running && (
        <>
          {result.warnings.map((w) => <WarnBox key={w} text={w} />)}
          {stale && <p className="text-xs text-amber-700">The settings changed since this run: run it again to update.</p>}
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <h4 className="text-sm font-semibold text-slate-700" data-testid="milode-headline">
                {result.n_genes_de.toLocaleString()} of {result.n_genes.toLocaleString()} genes DE in at least one of{" "}
                {result.n_tested.toLocaleString()} tested neighbourhoods (spatial FDR &lt; {result.alpha})
              </h4>
              <div className="ml-auto flex flex-wrap gap-1.5">
                <button className={btnCls} onClick={() => askCopilot(miloDEQuestion(result))}
                  title="Ask the co-pilot what these results suggest">
                  <Sparkles className="h-3 w-3" /> Ask
                </button>
                <button className={btnCls} title="Export the gene table as CSV"
                  onClick={() => {
                    const { headers, rows: csv } = miloDEGeneCsv(result);
                    downloadCsv(`milode_${result.condition_key}_${result.conditions[1]}_vs_${result.conditions[0]}_${datasetId}.csv`, headers, csv);
                  }}>
                  <Download className="h-3 w-3" /> CSV
                </button>
              </div>
            </div>
            <Caption>
              {describeMiloDE(result)}
              {result.notes.map((n) => ` ${n}`)}
              {result.dropped_samples.length > 0 && ` Left out (under 10 cells): ${result.dropped_samples.join(", ")}.`}
            </Caption>
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-2 flex items-center gap-2">
                <h5 className="text-sm font-medium text-slate-700">Genes</h5>
                <InfoTip label="About the gene table">
                  How widespread each gene&apos;s DE is: in how many of the neighbourhoods where it was tested it
                  is DE (spatial FDR &lt; {result.alpha}), and its median log2 fold change over them. A gene DE
                  in nearly all neighbourhoods changes everywhere; one DE in a few changes in one region.
                </InfoTip>
                <input type="search" aria-label="Filter genes" placeholder="Filter genes" value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="ml-auto w-36 rounded border border-slate-300 px-2 py-0.5 text-xs" />
              </div>
              <div className="max-h-96 overflow-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-slate-50 text-slate-600">
                    <tr>
                      <SortableTh onSort={() => sortBy("gene")} sort={ariaSort("gene", sort.field, sort.dir)} className="px-2 py-1.5">Gene</SortableTh>
                      <SortableTh onSort={() => sortBy("n_de")} sort={ariaSort("n_de", sort.field, sort.dir)} align="right"
                        className="px-2 py-1.5" title={`Neighbourhoods where the gene is DE (spatial FDR < ${result.alpha}) / where it was tested`}>
                        DE / tested
                      </SortableTh>
                      <SortableTh onSort={() => sortBy("share")} sort={ariaSort("share", sort.field, sort.dir)} align="right"
                        className="px-2 py-1.5" title="Share of the tested neighbourhoods where it is DE">
                        Share
                      </SortableTh>
                      <th className="px-2 py-1.5 text-right font-medium" title={`DE neighbourhoods higher / lower in ${result.conditions[1]}`}>Up / down</th>
                      <SortableTh onSort={() => sortBy("median_logfc")} sort={ariaSort("median_logfc", sort.field, sort.dir)} align="right"
                        className="px-2 py-1.5" title={`Median log2 fold change (${result.conditions[1]} vs ${result.conditions[0]}) over the tested neighbourhoods`}>
                        Median log2FC
                      </SortableTh>
                    </tr>
                  </thead>
                  <tbody>
                    {geneRows.slice(0, GENE_ROWS).map((g) => (
                      <tr key={g.gene} onClick={() => setGene(g.gene)} aria-selected={g.gene === gene}
                        className={`cursor-pointer border-b border-slate-50 hover:bg-slate-50 ${g.gene === gene ? "bg-primary/10" : ""}`}>
                        <td className="px-2 py-1">
                          <button type="button" className="font-medium text-slate-800 hover:text-primary" onClick={() => setGene(g.gene)}>
                            {g.gene}
                          </button>
                        </td>
                        <td className="px-2 py-1 text-right tabular-nums">{g.n_de} / {g.n_tested}</td>
                        <td className="px-2 py-1 text-right tabular-nums">{Math.round(g.share * 100)}%</td>
                        <td className="px-2 py-1 text-right tabular-nums">{g.n_up} / {g.n_down}</td>
                        <td className="px-2 py-1 text-right tabular-nums">{g.median_logfc == null ? "–" : g.median_logfc.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <CapNote shown={Math.min(GENE_ROWS, geneRows.length)} total={geneRows.length} by="the sorted column" rest="the CSV has all" />
              <Caption n={{ value: geneRows.length, unit: "genes", of: result.genes.length }}>Click a gene to show it on the embedding.</Caption>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
                <span>Layout</span>
                <select aria-label="Neighbourhood DE layout" value={layout ?? ""} onChange={(e) => setLayout(e.target.value)}
                  className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700">
                  {Object.keys(result.layouts).map((key) => <option key={key} value={key}>{key}</option>)}
                </select>
              </div>
              {geneError && <ErrorBox text={geneError} />}
              {geneDetail && layout ? (
                <GenePlot result={result} detail={geneDetail} layout={layout} datasetName={dataset.name} onPick={pickNhood} />
              ) : (
                <p className="py-12 text-center text-sm text-slate-400">Pick a gene.</p>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 flex items-center gap-2">
              <h5 className="text-sm font-medium text-slate-700">Neighbourhoods</h5>
              <InfoTip label="About the neighbourhood table">
                Genes DE in each neighbourhood: by the spatial FDR across neighbourhoods (as the gene table),
                and by Benjamini-Hochberg within the neighbourhood alone. Click a row to select its cells and
                list its genes. &quot;Few cells of one condition&quot;: under {Math.round(result.one_sided * 100)}% of
                its cells come from one condition, so its fold changes rest on a handful of cells per sample.
              </InfoTip>
            </div>
            {picked && (
              <p className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-600" data-testid="milode-picked">
                {picked}
                <button className="text-primary underline" onClick={() => setPanel("unified")}>Show on the scatter</button>
              </p>
            )}
            <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
              <div className="max-h-80 overflow-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-slate-50 text-left text-slate-600">
                    <tr>
                      <th className="px-2 py-1.5 font-medium">#</th>
                      {result.label_key && <th className="px-2 py-1.5 font-medium">{prettyObsLabel(result.label_key)}</th>}
                      <th className="px-2 py-1.5 text-right font-medium" title={`Cells: ${result.conditions.join(" / ")}`}>Cells</th>
                      <th className="px-2 py-1.5 text-right font-medium" title={`Samples with ${result.min_cells_per_sample}+ cells here`}>Samples</th>
                      <th className="px-2 py-1.5 text-right font-medium" title={`Genes DE at spatial FDR < ${result.alpha}`}>DE genes</th>
                      <th className="px-2 py-1.5 text-right font-medium" title={`Genes DE by BH within this neighbourhood (< ${result.alpha})`}>DE within</th>
                    </tr>
                  </thead>
                  <tbody>
                    {nhoodOrder.slice(0, NHOOD_ROWS).map((n) => (
                      <tr key={n.id} onClick={() => pickNhood(n.id)} aria-selected={nhood?.nhood === n.id}
                        className={`cursor-pointer border-b border-slate-50 hover:bg-slate-50 ${nhood?.nhood === n.id ? "bg-primary/10" : ""} ${n.tested ? "" : "text-slate-400"}`}>
                        <td className="px-2 py-1">
                          <button type="button" className="hover:text-primary" onClick={(e) => { e.stopPropagation(); pickNhood(n.id); }}>
                            {n.id}
                          </button>
                        </td>
                        {result.label_key && <td className="px-2 py-1">{n.label}</td>}
                        <td className="px-2 py-1 text-right tabular-nums">
                          {n.cells.join(" / ")}
                          {n.oneSided && <span className="ml-1 text-amber-700" title="Under 10% of its cells from one condition">(few cells of one condition)</span>}
                        </td>
                        <td className="px-2 py-1 text-right tabular-nums">{n.nSamples}</td>
                        <td className="px-2 py-1 text-right tabular-nums">{n.tested ? n.nDe : "not tested"}</td>
                        <td className="px-2 py-1 text-right tabular-nums">{n.tested ? n.nDeWithin : ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <CapNote shown={Math.min(NHOOD_ROWS, nhoodOrder.length)} total={nhoodOrder.length} by="DE genes" />
                <Caption n={[{ value: result.n_tested, unit: "tested", of: result.n_nhoods }, { value: nOneSided, unit: "with few cells of one condition" }]}>
                  Neighbourhoods, most DE genes first.
                </Caption>
              </div>
              <div>
                {nhood ? (
                  <>
                    <p className="mb-1 text-xs font-medium text-slate-600" data-testid="milode-nhood">
                      Neighbourhood {nhood.nhood}{nhood.label ? ` (${nhood.label})` : ""}: {nhood.size.toLocaleString()} cells,{" "}
                      {nhood.tested ? `${nhood.n_de} genes DE (${nhood.n_de_within} within)` : "not tested"}
                    </p>
                    <table className="w-full text-xs">
                      <thead className="text-left text-slate-500">
                        <tr>
                          <th className="py-1 font-medium">Gene</th>
                          <th className="py-1 text-right font-medium">log2FC</th>
                          <th className="py-1 text-right font-medium" title="Benjamini-Hochberg within this neighbourhood">FDR</th>
                          <th className="py-1 text-right font-medium" title="Spatial FDR across neighbourhoods">Spatial FDR</th>
                        </tr>
                      </thead>
                      <tbody>
                        {nhood.genes.slice(0, 15).map((g) => (
                          <tr key={g.gene} className="border-b border-slate-50">
                            <td className="py-0.5">
                              <button type="button" className="hover:text-primary" onClick={() => setGene(g.gene)}>{g.gene}</button>
                            </td>
                            <td className="py-0.5 text-right tabular-nums">{g.log_fc == null ? "–" : g.log_fc.toFixed(2)}</td>
                            <td className="py-0.5 text-right tabular-nums">{g.fdr == null ? "–" : formatPValue(g.fdr)}</td>
                            <td className="py-0.5 text-right tabular-nums">{g.spatial_fdr == null ? "–" : formatPValue(g.spatial_fdr)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <Caption n={{ value: Math.min(15, nhood.genes.length), unit: "genes", of: nhood.n_genes_tested }}>
                      Smallest p first.
                    </Caption>
                  </>
                ) : (
                  <p className="py-8 text-center text-xs text-slate-400">Click a neighbourhood to list its genes.</p>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/** One gene on the embedding: a point per tested neighbourhood at its index
 *  cell, coloured by log2FC where DE, grey otherwise. */
function GenePlot({ result, detail, layout, datasetName, onPick }: {
  result: MiloDEResult;
  detail: MiloDEGeneDetail;
  layout: string;
  datasetName: string;
  onPick: (id: number) => void;
}) {
  const range = useMemo(() => geneRange(detail), [detail]);
  const traces = useMemo(() => {
    const pos = result.layouts[layout] ?? [];
    const { sig, grey } = genePoints(result, detail, layout);
    const maxSize = Math.max(1, ...result.nhoods.size);
    const size = (i: number) => 5 + 13 * Math.sqrt((result.nhoods.size[i] ?? 0) / maxSize);
    const hover = (i: number) => {
      const fc = detail.log_fc[i];
      const fdr = detail.spatial_fdr[i];
      const label = result.nhoods.label?.[i];
      return `${label ? `${label}<br>` : ""}neighbourhood ${i}, ${result.nhoods.size[i]} cells<br>` +
        `log2FC ${fc == null ? "–" : fc.toFixed(2)}, spatial FDR ${fdr == null ? "–" : formatPValue(fdr)}`;
    };
    const part = (ids: number[], isSig: boolean) => ({
      type: "scatter" as const,
      mode: "markers" as const,
      name: isSig ? `spatial FDR < ${detail.alpha}` : "not DE",
      x: ids.map((i) => pos[i]![0]!),
      y: ids.map((i) => pos[i]![1]!),
      customdata: ids,
      text: ids.map(hover),
      hovertemplate: "%{text}<extra></extra>",
      marker: isSig
        ? {
            size: ids.map(size), color: ids.map((i) => detail.log_fc[i]!), colorscale: DIVERGING_COLORSCALE,
            cmin: -range, cmax: range, line: { width: 0.5, color: "#334155" },
            colorbar: { title: { text: "log2FC", side: "right" as const }, thickness: 10, len: 0.6 },
          }
        : { size: ids.map(size), color: GREY, line: { width: 0.3, color: "#94a3b8" } },
    });
    return [part(grey, false), part(sig, true)];
  }, [result, detail, layout, range]);
  const nSig = detail.significant.filter(Boolean).length;
  const nTested = detail.log_fc.filter((v) => v != null).length;
  const axis = { showgrid: false, zeroline: false, showticklabels: false, title: { text: "" } };
  return (
    <>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 420,
          margin: { t: 34, r: 16, b: 16, l: 16 },
          title: {
            text: `${detail.gene}: ${result.conditions[1]} vs ${result.conditions[0]} by neighbourhood`,
            font: { size: 13, color: "#334155" },
          },
          xaxis: axis,
          yaxis: { ...axis, scaleanchor: "x" },
          hovermode: "closest",
          showlegend: false,
          paper_bgcolor: "white",
          plot_bgcolor: "white",
        }}
        config={plotConfig(`milode_${detail.gene}_${layout}`, datasetName)}
        onClick={(e: Readonly<PlotMouseEvent>) => {
          const id = e.points?.[0]?.customdata;
          if (typeof id === "number") onPick(id);
        }}
        useResizeHandler
        style={{ width: "100%" }}
      />
      <Caption n={[{ value: nSig, unit: "DE", of: nTested }, { value: nTested, unit: "tested neighbourhoods" }]}>
        One point per neighbourhood at its centre cell on {layout.replace(/^X_/, "").toUpperCase()}, sized by its
        cells; grey: tested, not DE. Click one to select its cells.
      </Caption>
    </>
  );
}

function ErrorBox({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <AlertCircle className="h-4 w-4 flex-shrink-0" />
      {text}
    </div>
  );
}

function WarnBox({ text }: { text: string }) {
  return (
    <div role="alert" className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm text-amber-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
      <span>{text}</span>
    </div>
  );
}
