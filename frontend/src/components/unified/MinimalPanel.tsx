import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Copy, ListPlus, Loader2, Square } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { MarkerPlot } from "@/components/plots/MarkerPlot";
import { Panel } from "@/components/common/Panel";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { plotConfig } from "@/lib/plotConfig";
import { plotlyColorscale } from "@/lib/colors";
import { useSettingsStore } from "@/stores/settingsStore";
import { useGeneListsStore } from "@/stores/geneListsStore";
import {
  PANEL_PER_GROUP_MAX,
  PANEL_SIZE_MAX,
  getMarkerPanel,
  runMarkerPanel,
  type BaselinePoint,
  type PanelResult,
  type PanelStart,
  type PanelStep,
} from "@/api/markerPanel";
import {
  confusionShares,
  curveLayout,
  curveTraces,
  hardestGroups,
  panelGenesText,
  panelText,
  parseGenes,
} from "@/lib/markerPanel";
import { errorMessage } from "@/lib/errorMessage";

const inputClass =
  "w-full rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700";
const buttonClass =
  "inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-2 py-0.5 text-xs text-slate-700 hover:bg-slate-50 disabled:opacity-50";

/**
 * "Minimal panel" section of the Markers subtab: the fewest genes that still tell
 * the column's groups apart (greedy forward selection on cross-validated balanced
 * accuracy), against the top markers taken in turn, with the confusion matrix and
 * a dot plot of the panel.
 */
export function MinimalPanel({ datasetId, groupby }: { datasetId: string; groupby: string }) {
  const [open, setOpen] = useState(false);
  const [size, setSize] = useState(20);
  const [perGroup, setPerGroup] = useState(10);
  const [required, setRequired] = useState("");
  const [excluded, setExcluded] = useState("");
  const [restrictId, setRestrictId] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [start, setStart] = useState<PanelStart | null>(null);
  const [steps, setSteps] = useState<PanelStep[]>([]);
  const [baseline, setBaseline] = useState<BaselinePoint[]>([]);
  const [result, setResult] = useState<PanelResult | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const lists = useGeneListsStore((s) => s.lists);
  const createList = useGeneListsStore((s) => s.create);
  const colormap = useSettingsStore((s) => s.continuousColormap);

  // A panel made earlier for this column (held by the server since the dataset loaded).
  useEffect(() => {
    if (!open || !datasetId) return;
    let cancelled = false;
    getMarkerPanel(datasetId)
      .then((r) => {
        if (!cancelled && r.result && r.result.groupby === groupby) {
          setResult(r.result);
          setSteps(r.result.path);
          setBaseline(r.result.baseline);
          setStart(r.result);
        }
      })
      .catch(() => {
        /* none held: nothing to show */
      });
    return () => {
      cancelled = true;
    };
  }, [open, datasetId, groupby]);

  // A different column: the shown panel is not of it.
  useEffect(() => {
    abort.current?.abort();
    setResult(null);
    setSteps([]);
    setBaseline([]);
    setStart(null);
    setError(null);
  }, [groupby, datasetId]);

  useEffect(() => () => abort.current?.abort(), []);

  const run = async () => {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setRunning(true);
    setError(null);
    setResult(null);
    setSteps([]);
    setBaseline([]);
    setStart(null);
    const restrict = lists.find((l) => l.id === restrictId)?.genes ?? null;
    try {
      await runMarkerPanel(
        datasetId,
        {
          groupby,
          size,
          per_group: perGroup,
          required: parseGenes(required),
          excluded: parseGenes(excluded),
          restrict,
        },
        {
          onStart: setStart,
          onStep: (s) => setSteps((prev) => [...prev, s]),
          onBaseline: setBaseline,
          onResult: setResult,
          signal: ctrl.signal,
        },
      );
    } catch (e) {
      if (!ctrl.signal.aborted) setError(errorMessage(e));
    } finally {
      if (abort.current === ctrl) {
        abort.current = null;
        setRunning(false);
      }
    }
  };

  const stop = () => {
    abort.current?.abort();
    abort.current = null;
    setRunning(false);
  };

  const flash = (text: string) => {
    setNotice(text);
    setTimeout(() => setNotice(null), 3000);
  };

  const copyGenes = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(panelGenesText(result));
      flash(`Copied ${result.panel.length} genes`);
    } catch {
      flash("Copy failed: the browser blocked the clipboard");
    }
  };

  const copySummary = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(panelText(result));
      flash("Copied the panel with its scores");
    } catch {
      flash("Copy failed: the browser blocked the clipboard");
    }
  };

  const saveList = () => {
    if (!result) return;
    const id = createList(`Minimal panel: ${groupby} (${result.panel.length})`, result.panel, "minimal marker panel");
    flash(id ? "Saved as a gene list (Gene lists tab)" : "Not saved: the gene list limit is reached");
  };

  const curve = useMemo(
    () => (steps.length ? { data: curveTraces(steps, baseline), layout: curveLayout(start?.size ?? size) } : null),
    [steps, baseline, start, size],
  );
  const confusion = useMemo(() => (result ? confusionShares(result.confusion.counts) : null), [result]);
  const last = steps[steps.length - 1];
  const heatHeight = result ? Math.max(220, 60 + 18 * result.groups.length) : 0;

  return (
    <section className="rounded-lg border border-slate-200" data-testid="minimal-panel">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-1 px-2 py-1.5 text-left text-xs font-semibold text-slate-600 hover:bg-slate-50"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        Minimal panel
        <span className="font-normal text-slate-500">: fewest genes that tell the groups apart</span>
      </button>
      {open && (
        <div className="flex flex-col gap-2 border-t border-slate-200 p-2 text-xs text-slate-600">
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-0.5">
              <span>Genes in the panel (max {PANEL_SIZE_MAX})</span>
              <input
                type="number"
                min={1}
                max={PANEL_SIZE_MAX}
                value={size}
                onChange={(e) => setSize(Math.min(PANEL_SIZE_MAX, Math.max(1, Number(e.target.value) || 1)))}
                className={inputClass}
              />
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="inline-flex items-center gap-1">
                Candidates per group
                <InfoTip label="About the candidates">
                  Each group&apos;s top markers as the app chooses them everywhere (expressed in at least 20% of
                  the group, adjusted p &lt; 0.05, by fold change, no ribosomal or mitochondrial genes).
                </InfoTip>
              </span>
              <input
                type="number"
                min={1}
                max={PANEL_PER_GROUP_MAX}
                value={perGroup}
                onChange={(e) => setPerGroup(Math.min(PANEL_PER_GROUP_MAX, Math.max(1, Number(e.target.value) || 1)))}
                className={inputClass}
              />
            </label>
            <label className="col-span-2 flex flex-col gap-0.5">
              <span>Required genes (always in the panel)</span>
              <input
                type="text"
                value={required}
                onChange={(e) => setRequired(e.target.value)}
                placeholder="e.g. CD3E, CD19"
                className={inputClass}
              />
            </label>
            <label className="col-span-2 flex flex-col gap-0.5">
              <span>Excluded genes</span>
              <input
                type="text"
                value={excluded}
                onChange={(e) => setExcluded(e.target.value)}
                placeholder="genes your assay cannot measure"
                className={inputClass}
              />
            </label>
            <label className="col-span-2 flex flex-col gap-0.5">
              <span>Only genes in a gene list (e.g. surface proteins)</span>
              <select value={restrictId} onChange={(e) => setRestrictId(e.target.value)} className={inputClass}>
                <option value="">Any gene</option>
                {lists.map((l) => (
                  <option key={l.id} value={l.id}>
                    {l.name} ({l.genes.length})
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="flex items-center gap-2">
            {running ? (
              <button type="button" onClick={stop} className={buttonClass}>
                <Square className="h-3 w-3" /> Stop
              </button>
            ) : (
              <button
                type="button"
                onClick={run}
                disabled={!groupby}
                className="rounded bg-primary px-2 py-0.5 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
              >
                Find panel
              </button>
            )}
            {running && (
              <span className="inline-flex items-center gap-1 text-slate-500" role="status">
                <Loader2 className="h-3 w-3 animate-spin" />
                {last
                  ? `${last.n_genes} of ${start?.size ?? size} genes: ${last.gene} (${last.balanced_accuracy.toFixed(2)})`
                  : start
                    ? `about ${Math.max(1, Math.round(start.estimate_seconds))} s`
                    : "Reading the candidates…"}
              </span>
            )}
          </div>
          {error && (
            <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
              {error}
            </p>
          )}
          {start?.notes.map((n) => (
            <p key={n} className="text-amber-700">
              {n}
            </p>
          ))}

          {result && (
            <p data-testid="panel-summary">
              <span className="font-semibold text-slate-700">{result.panel.length} genes</span> reach balanced
              accuracy <span className="font-semibold text-slate-700">{result.balanced_accuracy.toFixed(2)}</span>
              {result.baseline_accuracy != null && (
                <>
                  {" "}
                  (the top {result.baseline.length} markers taken in turn: {result.baseline_accuracy.toFixed(2)})
                </>
              )}
              . {result.best_n} genes come within {Math.round(result.within * 100)} point of the best. Hardest:{" "}
              {hardestGroups(result)
                .map((h) => `${h.group} ${h.recall.toFixed(2)}`)
                .join(", ")}
              .
            </p>
          )}

          {curve && (
            <Panel title="Accuracy vs genes" expandable className="h-[280px]" bodyClassName="p-1">
              <Plot
                data={curve.data}
                layout={curve.layout}
                config={plotConfig(`marker_panel_curve_${groupby}`)}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
              />
            </Panel>
          )}
          {start && steps.length > 0 && (
            <Caption n={[{ value: start.n_cells, unit: "cells" }, { value: start.groups.length, unit: "groups" }]}>
              Balanced accuracy (mean recall over groups) of a multinomial logistic regression, {start.cv}, on
              up to {start.cells_per_group} cells per group drawn with a fixed seed; z-scored log expression. At
              each step the {start.screen} most promising of {start.n_candidates} candidates (score test) are
              cross-validated and the best joins. Dashed: each group&apos;s top marker in turn, no selection.
            </Caption>
          )}

          {result && (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={copyGenes} className={buttonClass}>
                  <Copy className="h-3 w-3" /> Copy genes
                </button>
                <button type="button" onClick={copySummary} className={buttonClass}>
                  <Copy className="h-3 w-3" /> Copy as text
                </button>
                <button type="button" onClick={saveList} className={buttonClass}>
                  <ListPlus className="h-3 w-3" /> Save as gene list
                </button>
                {notice && (
                  <span role="status" className="text-slate-500">
                    {notice}
                  </span>
                )}
              </div>
              <table className="w-full text-left" data-testid="panel-genes">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="px-1 py-0.5 font-medium">#</th>
                    <th className="px-1 py-0.5 font-medium">Gene</th>
                    <th className="px-1 py-0.5 font-medium" title="Groups whose odds the gene raises most (standardised coefficient at least half its largest)">
                      Helps tell apart
                    </th>
                    <th className="px-1 py-0.5 text-right font-medium" title="Balanced accuracy once this gene is in the panel">
                      Accuracy
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.genes.map((g, i) => (
                    <tr key={g.gene} className="border-b border-slate-100">
                      <td className="px-1 py-0.5 tabular-nums text-slate-500">{i + 1}</td>
                      <td className="px-1 py-0.5 font-medium text-slate-700">
                        {g.gene}
                        {g.required && <span className="ml-1 text-[10px] text-slate-500">(required)</span>}
                      </td>
                      <td className="px-1 py-0.5" title={Object.entries(g.coef).map(([k, v]) => `${k}: ${v}`).join("\n")}>
                        {g.helps.join(", ") || "-"}
                      </td>
                      <td className="px-1 py-0.5 text-right tabular-nums">
                        {result.path[i]?.balanced_accuracy.toFixed(2) ?? ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <Caption n={{ value: result.panel.length, unit: "genes" }}>{result.importance_note} Hover a row for every group&apos;s coefficient.</Caption>

              {confusion && (
                <Panel title="Confusion (cross-validated)" expandable className="h-auto" bodyClassName="p-1">
                  <Plot
                    data={[
                      {
                        type: "heatmap",
                        x: result.groups,
                        y: result.groups,
                        z: confusion,
                        zmin: 0,
                        zmax: 1,
                        colorscale: plotlyColorscale(colormap),
                        colorbar: { title: { text: "share", font: { size: 10 } }, thickness: 10 },
                        customdata: result.confusion.counts as unknown as number[],
                        hovertemplate:
                          "true %{y}, called %{x}: %{z:.0%} (%{customdata} cells)<extra></extra>",
                        xgap: 1,
                        ygap: 1,
                      },
                    ]}
                    layout={{
                      height: heatHeight,
                      margin: { l: 10, r: 10, t: 6, b: 10 },
                      xaxis: { type: "category", tickangle: -45, automargin: true, title: { text: "called" } },
                      yaxis: { type: "category", automargin: true, autorange: "reversed", title: { text: "true group" } },
                      paper_bgcolor: "rgba(0,0,0,0)",
                      plot_bgcolor: "rgba(0,0,0,0)",
                    }}
                    config={plotConfig(`marker_panel_confusion_${groupby}`)}
                    style={{ width: "100%", height: heatHeight }}
                    useResizeHandler
                  />
                </Panel>
              )}
              <Caption n={{ value: result.n_cells, unit: "cells" }}>
                Each row: the share of a group&apos;s sampled cells the panel&apos;s model calls each group, each
                cell predicted by the model fitted on the other folds ({result.cv}).
              </Caption>

              <Panel title="Panel genes by group" expandable className="h-auto" bodyClassName="p-1">
                <MarkerPlot
                  datasetId={datasetId}
                  groupby={groupby}
                  groups={result.groups}
                  markers={{}}
                  kind="dotplot"
                  perGroup={1}
                  genes={result.panel}
                />
              </Panel>
            </>
          )}
        </div>
      )}
    </section>
  );
}
