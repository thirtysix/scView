import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import type { Data, Layout } from "plotly.js";
import {
  attachVelocity,
  getVelocity,
  getVelocityGenes,
  getVelocityPhase,
  runVelocity,
  type VelocityGene,
  type VelocityMethod,
  type VelocityPhase,
  type VelocityRun,
  type VelocitySource,
  type VelocityStatus,
} from "@/api/velocity";
import { Caption, capText } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import Plot from "@/components/plots/Plot";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { useEmbedding } from "@/hooks/useEmbedding";
import { mapCategoryToColor } from "@/lib/colors";
import { errorMessage } from "@/lib/errorMessage";
import { aboutGb } from "@/lib/labelTransfer";
import { labelColumns } from "@/lib/spatialDomains";
import { plotConfig } from "@/lib/plotConfig";
import { ARROW_DENSITY_RANGE, ARROW_SCALE_RANGE, arrowsBlocked, isOtherPcPair } from "@/lib/velocityArrows";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const GENE_LIMIT = 100;

function matchLine(s: VelocitySource): string {
  const cells = s.cells_matched != null && s.cells_in_dataset != null
    ? `${s.cells_matched.toLocaleString()} of ${s.cells_in_dataset.toLocaleString()} cells matched`
    : null;
  const byId = s.genes_by_id ?? 0;
  const byName = s.genes_by_name ?? 0;
  const genes = s.genes_in_file != null
    ? `${(byId + byName).toLocaleString()} of ${s.genes_in_file.toLocaleString()} genes in the dataset (${byId.toLocaleString()} by id, ${byName.toLocaleString()} by name)`
    : null;
  return [cells, genes].filter(Boolean).join("; ");
}

function phaseTraces(p: VelocityPhase, palette: Parameters<typeof mapCategoryToColor>[1]): Data[] {
  const marker = { size: 4, opacity: 0.7 };
  const traces: Data[] = [];
  const color = p.color;
  if (color?.kind === "category") {
    color.categories.forEach((name, code) => {
      const take = color.codes.flatMap((c, i) => (c === code ? [i] : []));
      if (!take.length) return;
      const [r, g, b] = mapCategoryToColor(code, palette);
      traces.push({
        type: "scattergl",
        mode: "markers",
        name,
        x: take.map((i) => p.spliced[i]!),
        y: take.map((i) => p.unspliced[i]!),
        marker: { ...marker, color: `rgb(${r},${g},${b})` },
      });
    });
  } else {
    traces.push({
      type: "scattergl",
      mode: "markers",
      name: "cells",
      showlegend: false,
      x: p.spliced,
      y: p.unspliced,
      marker: color?.kind === "numeric"
        ? { ...marker, color: color.values.map((v) => v ?? NaN), colorscale: "Viridis", showscale: true, colorbar: { title: { text: color.column }, thickness: 10 } }
        : { ...marker, color: "#64748b" },
    });
  }
  const xmax = p.spliced.reduce((a, b) => Math.max(a, b), 0);
  traces.push({
    type: "scatter",
    mode: "lines",
    name: `gamma ${p.gamma.toPrecision(3)}`,
    x: [0, xmax],
    y: [0, p.gamma * xmax],
    line: { color: "#dc2626", width: 2 },
  });
  return traces;
}

/**
 * RNA velocity (backend core/velocity.py): counts attached or carried by the upload, a run
 * (steady state or stochastic, as scVelo), arrows on the main plot, what to judge them by.
 */
export function VelocitySubtab() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const palette = useSettingsStore((s) => s.categoricalPalette);
  const pcaDims = useSettingsStore((s) => s.pcaDims);
  const embedding = useSettingsStore((s) => s.embedding);
  const { dimensions } = useEmbedding();
  const refresh = useDatasetRefresh();
  const arrows = useUnifiedViewStore((s) => s.velocityArrows);
  const setArrows = useUnifiedViewStore((s) => s.setVelocityArrows);
  const setVelocityRun = useUnifiedViewStore((s) => s.setVelocityRun);
  const gridNote = useUnifiedViewStore((s) => s.velocityGridNote);
  const [status, setStatus] = useState<VelocityStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attaching, setAttaching] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [memoryNote, setMemoryNote] = useState<string | null>(null);
  const [method, setMethod] = useState<VelocityMethod>("stochastic");
  const [nTop, setNTop] = useState(2000);
  const [nNeighbors, setNNeighbors] = useState(30);
  const [groupBy, setGroupBy] = useState("");
  const [genes, setGenes] = useState<VelocityGene[] | null>(null);
  const [gene, setGene] = useState<string | null>(null);
  const [phase, setPhase] = useState<VelocityPhase | null>(null);
  const [phaseError, setPhaseError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const load = async (id: string) => {
    const s = await getVelocity(id);
    setStatus(s);
    setVelocityRun(s.run && !s.stale ? { datasetId: id, id: s.run.result_id } : null);
    return s;
  };

  useEffect(() => {
    if (!datasetId) return;
    setStatus(null);
    setLoadError(null);
    setError(null);
    setGenes(null);
    setGene(null);
    setPhase(null);
    let live = true;
    getVelocity(datasetId)
      .then((s) => {
        if (!live) return;
        setStatus(s);
        setVelocityRun(s.run && !s.stale ? { datasetId, id: s.run.result_id } : null);
        if (s.run) {
          setMethod(s.run.method);
          setNTop(s.run.params.n_top_genes);
          setNNeighbors(s.run.params.n_neighbors);
          setGroupBy(s.run.group_column ?? "");
        }
      })
      .catch((e) => live && setLoadError(errorMessage(e)));
    return () => {
      live = false;
      abort.current?.abort();
    };
  }, [datasetId, setVelocityRun]);

  const runId = status?.run?.result_id ?? null;
  useEffect(() => {
    setGenes(null);
    setGene(null);
    setPhase(null);
    if (!datasetId || !runId || status?.stale) return;
    let live = true;
    getVelocityGenes(datasetId)
      .then((g) => {
        if (!live) return;
        setGenes(g.genes);
        setGene(g.genes[0]?.gene ?? null);
      })
      .catch((e) => live && setPhaseError(errorMessage(e)));
    return () => {
      live = false;
    };
  }, [datasetId, runId, status?.stale]);

  useEffect(() => {
    setPhaseError(null);
    if (!datasetId || !gene) return;
    let live = true;
    getVelocityPhase(datasetId, gene, colorBy)
      .then((p) => live && setPhase(p))
      .catch((e) => live && setPhaseError(errorMessage(e)));
    return () => {
      live = false;
    };
  }, [datasetId, gene, colorBy, runId]);

  const traces = useMemo(() => (phase ? phaseTraces(phase, palette) : []), [phase, palette]);
  const layout = useMemo<Partial<Layout>>(() => ({
    height: 300,
    margin: { l: 50, r: 10, t: 24, b: 40 },
    title: { text: phase ? phase.gene : "", font: { size: 12 } },
    xaxis: { title: { text: "Spliced (first moment)" }, rangemode: "tozero", zeroline: false },
    yaxis: { title: { text: "Unspliced (first moment)" }, rangemode: "tozero", zeroline: false },
    legend: { font: { size: 10 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
  }), [phase]);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  if (loadError) return <p className="text-xs text-red-600">{loadError}</p>;
  if (!status) return <div className="text-xs text-slate-400">Loading...</div>;

  const run: VelocityRun | null = status.run;
  const est = status.estimate;
  const groupColumns = labelColumns(dataset.obs_columns ?? []);
  const blocked = arrowsBlocked(dimensions, embedding, isOtherPcPair(embedding, pcaDims));

  const attach = async (files: File[]) => {
    if (!files.length) return;
    setAttaching(true);
    setError(null);
    try {
      await attachVelocity(datasetId, files);
      await load(datasetId);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setAttaching(false);
    }
  };

  const start = async () => {
    setRunning(true);
    setError(null);
    setMemoryNote(null);
    setProgress("Starting");
    abort.current = new AbortController();
    try {
      await runVelocity(
        datasetId,
        { method, n_top_genes: nTop, n_neighbors: nNeighbors, group_by: groupBy || null },
        { signal: abort.current.signal, onProgress: setProgress, onWatchdog: setMemoryNote },
      );
      await refresh(datasetId);
      await load(datasetId);
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(errorMessage(e));
    } finally {
      setRunning(false);
      setProgress("");
    }
  };

  const colourByConfidence = () => {
    useUnifiedViewStore.getState().clearOverlayState();
    useSettingsStore.getState().setColorBy("velocity_confidence");
  };

  const shownGenes = genes?.slice(0, GENE_LIMIT) ?? [];
  const genesCap = genes ? capText({ shown: shownGenes.length, total: genes.length, by: "velocity genes first, then r2" }) : null;

  return (
    <div className="space-y-3 text-xs text-slate-700">
      {!status.attached && (
        <div className="rounded-md border border-slate-200 bg-slate-50 p-2.5">
          <p className="font-medium">This dataset has no spliced and unspliced counts yet.</p>
          <p className="mt-1 text-slate-600">
            Bring what one of these made from the same BAM files: <b>velocyto</b> (a .loom), <b>STARsolo</b> with{" "}
            <code>--soloFeatures Velocyto</code> (the files of <code>Solo.out/Velocyto/filtered</code>), or <b>kb</b> with{" "}
            <code>--workflow lamanno</code> (an .h5ad with layers spliced and unspliced).
          </p>
        </div>
      )}

      {status.attached && status.source && (
        <div className="rounded-md border border-slate-200 p-2">
          <p>
            Counts from <b>{status.source.file ?? "the upload"}</b>
            {status.source.matrices ? ` (${status.source.matrices.join(", ")})` : ""}.
          </p>
          {matchLine(status.source) && <p className="text-slate-600">{matchLine(status.source)}.</p>}
          {!!status.source.cells_without_now && (
            <p className="text-amber-700">
              {status.source.cells_without_now.toLocaleString()} of this dataset's cells have no velocity counts and get no arrow.
            </p>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <label className="cursor-pointer rounded-md border border-slate-300 bg-white px-2 py-1 hover:bg-slate-50">
          {status.attached ? "Attach other counts" : "Attach a file"} (.loom, .h5ad)
          <input
            type="file"
            accept=".loom,.h5ad"
            className="hidden"
            disabled={attaching || running}
            onChange={(e) => {
              void attach(Array.from(e.target.files ?? []));
              e.target.value = "";
            }}
          />
        </label>
        <label className="cursor-pointer rounded-md border border-slate-300 bg-white px-2 py-1 hover:bg-slate-50">
          STARsolo files
          <input
            type="file"
            multiple
            accept=".mtx,.tsv,.gz"
            aria-label="STARsolo Velocyto/filtered files"
            className="hidden"
            disabled={attaching || running}
            onChange={(e) => {
              void attach(Array.from(e.target.files ?? []));
              e.target.value = "";
            }}
          />
        </label>
        {attaching && (
          <span className="flex items-center gap-1 text-slate-500">
            <Loader2 className="h-3 w-3 animate-spin" /> Matching cells and genes
          </span>
        )}
      </div>

      {status.attached && (
        <div className="space-y-2 rounded-md border border-slate-200 p-2">
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-0.5">
              <span className="text-slate-500">Model</span>
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value as VelocityMethod)}
                className="rounded border border-slate-300 bg-white px-1 py-0.5"
              >
                {status.methods.map((m) => (
                  <option key={m} value={m}>{status.method_names[m]}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="text-slate-500">Table by</span>
              <select
                value={groupBy}
                onChange={(e) => setGroupBy(e.target.value)}
                className="rounded border border-slate-300 bg-white px-1 py-0.5"
              >
                <option value="">The clustering</option>
                {groupColumns.map((c) => (
                  <option key={c.name} value={c.name}>{c.name}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="text-slate-500">Top genes</span>
              <input
                type="number"
                min={100}
                max={20000}
                step={100}
                value={nTop}
                onChange={(e) => setNTop(Number(e.target.value))}
                className="rounded border border-slate-300 bg-white px-1 py-0.5"
              />
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="text-slate-500">Neighbours</span>
              <input
                type="number"
                min={5}
                max={100}
                value={nNeighbors}
                onChange={(e) => setNNeighbors(Number(e.target.value))}
                className="rounded border border-slate-300 bg-white px-1 py-0.5"
              />
            </label>
          </div>
          {est && (
            <p className="text-slate-500">
              {est.n_cells.toLocaleString()} cells: {est.note}, {aboutGb(est.peak_bytes)} of memory.
            </p>
          )}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={running || attaching || nTop < 100 || nTop > 20000 || nNeighbors < 5 || nNeighbors > 100}
              onClick={() => void start()}
              className="rounded-md bg-blue-600 px-2.5 py-1 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {run ? "Run again" : "Run velocity"}
            </button>
            {running && (
              <>
                <button
                  type="button"
                  onClick={() => abort.current?.abort()}
                  className="rounded-md border border-slate-300 px-2 py-1 hover:bg-slate-50"
                >
                  Cancel
                </button>
                <span className="flex items-center gap-1 text-slate-500">
                  <Loader2 className="h-3 w-3 animate-spin" /> {progress}
                </span>
              </>
            )}
          </div>
          {memoryNote && <p className="text-amber-700">{memoryNote}</p>}
          <p className="text-slate-500">
            Steady state and stochastic, as scVelo 0.3.3. The dynamical model and latent time are not offered: they take
            minutes per thousand cells, and that fit has not been validated here.
          </p>
        </div>
      )}

      {error && <p className="text-red-600">{error}</p>}

      {status.stale && <p className="rounded-md border border-amber-300 bg-amber-50 p-2 text-amber-800">{status.stale}</p>}

      {run && !status.stale && (
        <>
          <div className="space-y-1 rounded-md border border-slate-200 p-2">
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-1">
                <input type="checkbox" checked={arrows.on} onChange={(e) => setArrows({ on: e.target.checked })} />
                Arrows on the plot
              </label>
              <label className="flex items-center gap-1">
                Scale
                <input
                  type="range"
                  aria-label="Arrow scale"
                  min={ARROW_SCALE_RANGE[0]}
                  max={ARROW_SCALE_RANGE[1]}
                  step={0.25}
                  value={arrows.scale}
                  onChange={(e) => setArrows({ scale: Number(e.target.value) })}
                />
              </label>
              <label className="flex items-center gap-1">
                Density
                <input
                  type="range"
                  aria-label="Arrow density"
                  min={ARROW_DENSITY_RANGE[0]}
                  max={ARROW_DENSITY_RANGE[1]}
                  step={0.2}
                  value={arrows.density}
                  onChange={(e) => setArrows({ density: Number(e.target.value) })}
                />
              </label>
              <button
                type="button"
                onClick={colourByConfidence}
                className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-50"
              >
                Colour by confidence
              </button>
            </div>
            {blocked && <p className="text-amber-700">{blocked}</p>}
            {gridNote && !blocked && <p className="text-red-600">{errorMessage(new Error(gridNote))}</p>}
            <p className="text-slate-600">{run.note}</p>
          </div>

          {run.warnings.map((w) => (
            <p key={w} className="rounded-md border border-amber-300 bg-amber-50 p-2 text-amber-800">{w}</p>
          ))}

          <Caption n={[{ value: run.n_cells, unit: "cells", of: run.n_dataset_cells }, { value: run.n_velocity_genes, unit: "velocity genes", of: run.n_genes_fit }]}>
            {run.method_name}. Median confidence {run.median_confidence.toFixed(2)}, unspliced {(run.unspliced_fraction * 100).toFixed(1)}%.{" "}
            {run.neighbours_note} {run.pca_note ? `${run.pca_note}.` : ""}
          </Caption>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead className="text-slate-500">
                <tr>
                  <th className="pr-2 font-medium">{run.group_column ?? "Group"}</th>
                  <th className="pr-2 font-medium" title="Cells in the group">n</th>
                  <th className="pr-2 font-medium" title="Mean velocity confidence: how far each cell's velocity agrees with its neighbours'">Coherence</th>
                  <th className="pr-2 font-medium" title="Mean length of the velocity vector">Length</th>
                  <th className="pr-2 font-medium" title="Unspliced share of the group's raw counts">Unspliced</th>
                  <th className="pr-2 font-medium" title="Velocity genes with unspliced counts in at least 10% of the group's cells">Genes</th>
                  <th className="font-medium" title="Velocity genes with the highest mean velocity in the group">Top genes</th>
                </tr>
              </thead>
              <tbody>
                {run.clusters.map((c) => (
                  <tr key={c.group} className="border-t border-slate-100">
                    <td className="pr-2">{c.group}</td>
                    <td className="pr-2">{c.n_cells.toLocaleString()}</td>
                    <td className="pr-2">{c.coherence.toFixed(2)}</td>
                    <td className="pr-2">{c.length.toFixed(1)}</td>
                    <td className="pr-2">{(c.unspliced_fraction * 100).toFixed(1)}%</td>
                    <td className="pr-2">{c.n_velocity_genes.toLocaleString()}</td>
                    <td>
                      {c.top_genes.slice(0, 3).map((g) => (
                        <button key={g} type="button" onClick={() => setGene(g)} className="mr-1 text-blue-600 hover:underline">{g}</button>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            <div className="mb-1 flex items-center gap-2">
              <span className="font-medium">Phase portrait</span>
              <InfoTip label="What a phase portrait shows">
                Each dot is a cell's spliced and unspliced first moments for this gene, the mean over the cell and its
                neighbours. The red line is the fitted steady-state ratio gamma. Cells above it are read as inducing the gene, cells
                below as repressing it. A gene with no clear almond shape gives little to fit.
              </InfoTip>
              <select
                aria-label="Gene"
                value={gene ?? ""}
                onChange={(e) => setGene(e.target.value)}
                className="max-w-[10rem] rounded border border-slate-300 bg-white px-1 py-0.5"
              >
                {shownGenes.map((g) => (
                  <option key={g.gene} value={g.gene}>
                    {g.gene} (r2 {g.r2.toFixed(2)}{g.velocity_gene ? "" : ", not a velocity gene"})
                  </option>
                ))}
                {gene && !shownGenes.some((g) => g.gene === gene) && <option value={gene}>{gene}</option>}
              </select>
            </div>
            {genesCap && <p className="mb-1 text-slate-500">{genesCap}</p>}
            {phaseError && <p className="text-red-600">{phaseError}</p>}
            {phase && (
              <>
                <Plot
                  data={traces}
                  layout={layout}
                  config={plotConfig(`velocity_phase_${phase.gene}`)}
                  style={{ width: "100%", height: 300 }}
                  useResizeHandler
                />
                <Caption n={{ value: phase.n_points, unit: "cells", of: phase.n_cells }}>
                  gamma {phase.gamma.toPrecision(3)}, r2 {phase.r2.toFixed(2)}
                  {phase.velocity_gene ? "" : "; not a velocity gene, so not in the arrows"}
                  {phase.capped ? `; a sample of ${phase.max_points.toLocaleString()} cells` : ""}.
                  {phase.color ? ` Coloured by ${phase.color.column}.` : ""}
                </Caption>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
