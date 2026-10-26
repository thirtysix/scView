import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Loader2, Shield } from "lucide-react";
import type { ProteinInfo } from "@/api/types";
import { apiFetchBinary } from "@/api/client";
import { decodeArrowBuffer } from "@/lib/arrow";
import { plotConfig } from "@/lib/plotConfig";
import { spearman, targetGenes } from "@/lib/protein";
import { normLabel, proteinLayerFor, type ProteinNorm } from "@/lib/dsb";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { GeneSearch } from "@/components/common/GeneSearch";
import { ColorLegend } from "@/components/plots/ColorLegend";
import type { ResolvedColorScale } from "@/lib/colorScale";
import { MarkerPlot } from "@/components/plots/MarkerPlot";
import Plot from "@/components/plots/Plot";
import { ProteinNormalisation } from "./ProteinNormalisation";

interface UnifiedProteinSubtabProps {
  datasetId: string;
  protein: ProteinInfo;
  onProteinSelect: (protein: string) => void;
  activeProtein: string | null;
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
  /** The active protein's resolved colour scale (diverging when its values have both signs) */
  proteinScale: ResolvedColorScale | null;
  /** CLR or DSB (DSB only once computed) */
  proteinNorm?: ProteinNorm;
  onProteinNormChange?: (norm: ProteinNorm) => void;
}

/** Points drawn in the protein-vs-RNA scatter (the correlation uses every cell). */
const MAX_POINTS = 20000;
const AXIS = { tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9", zeroline: false };

const toFloat = (v: Float32Array | Int32Array | Uint8Array | undefined) =>
  v instanceof Float32Array ? v : v ? Float32Array.from(v) : null;

/**
 * CITE-seq: the antibody panel as a protein modality. Click a protein to colour
 * the embedding (and its violin by group); the heatmap shows every protein by
 * group; the protein-vs-RNA view sets a protein against the gene that encodes
 * its target, where RNA dropout shows as cells with protein but no transcript.
 */
export function UnifiedProteinSubtab({
  datasetId,
  protein,
  onProteinSelect,
  activeProtein,
  groupByColumn,
  setGroupByColumn,
  proteinScale,
  proteinNorm = "clr",
  onProteinNormChange = () => {},
}: UnifiedProteinSubtabProps) {
  const layer = proteinLayerFor(protein, proteinNorm) ?? protein.layer;
  const dataset = useDatasetStore((s) => s.currentDataset);
  const expressionLayer = useSettingsStore((s) => s.expressionLayer);
  const targets = useMemo(
    () => protein.names.filter((n) => !protein.controls.includes(n)),
    [protein],
  );

  const categoricalColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter((c) => {
      if (c.dtype !== "category" && c.dtype !== "object" && c.dtype !== "bool") return false;
      if (c.n_unique > 100) return false;
      return !(nCells > 0 && c.n_unique / nCells >= 0.9);
    });
  }, [dataset]);

  const chip = (name: string, control: boolean) => {
    const on = name === activeProtein;
    return (
      <button
        key={name}
        onClick={() => onProteinSelect(name)}
        aria-pressed={on}
        title={control ? `${name}: isotype control (background binding)` : `Colour the cells by ${name} protein`}
        className={`rounded-full border px-2 py-0.5 text-xs transition-colors ${
          on
            ? "border-primary bg-primary text-white"
            : control
              ? "border-dashed border-slate-300 text-slate-400 hover:border-slate-400"
              : "border-slate-300 text-slate-700 hover:border-primary hover:text-primary"
        }`}
      >
        {name}
      </button>
    );
  };

  return (
    <div className="space-y-3">
      <p className="text-[11px] leading-snug text-slate-500">
        {targets.length} antibod{targets.length === 1 ? "y" : "ies"} measured with the RNA (CITE-seq).
        Click one to colour the cells.
      </p>
      <ProteinNormalisation
        datasetId={datasetId}
        protein={protein}
        norm={proteinNorm}
        onNormChange={onProteinNormChange}
      />
      <div className="flex flex-wrap gap-1.5" aria-label="Proteins">
        {targets.map((n) => chip(n, false))}
      </div>
      {protein.controls.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="flex items-center gap-1 text-[11px] text-slate-400">
            <Shield className="h-3 w-3" /> Isotype controls:
          </span>
          {protein.controls.map((n) => chip(n, true))}
        </div>
      )}

      {activeProtein && proteinScale && (
        <div className="rounded-lg border border-blue-100 bg-blue-50/50 p-2.5">
          <ColorLegend
            type="continuous"
            scale={proteinScale}
            label={`${activeProtein} protein (${normLabel(layer)})`}
          />
        </div>
      )}

      <div>
        <label className="mb-1 block text-[11px] font-medium text-slate-500">Group by (violin, heatmap)</label>
        <select
          aria-label="Group by"
          value={groupByColumn}
          onChange={(e) => setGroupByColumn(e.target.value)}
          className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
        >
          {categoricalColumns.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name} ({col.n_unique} groups)
            </option>
          ))}
        </select>
      </div>

      {groupByColumn && targets.length > 0 && (
        <div>
          <h4 className="mb-1 text-xs font-semibold text-slate-600">Proteins by group</h4>
          <MarkerPlot
            datasetId={datasetId}
            groupby={groupByColumn}
            groups={[]}
            markers={{}}
            kind="heatmap"
            perGroup={0}
            genes={targets}
            layer={layer}
          />
        </div>
      )}

      <ProteinVsRna
        datasetId={datasetId}
        proteins={targets}
        proteinLayer={layer}
        rnaLayer={expressionLayer}
        initial={activeProtein && targets.includes(activeProtein) ? activeProtein : targets[0] ?? ""}
      />
    </div>
  );
}

interface Pair {
  protein: string;
  gene: string;
  x: Float32Array; // RNA
  y: Float32Array; // protein (CLR or DSB)
}

function ProteinVsRna({
  datasetId,
  proteins,
  proteinLayer,
  rnaLayer,
  initial,
}: {
  datasetId: string;
  proteins: string[];
  proteinLayer: string;
  rnaLayer: string;
  initial: string;
}) {
  const [proteinName, setProteinName] = useState(initial);
  const [chosenGene, setChosenGene] = useState<string | null>(null); // null: the target's gene
  const [pair, setPair] = useState<Pair | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Follow the protein clicked above until one is picked here.
  useEffect(() => setProteinName(initial), [initial]);
  useEffect(() => setChosenGene(null), [proteinName]);

  useEffect(() => {
    if (!proteinName) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    const candidates = chosenGene ? [chosenGene] : targetGenes(proteinName);
    const layerQ = rnaLayer ? `&layer=${encodeURIComponent(rnaLayer)}` : "";
    Promise.all([
      apiFetchBinary(`/datasets/${datasetId}/expression?genes=${encodeURIComponent(proteinName)}&layer=${encodeURIComponent(proteinLayer)}`),
      apiFetchBinary(`/datasets/${datasetId}/expression?genes=${candidates.map(encodeURIComponent).join(",")}${layerQ}`),
    ])
      .then(([pBuf, gBuf]) => {
        if (cancelled) return;
        const p = decodeArrowBuffer(pBuf);
        const g = decodeArrowBuffer(gBuf);
        const gene = candidates.find((c) => c in g);
        const y = toFloat(p[proteinName] ?? Object.values(p)[0]);
        const x = gene ? toFloat(g[gene]) : null;
        if (!x || !y || !gene) throw new Error("not found");
        setPair({ protein: proteinName, gene, x, y });
      })
      .catch(() => {
        if (cancelled) return;
        setPair(null);
        setError(
          chosenGene
            ? `${chosenGene} is not in the RNA layer.`
            : `No gene for ${proteinName} found in the RNA (tried ${candidates.join(", ")}). Pick one below.`,
        );
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, proteinName, chosenGene, proteinLayer, rnaLayer]);

  const stats = useMemo(() => {
    if (!pair) return null;
    const n = pair.x.length;
    let noRna = 0;
    for (let i = 0; i < n; i++) if (pair.x[i]! === 0) noRna++;
    const step = Math.max(1, Math.ceil(n / MAX_POINTS));
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < n; i += step) {
      xs.push(pair.x[i]!);
      ys.push(pair.y[i]!);
    }
    return { rho: spearman(pair.x, pair.y), noRna: noRna / Math.max(n, 1), xs, ys, n, drawn: xs.length };
  }, [pair]);

  if (proteins.length === 0) return null;
  return (
    <div className="space-y-2 rounded-lg border border-slate-200 p-2">
      <h4 className="text-xs font-semibold text-slate-600">Protein vs RNA</h4>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <select
          aria-label="Protein to compare"
          value={proteinName}
          onChange={(e) => setProteinName(e.target.value)}
          className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700"
        >
          {proteins.map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
        <span className="text-slate-400">vs gene</span>
        <span className="font-medium text-slate-700">{pair?.gene ?? chosenGene ?? "…"}</span>
      </div>
      <GeneSearch datasetId={datasetId} onSelect={setChosenGene} placeholder="Other gene…" className="w-full" />

      {loading && (
        <div className="flex items-center gap-2 py-6 text-xs text-slate-400">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading…
        </div>
      )}
      {!loading && error && (
        <p className="flex items-center gap-1.5 text-xs text-amber-700">
          <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" /> {error}
        </p>
      )}
      {!loading && pair && stats && (
        <>
          <Plot
            data={[
              {
                type: "scattergl" as const,
                mode: "markers" as const,
                x: stats.xs,
                y: stats.ys,
                marker: { size: 3, color: "#2563eb", opacity: 0.35 },
                hovertemplate: `${pair.gene} %{x:.2f}<br>${pair.protein} %{y:.2f}<extra></extra>`,
              },
            ]}
            layout={{
              height: 240,
              margin: { t: 8, r: 10, b: 40, l: 46 },
              xaxis: { ...AXIS, title: { text: `${pair.gene} RNA`, font: { size: 11, color: "#64748b" } } },
              yaxis: { ...AXIS, title: { text: `${pair.protein} protein (${normLabel(proteinLayer)})`, font: { size: 11, color: "#64748b" } } },
              showlegend: false,
              paper_bgcolor: "white",
              plot_bgcolor: "white",
              font: { family: "Inter, system-ui, sans-serif" },
            }}
            config={plotConfig(`protein_vs_rna_${pair.protein}_${pair.gene}`)}
            useResizeHandler
            style={{ width: "100%" }}
          />
          <p className="text-[11px] leading-snug text-slate-500" data-testid="protein-rna-stats">
            Spearman ρ {stats.rho == null ? "n/a" : stats.rho.toFixed(2)} over {stats.n.toLocaleString()} cells
            {stats.drawn < stats.n ? ` (${stats.drawn.toLocaleString()} drawn)` : ""}. {pair.gene} RNA is not
            detected in {(stats.noRna * 100).toFixed(0)}% of cells: transcripts drop out where the protein
            is still measured, which is what the antibodies add.
          </p>
        </>
      )}
    </div>
  );
}
