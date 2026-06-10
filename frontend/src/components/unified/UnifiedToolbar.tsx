import { useSettingsStore } from "@/stores/settingsStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { Lasso, RotateCcw, X } from "lucide-react";
import { formatNumber, prettyObsLabel } from "@/lib/formatting";

interface UnifiedToolbarProps {
  onResetView?: () => void;
  numCells: number;
}

export function UnifiedToolbar({ onResetView, numCells }: UnifiedToolbarProps) {
  const dataset = useDatasetStore((s) => s.currentDataset);

  const embedding = useSettingsStore((s) => s.embedding);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const expressionLayer = useSettingsStore((s) => s.expressionLayer);
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);
  const setColorBy = useSettingsStore((s) => s.setColorBy);
  const setExpressionLayer = useSettingsStore((s) => s.setExpressionLayer);
  const setPointSize = useSettingsStore((s) => s.setPointSize);
  const setOpacity = useSettingsStore((s) => s.setOpacity);
  const plotBackground = useSettingsStore((s) => s.plotBackground);
  const setPlotBackground = useSettingsStore((s) => s.setPlotBackground);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const setMaxRenderedCells = useSettingsStore((s) => s.setMaxRenderedCells);

  const selectionMode = useSelectionStore((s) => s.selectionMode);
  const setSelectionMode = useSelectionStore((s) => s.setSelectionMode);
  const selectedIndices = useSelectionStore((s) => s.selectedCellIndices);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const availableEmbeddings = dataset?.available_embeddings ?? [];
  const nCells = dataset?.n_cells ?? 0;

  const sortedObsColumns = (dataset?.obs_columns ?? [])
    .filter((c) => {
      const isCat = c.dtype === "category" || c.dtype === "object" || c.dtype === "bool";
      if (isCat && c.n_unique > 100) return false;
      if (isCat && nCells > 0 && c.n_unique / nCells >= 0.9) return false;
      return true;
    })
    .sort((a, b) => {
      const aIsCat = a.dtype === "category" || a.dtype === "object" || a.dtype === "bool";
      const bIsCat = b.dtype === "category" || b.dtype === "object" || b.dtype === "bool";
      if (aIsCat && !bIsCat) return -1;
      if (!aIsCat && bIsCat) return 1;
      const aScview = a.name.startsWith("scview_");
      const bScview = b.name.startsWith("scview_");
      if (aScview && !bScview) return -1;
      if (!aScview && bScview) return 1;
      return a.name.localeCompare(b.name);
    });

  return (
    <div className="flex flex-shrink-0 flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 shadow-sm">
      {/* Embedding selector */}
      <div className="flex items-center gap-1.5">
        <label className="text-[11px] font-medium text-slate-500">Embedding</label>
        <select
          value={embedding}
          onChange={(e) => setEmbedding(e.target.value)}
          className="rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
        >
          {availableEmbeddings.length === 0 && <option value="">None</option>}
          {availableEmbeddings.map((name) => {
            const dims = dataset?.embedding_dimensions?.[name];
            return (
              <option key={name} value={name}>
                {name.replace("X_", "").toUpperCase()}{dims === 3 ? " (3D)" : ""}
              </option>
            );
          })}
        </select>
      </div>

      <div className="h-4 w-px bg-slate-200" />

      {/* Color by */}
      <div className="flex items-center gap-1.5">
        <label className="text-[11px] font-medium text-slate-500">Color</label>
        <select
          value={colorBy}
          onChange={(e) => setColorBy(e.target.value)}
          className="max-w-[160px] rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
        >
          <option value="">None</option>
          {sortedObsColumns.map((col) => (
            <option key={col.name} value={col.name}>
              {prettyObsLabel(col.name)}
              {col.dtype === "category" || col.dtype === "object"
                ? ` (${col.n_unique})`
                : ""}
            </option>
          ))}
        </select>
      </div>

      {/* Expression layer (units for overlay + violin) */}
      {(dataset?.expression_layers?.length ?? 0) > 0 && (
        <>
          <div className="h-4 w-px bg-slate-200" />
          <div className="flex items-center gap-1.5">
            <label className="text-[11px] font-medium text-slate-500">Layer</label>
            <select
              value={expressionLayer}
              onChange={(e) => setExpressionLayer(e.target.value)}
              title="Expression units used for the gene overlay and violin plots"
              className="max-w-[160px] rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
            >
              {dataset?.expression_layers?.map((l) => (
                <option key={l.key} value={l.key}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>
        </>
      )}

      <div className="h-4 w-px bg-slate-200" />

      {/* Point size */}
      <div className="flex items-center gap-1.5">
        <label className="text-[11px] font-medium text-slate-500">Size</label>
        <input
          type="range"
          min={1}
          max={10}
          step={0.5}
          value={pointSize}
          onChange={(e) => setPointSize(parseFloat(e.target.value))}
          className="w-16 accent-blue-500"
        />
        <span className="w-4 text-[10px] tabular-nums text-slate-400">{pointSize}</span>
      </div>

      {/* Opacity */}
      <div className="flex items-center gap-1.5">
        <label className="text-[11px] font-medium text-slate-500">Opacity</label>
        <input
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={opacity}
          onChange={(e) => setOpacity(parseFloat(e.target.value))}
          className="w-16 accent-blue-500"
        />
      </div>

      {/* Background light/dark */}
      <div className="flex items-center gap-1.5">
        <label className="text-[11px] font-medium text-slate-500">BG</label>
        <div className="flex overflow-hidden rounded border border-slate-300">
          {(["white", "dark"] as const).map((bg) => (
            <button
              key={bg}
              onClick={() => setPlotBackground(bg)}
              className={`px-1.5 py-0.5 text-[10px] capitalize transition-colors ${
                plotBackground === bg
                  ? "bg-blue-500 text-white"
                  : "bg-white text-slate-600 hover:bg-slate-100"
              }`}
            >
              {bg === "white" ? "Light" : "Dark"}
            </button>
          ))}
        </div>
      </div>

      {/* Max rendered cells */}
      <div className="flex items-center gap-1.5">
        <label className="text-[11px] font-medium text-slate-500" title="Larger datasets are downsampled for rendering">
          Max
        </label>
        <select
          value={maxRenderedCells}
          onChange={(e) => setMaxRenderedCells(parseInt(e.target.value))}
          className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[10px] text-slate-700 focus:border-blue-400 focus:outline-none"
        >
          {[25000, 50000, 100000, 250000, 500000].map((n) => (
            <option key={n} value={n}>
              {n / 1000}K
            </option>
          ))}
        </select>
      </div>

      <div className="h-4 w-px bg-slate-200" />

      {/* Lasso */}
      <button
        onClick={() => setSelectionMode(selectionMode === "lasso" ? "none" : "lasso")}
        title="Lasso selection (L)"
        className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
          selectionMode === "lasso"
            ? "bg-orange-50 text-orange-700 ring-1 ring-orange-300"
            : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        <Lasso className="h-3.5 w-3.5" />
        Lasso
      </button>

      {/* Selection info */}
      {selectedIndices && selectedIndices.size > 0 && (
        <span className="flex items-center gap-1 text-xs text-orange-600">
          {formatNumber(selectedIndices.size)} selected
          <button onClick={clearSelection} className="rounded p-0.5 hover:bg-orange-100">
            <X className="h-3 w-3" />
          </button>
        </span>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Cell count */}
      {numCells > 0 && (
        <span className="text-[11px] text-slate-400">
          {formatNumber(numCells)} cells
        </span>
      )}

      {/* Reset view */}
      {onResetView && (
        <button
          onClick={onResetView}
          title="Reset view"
          className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}
