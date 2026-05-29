import { useSettingsStore } from "@/stores/settingsStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { RotateCcw, Lasso } from "lucide-react";

interface PlotControlsProps {
  onResetView?: () => void;
}

export function PlotControls({ onResetView }: PlotControlsProps) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const embedding = useSettingsStore((s) => s.embedding);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const plotBackground = useSettingsStore((s) => s.plotBackground);
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);
  const setColorBy = useSettingsStore((s) => s.setColorBy);
  const setPointSize = useSettingsStore((s) => s.setPointSize);
  const setOpacity = useSettingsStore((s) => s.setOpacity);
  const setPlotBackground = useSettingsStore((s) => s.setPlotBackground);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const setMaxRenderedCells = useSettingsStore((s) => s.setMaxRenderedCells);
  const selectionMode = useSelectionStore((s) => s.selectionMode);
  const setSelectionMode = useSelectionStore((s) => s.setSelectionMode);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const availableEmbeddings = dataset?.available_embeddings ?? [];
  const nCells = dataset?.n_cells ?? 0;

  // Sort obs columns: categorical first, then numeric; filter non-informative
  const sortedObsColumns = (dataset?.obs_columns ?? [])
    .filter((c) => {
      // Exclude columns where nearly every cell has a unique value (cell_id, barcodes)
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
      // Within categoricals, sort scview_ columns first (pipeline-generated)
      const aScview = a.name.startsWith("scview_");
      const bScview = b.name.startsWith("scview_");
      if (aScview && !bScview) return -1;
      if (!aScview && bScview) return 1;
      return a.name.localeCompare(b.name);
    });

  return (
    <div className="flex flex-col gap-4">
      {/* Embedding selector */}
      <div>
        <label className="mb-1 block text-xs font-medium text-slate-600">
          Embedding
        </label>
        <select
          value={embedding}
          onChange={(e) => setEmbedding(e.target.value)}
          className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-800 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
        >
          {availableEmbeddings.length === 0 && (
            <option value="">No embeddings available</option>
          )}
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

      {/* Color by selector */}
      <div>
        <label className="mb-1 block text-xs font-medium text-slate-600">
          Color by
        </label>
        <select
          value={colorBy}
          onChange={(e) => setColorBy(e.target.value)}
          className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-800 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
        >
          <option value="">None</option>
          {sortedObsColumns.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name}
              {col.dtype === "category" || col.dtype === "object"
                ? ` (${col.n_unique} categories)`
                : ` (${col.dtype})`}
            </option>
          ))}
        </select>
      </div>

      {/* Point size slider */}
      <div>
        <label className="mb-1 flex items-center justify-between text-xs font-medium text-slate-600">
          <span>Point size</span>
          <span className="tabular-nums text-slate-400">{pointSize}</span>
        </label>
        <input
          type="range"
          min={1}
          max={10}
          step={0.5}
          value={pointSize}
          onChange={(e) => setPointSize(parseFloat(e.target.value))}
          className="w-full accent-blue-500"
        />
      </div>

      {/* Opacity slider */}
      <div>
        <label className="mb-1 flex items-center justify-between text-xs font-medium text-slate-600">
          <span>Opacity</span>
          <span className="tabular-nums text-slate-400">
            {opacity.toFixed(2)}
          </span>
        </label>
        <input
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={opacity}
          onChange={(e) => setOpacity(parseFloat(e.target.value))}
          className="w-full accent-blue-500"
        />
      </div>

      {/* Background toggle */}
      <div>
        <label className="mb-1 block text-xs font-medium text-slate-600">
          Background
        </label>
        <div className="flex gap-1">
          <button
            onClick={() => setPlotBackground("white")}
            className={`flex-1 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
              plotBackground === "white"
                ? "border-blue-400 bg-blue-50 text-blue-700"
                : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
            }`}
          >
            Light
          </button>
          <button
            onClick={() => setPlotBackground("dark")}
            className={`flex-1 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
              plotBackground === "dark"
                ? "border-blue-400 bg-blue-50 text-blue-700"
                : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
            }`}
          >
            Dark
          </button>
        </div>
      </div>

      {/* Max rendered cells slider */}
      <div>
        <label className="mb-1 flex items-center justify-between text-xs font-medium text-slate-600">
          <span>Max rendered cells</span>
          <span className="tabular-nums text-slate-400">
            {maxRenderedCells >= 1000
              ? `${(maxRenderedCells / 1000).toFixed(0)}K`
              : maxRenderedCells}
          </span>
        </label>
        <input
          type="range"
          min={10000}
          max={500000}
          step={10000}
          value={maxRenderedCells}
          onChange={(e) => setMaxRenderedCells(parseInt(e.target.value))}
          className="w-full accent-blue-500"
        />
        <p className="mt-0.5 text-[10px] text-slate-400">
          Larger datasets are randomly sampled for rendering
        </p>
      </div>

      {/* Divider */}
      <hr className="border-slate-200" />

      {/* Selection tools */}
      <div>
        <label className="mb-1.5 block text-xs font-medium text-slate-600">
          Selection
        </label>
        <div className="flex gap-1.5">
          <button
            onClick={() =>
              setSelectionMode(selectionMode === "lasso" ? "none" : "lasso")
            }
            title="Lasso selection (L)"
            className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
              selectionMode === "lasso"
                ? "border-orange-400 bg-orange-50 text-orange-700"
                : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
            }`}
          >
            <Lasso className="h-3.5 w-3.5" />
            Lasso
          </button>
          <button
            onClick={clearSelection}
            title="Clear selection"
            className="flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Reset view */}
      {onResetView && (
        <button
          onClick={onResetView}
          className="flex items-center justify-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Reset view
        </button>
      )}
    </div>
  );
}
