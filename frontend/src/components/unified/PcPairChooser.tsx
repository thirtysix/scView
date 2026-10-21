import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { isPcaEmbedding } from "@/hooks/useEmbedding";

/**
 * "PCs: x [ ] y [ ]" next to the embedding selector, only while a PCA with more
 * than two columns is shown: which two components the scatter plots (the first
 * two by default). The pair is kept when switching embeddings and back.
 */
export function PcPairChooser() {
  const embedding = useSettingsStore((s) => s.embedding);
  const pcaDims = useSettingsStore((s) => s.pcaDims);
  const setPcaDims = useSettingsStore((s) => s.setPcaDims);
  const nDims = useDatasetStore((s) => s.currentDataset?.embedding_dimensions?.[embedding]);
  if (!isPcaEmbedding(embedding) || nDims === undefined || nDims <= 2) return null;
  // A pair from a bigger dataset: shown as the default the plot falls back to.
  const shown: [number, number] = Math.max(...pcaDims) < nDims ? pcaDims : [0, 1];
  const options = Array.from({ length: nDims }, (_, i) => i);
  const select = (axis: 0 | 1) => (
    <select
      aria-label={axis === 0 ? "PC on the x axis" : "PC on the y axis"}
      value={shown[axis]}
      onChange={(e) => {
        const next: [number, number] = [shown[0], shown[1]];
        next[axis] = Number(e.target.value);
        if (next[0] === next[1]) next[axis === 0 ? 1 : 0] = shown[axis]; // swap, never the same PC twice
        setPcaDims(next);
      }}
      className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700 focus:border-blue-400 focus:outline-none"
    >
      {options.map((i) => (
        <option key={i} value={i}>
          PC{i + 1}
        </option>
      ))}
    </select>
  );
  return (
    <span className="flex items-center gap-1" title="Which two principal components the scatter shows">
      <label className="text-[11px] font-medium text-slate-500">x</label>
      {select(0)}
      <label className="text-[11px] font-medium text-slate-500">y</label>
      {select(1)}
    </span>
  );
}
