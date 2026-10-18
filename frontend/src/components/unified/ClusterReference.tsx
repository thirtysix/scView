import { useCallback, useEffect, useMemo, useState } from "react";

import { EmbeddingScatter } from "@/components/plots/EmbeddingScatter";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { decodeArrowBuffer } from "@/lib/arrow";
import { mapCategoryToColor } from "@/lib/colors";
import { useCategorySlots } from "@/hooks/useCategoryColors";
import { useObsRevision } from "@/stores/obsRevisionStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * A compact reference scatter showing the cells coloured by the active grouping
 * column (e.g. Leiden clusters) on the same embedding as the main plot — so the
 * cluster layout stays visible while the main plot is recoloured by a gene /
 * gene-set score. Clicking a legend entry highlights that cluster (dimming the
 * rest) so you can see exactly which cells a colour change refers to. The
 * highlight is the shared one (selection store): a cluster clicked here also
 * opens in the Markers subtab, and one clicked in the main legend shows here.
 */
export function ClusterReference({
  datasetId,
  embedding,
  column,
  positions,
  dimensions,
  categories,
  viewState,
  onViewStateChange,
}: {
  datasetId: string;
  embedding: string;
  column: string;
  positions: Float32Array | null;
  dimensions: 2 | 3;
  categories: string[];
  /** Linked camera from the main plot (so the two rotate/pan together). */
  viewState?: Record<string, unknown> | null;
  onViewStateChange?: (vs: Record<string, unknown>) => void;
}) {
  const paletteName = useSettingsStore((s) => s.categoricalPalette);
  const slots = useCategorySlots(column);
  const [clusterValues, setClusterValues] = useState<Int32Array | null>(null);
  const revision = useObsRevision(datasetId, column); // refetch after hand labels
  const highlightedGroup = useSelectionStore((s) => s.highlightedGroup);
  const setHighlightedGroup = useSelectionStore((s) => s.setHighlight);
  // Index of the highlighted group in this column's categories (null when the
  // highlight belongs to another column).
  const highlight = useMemo(() => {
    if (!highlightedGroup || highlightedGroup.column !== column) return null;
    const idx = categories.indexOf(highlightedGroup.value);
    return idx >= 0 ? idx : null;
  }, [highlightedGroup, column, categories]);
  const setHighlight = useCallback(
    (idx: number | null) => {
      const value = idx == null ? undefined : categories[idx];
      setHighlightedGroup(value == null ? null : { column, value });
    },
    [categories, column, setHighlightedGroup],
  );

  useEffect(() => {
    if (!datasetId || !column || !embedding) {
      setClusterValues(null);
      return;
    }
    let cancelled = false;
    fetchEmbeddingBinary(datasetId, embedding, column)
      .then((buf) => {
        if (cancelled) return;
        const decoded = decodeArrowBuffer(buf);
        const v = decoded[column];
        // Force Int32Array so EmbeddingScatter treats it as categorical.
        if (v) setClusterValues(new Int32Array(v as ArrayLike<number>));
      })
      .catch(() => {
        if (!cancelled) setClusterValues(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, embedding, column, revision]);

  const selectedIndices = useMemo(() => {
    if (highlight == null || !clusterValues) return null;
    const s = new Set<number>();
    for (let i = 0; i < clusterValues.length; i++) {
      if (clusterValues[i] === highlight) s.add(i);
    }
    return s;
  }, [highlight, clusterValues]);

  // Clicking a point selects that cell's cluster — same as clicking its legend
  // chip (toggle off if already highlighted; click empty space to clear).
  const handlePointClick = useCallback(
    (index: number | null) => {
      if (index == null) {
        setHighlight(null);
        return;
      }
      const code = clusterValues?.[index];
      if (code == null || code < 0) return;
      setHighlight(highlight === code ? null : code);
    },
    [clusterValues, highlight, setHighlight],
  );

  if (!positions || !clusterValues) return null;

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <div className="border-b border-slate-100 px-2.5 py-1.5 text-[11px] font-semibold text-slate-600">
        Cluster map · {column}
        {highlight != null && categories[highlight] && (
          <span className="ml-1 font-normal text-slate-400">
            (highlighting {categories[highlight]})
          </span>
        )}
      </div>
      <div className="h-40">
        <EmbeddingScatter
          positions={positions}
          colorValues={clusterValues}
          colorType="categorical"
          categorySlots={slots}
          dimensions={dimensions}
          pointSize={1.5}
          selectedIndices={selectedIndices}
          externalViewState={viewState}
          onViewStateChange={onViewStateChange}
          onClick={handlePointClick}
          dimToGray
          minimal
        />
      </div>
      <div className="flex flex-wrap gap-1 p-2">
        {categories.map((name, idx) => {
          const [r, g, b] = mapCategoryToColor(idx, paletteName, slots);
          const active = highlight === idx;
          return (
            <button
              key={name}
              onClick={() => setHighlight(active ? null : idx)}
              title={active ? "Clear highlight" : `Highlight ${name}`}
              className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition-colors ${
                active ? "bg-slate-200 font-medium text-slate-800" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-sm"
                style={{ background: `rgb(${r},${g},${b})` }}
              />
              {name}
            </button>
          );
        })}
      </div>
    </div>
  );
}
