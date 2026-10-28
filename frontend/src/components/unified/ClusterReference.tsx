import { useCallback, useEffect, useMemo, useState } from "react";

import { EmbeddingScatter } from "@/components/plots/EmbeddingScatter";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { decodeArrowBuffer } from "@/lib/arrow";
import { mapCategoryToColor } from "@/lib/colors";
import { useCategorySlots } from "@/hooks/useCategoryColors";
import { useEmbeddingReady } from "@/hooks/useEmbeddingReady";
import { useObsRevision } from "@/stores/obsRevisionStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useScatterBackground } from "@/lib/theme";
import { ClusterReportButton } from "@/components/report/ReportButtons";

/**
 * A compact reference scatter showing the cells coloured by the active grouping
 * column (e.g. Leiden clusters) on the same embedding as the main plot — so the
 * cluster layout stays visible while the main plot is recoloured by a gene /
 * gene-set score. Clicking a legend entry highlights that cluster (dimming the
 * rest) so you can see exactly which cells a colour change refers to. The
 * highlight is the shared one (selection store): a cluster clicked here also
 * opens in the Markers subtab, and one clicked in the main legend shows here.
 * Without a highlight it shows the selection (a lasso on the main plot).
 *
 * `positions` null: the map draws its own embedding's coordinates (from the
 * same fetch as the cluster codes). The tissue plot uses this, so the map
 * keeps showing the UMAP while the main plot shows the section.
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
  /** Linked camera from the main plot (so the two rotate/pan together); leave
   *  it out when the main plot shows another space (the tissue). */
  viewState?: Record<string, unknown> | null;
  onViewStateChange?: (vs: Record<string, unknown>) => void;
}) {
  const paletteName = useSettingsStore((s) => s.categoricalPalette);
  const background = useScatterBackground();
  const slots = useCategorySlots(column);
  const [clusterValues, setClusterValues] = useState<Int32Array | null>(null);
  const [own, setOwn] = useState<{ positions: Float32Array; dimensions: 2 | 3 } | null>(null);
  const lassoed = useSelectionStore((s) => s.selectedCellIndices);
  const needOwn = positions == null;
  const revision = useObsRevision(datasetId, column); // refetch after hand labels
  const ready = useEmbeddingReady(datasetId, embedding);
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
    if (!ready || !column) {
      setClusterValues(null);
      setOwn(null);
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
        setOwn(needOwn ? ownPositions(decoded) : null);
      })
      .catch(() => {
        if (!cancelled) {
          setClusterValues(null);
          setOwn(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [ready, datasetId, embedding, column, revision, needOwn]);

  const selectedIndices = useMemo(() => {
    if (highlight == null) return lassoed && lassoed.size > 0 ? lassoed : null;
    if (!clusterValues) return null;
    const s = new Set<number>();
    for (let i = 0; i < clusterValues.length; i++) {
      if (clusterValues[i] === highlight) s.add(i);
    }
    return s;
  }, [highlight, clusterValues, lassoed]);

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

  const shownPositions = positions ?? own?.positions ?? null;
  const shownDimensions = positions ? dimensions : (own?.dimensions ?? 2);
  if (!shownPositions || !clusterValues) return null;

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <div className="border-b border-slate-100 px-2.5 py-1.5 text-[11px] font-semibold text-slate-600">
        Cluster map · {column}
        {highlight != null && categories[highlight] && (
          <span className="ml-1 font-normal text-slate-400">
            (highlighting {categories[highlight]})
            <ClusterReportButton column={column} value={categories[highlight]} />
          </span>
        )}
      </div>
      <div className="h-40">
        <EmbeddingScatter
          positions={shownPositions}
          colorValues={clusterValues}
          colorType="categorical"
          categorySlots={slots}
          dimensions={shownDimensions}
          pointSize={1.5}
          selectedIndices={selectedIndices}
          externalViewState={viewState}
          onViewStateChange={onViewStateChange}
          onClick={handlePointClick}
          dimToGray
          minimal
          background={background}
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

/** The embedding's own x, y (and z) from a decoded embedding fetch, interleaved. */
function ownPositions(
  decoded: Record<string, Float32Array | Int32Array | Uint8Array>,
): { positions: Float32Array; dimensions: 2 | 3 } | null {
  const { x, y, z } = decoded;
  if (!x || !y) return null;
  const stride = z ? 3 : 2;
  const out = new Float32Array(x.length * stride);
  for (let i = 0; i < x.length; i++) {
    out[i * stride] = x[i]!;
    out[i * stride + 1] = y[i]!;
    if (z) out[i * stride + 2] = z[i]!;
  }
  return { positions: out, dimensions: stride };
}
