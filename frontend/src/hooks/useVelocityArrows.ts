import { useEffect, useMemo, useState } from "react";
import { LineLayer } from "@deck.gl/layers";
import type { Layer } from "@deck.gl/core";
import { getVelocityGrid, type VelocityGrid } from "@/api/velocity";
import { useEmbeddingReady } from "@/hooks/useEmbeddingReady";
import { arrowColor, arrowSegments, arrowsBlocked, type ArrowSegment } from "@/lib/velocityArrows";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const NO_LAYERS: readonly Layer[] = [];

/**
 * The velocity arrows of the main scatter as a deck.gl layer. Nothing is fetched or
 * built while the toggle is off (``velocityRun`` is set only for a dataset with a run on
 * its current cells), so a dataset with no velocity costs nothing. The grid is fetched per
 * dataset, embedding, density and run; the scale only rescales the segments here.
 */
export function useVelocityArrows(
  datasetId: string | null | undefined,
  embedding: string,
  dimensions: number,
  pcaPair: boolean,
  background: "white" | "dark",
): readonly Layer[] {
  const arrows = useUnifiedViewStore((s) => s.velocityArrows);
  const run = useUnifiedViewStore((s) => s.velocityRun);
  const runId = run?.datasetId === datasetId ? run?.id : null;
  const ready = useEmbeddingReady(datasetId, embedding);
  const wanted = arrows.on && !!runId && ready && !arrowsBlocked(dimensions, embedding, pcaPair);
  const [grid, setGrid] = useState<(VelocityGrid & { key: string }) | null>(null);
  const key = wanted ? `${datasetId}:${embedding}:${arrows.density}:${runId}` : null;

  useEffect(() => {
    if (!key || !datasetId) return;
    const ctl = new AbortController();
    getVelocityGrid(datasetId, embedding, arrows.density, ctl.signal)
      .then((g) => {
        setGrid({ ...g, key });
        useUnifiedViewStore.getState().setVelocityGridNote(null);
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setGrid(null);
        useUnifiedViewStore.getState().setVelocityGridNote(e instanceof Error ? e.message : String(e));
      });
    return () => ctl.abort();
  }, [key, datasetId, embedding, arrows.density]);

  return useMemo(() => {
    if (!key || !grid || grid.key !== key) return NO_LAYERS;
    return [
      new LineLayer<ArrowSegment>({
        id: "velocity-arrows",
        data: arrowSegments(grid, arrows.scale),
        getSourcePosition: (d) => d.from,
        getTargetPosition: (d) => d.to,
        getColor: arrowColor(background),
        getWidth: 1.3,
        widthUnits: "pixels",
        pickable: false,
      }) as unknown as Layer,
    ];
  }, [key, grid, arrows.scale, background]);
}
