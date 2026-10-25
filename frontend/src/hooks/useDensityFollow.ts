import { useEffect } from "react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useEmbeddingReady } from "@/hooks/useEmbeddingReady";
import { currentDensityView, sameDensityView, showDensity } from "@/lib/density";

/**
 * A density overlay follows the plot: switching the embedding, the PCA pair or
 * the expression layer computes it again for what is on screen (the bandwidth
 * belongs to the layout, so the old values would be wrong on the new one).
 */
export function useDensityFollow(): void {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const embedding = useSettingsStore((s) => s.embedding);
  const pcaDims = useSettingsStore((s) => s.pcaDims);
  const layer = useSettingsStore((s) => s.expressionLayer);
  const overlay = useUnifiedViewStore((s) => s.scatterOverlay);
  const overlayDatasetId = useUnifiedViewStore((s) => s.overlayDatasetId);
  const ready = useEmbeddingReady(datasetId, embedding);
  const density = overlay?.type === "density" && overlayDatasetId === datasetId ? overlay : null;

  useEffect(() => {
    if (!density || !datasetId || !ready) return;
    if (sameDensityView(density, currentDensityView())) return;
    showDensity(datasetId, density.genes).catch((err) => {
      console.error("Failed to recompute the density for this plot:", err);
      const store = useUnifiedViewStore.getState();
      if (store.scatterOverlay?.type === "density") {
        store.setScatterOverlay(null);
        store.setOverlayValues(null);
        store.setOverlayLabel("");
      }
    });
  }, [density, datasetId, ready, embedding, pcaDims, layer]);
}
