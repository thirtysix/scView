import { useEffect, useRef } from "react";
import { getDataset } from "@/api/datasets";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { useSettingsStore } from "@/stores/settingsStore";
import type { PanelId } from "@/lib/constants";

const KEY = "scview.session.v1";

interface Snapshot {
  datasetId?: string | null;
  activePanel?: PanelId;
  colorBy?: string;
  embedding?: string;
}

/**
 * Restores the workspace across reloads: the active panel, the scatter
 * color/embedding, and the last-open dataset (re-fetched, since the full object
 * isn't serialized). Only this small, intentional slice is persisted — transient
 * state (selections, overlays, loading flags) is deliberately left out.
 */
export function useSessionRestore() {
  const restored = useRef(false);

  // Restore once, on mount.
  useEffect(() => {
    if (restored.current) return;
    restored.current = true;
    let snap: Snapshot | null = null;
    try {
      const raw = localStorage.getItem(KEY);
      snap = raw ? (JSON.parse(raw) as Snapshot) : null;
    } catch {
      snap = null;
    }
    if (!snap) return;
    if (snap.colorBy) useSettingsStore.getState().setColorBy(snap.colorBy);
    if (snap.embedding) useSettingsStore.getState().setEmbedding(snap.embedding);
    if (snap.datasetId) {
      getDataset(snap.datasetId)
        .then((d) => {
          useDatasetStore.getState().setCurrentDataset(d);
          if (snap!.activePanel) useViewStore.getState().setPanel(snap!.activePanel);
        })
        .catch(() => {
          /* dataset no longer exists — stay on the load panel */
        });
    } else if (snap.activePanel) {
      useViewStore.getState().setPanel(snap.activePanel);
    }
  }, []);

  // Persist the slice whenever it changes (after the initial restore).
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const activePanel = useViewStore((s) => s.activePanel);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const embedding = useSettingsStore((s) => s.embedding);
  useEffect(() => {
    if (!restored.current) return;
    const snap: Snapshot = { datasetId, activePanel, colorBy, embedding };
    try {
      localStorage.setItem(KEY, JSON.stringify(snap));
    } catch {
      /* storage unavailable — non-fatal */
    }
  }, [datasetId, activePanel, colorBy, embedding]);
}
