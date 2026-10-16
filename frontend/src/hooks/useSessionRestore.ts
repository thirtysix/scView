import { useEffect, useRef } from "react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { SESSION_FORMAT, SESSION_VERSION, sessionFromStorage, type SessionFile } from "@/lib/session";
import { restoreSession } from "@/lib/sessionRestore";

const KEY = "scview.session.v1";

/**
 * Restores the workspace across reloads: the active panel, the scatter
 * color/embedding, and the last-open dataset (re-fetched, since the full object
 * isn't serialized). Only this small, intentional slice is persisted — transient
 * state (selections, overlays, loading flags) is deliberately left out; a saved
 * session file (lib/session.ts) holds the rest. The slice is stored in that
 * file's format and restored the same way (older flat snapshots still read).
 */
export function useSessionRestore() {
  const restored = useRef(false);

  // Restore once, on mount; persist only after that, so the defaults the app
  // starts with don't overwrite the snapshot while it is being restored.
  useEffect(() => {
    let session: SessionFile | null = null;
    try {
      const raw = localStorage.getItem(KEY);
      session = raw ? sessionFromStorage(JSON.parse(raw)) : null;
    } catch {
      session = null;
    }
    if (!session) {
      restored.current = true;
      return;
    }
    restoreSession(session)
      .catch(() => {
        /* dataset gone or unreachable — stay where we are */
      })
      .finally(() => {
        restored.current = true;
      });
  }, []);

  // Persist the slice whenever it changes (after the initial restore).
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const activePanel = useViewStore((s) => s.activePanel);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const colorByChosen = useSettingsStore((s) => s.colorByChosen);
  const embedding = useSettingsStore((s) => s.embedding);
  const categoricalPalette = useSettingsStore((s) => s.categoricalPalette);
  const continuousColormap = useSettingsStore((s) => s.continuousColormap);
  useEffect(() => {
    if (!restored.current) return;
    const snap: SessionFile = {
      format: SESSION_FORMAT,
      version: SESSION_VERSION,
      ...(datasetId ? { dataset: { id: datasetId } } : {}),
      view: { panel: activePanel, color_by: colorBy, color_by_chosen: colorByChosen, embedding },
      colors: { categorical_palette: categoricalPalette, continuous_colormap: continuousColormap },
    };
    try {
      localStorage.setItem(KEY, JSON.stringify(snap));
    } catch {
      /* storage unavailable — non-fatal */
    }
  }, [datasetId, activePanel, colorBy, colorByChosen, embedding, categoricalPalette, continuousColormap]);
}
