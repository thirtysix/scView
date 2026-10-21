import { describe, it, expect, vi, beforeEach } from "vitest";

const api = vi.hoisted(() => ({ getDataset: vi.fn(), apiFetch: vi.fn() }));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
vi.mock("@/api/client", () => ({ apiFetch: api.apiFetch }));

import type { DatasetInfo } from "@/api/types";
import { parseSession, sessionJson, SessionError } from "@/lib/session";
import { captureSession, restoreSession } from "@/lib/sessionRestore";
import { describeSharedView } from "@/lib/shareUrl";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const kang = (extra: Partial<DatasetInfo> = {}): DatasetInfo => ({
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 100,
  n_genes: 50,
  status: "ready",
  available_embeddings: ["X_umap", "X_pca"],
  obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "NK"] }],
  cells_fingerprint: "fp-kang",
  created_at: "",
  ...extra,
});

const ref = { embedding: "X_pca", dims: [2, 3] as [number, number], k: 15, chance: 0.15, random: 0.14 };

function saveAndReset() {
  const file = sessionJson(captureSession(new Date("2026-10-21T20:00:00Z")));
  useUnifiedViewStore.getState().clearOverlayState();
  useUnifiedViewStore.setState({ pendingOverlay: null });
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  return parseSession(JSON.parse(file));
}

beforeEach(() => {
  vi.clearAllMocks();
  api.getDataset.mockResolvedValue(kang());
  useDatasetStore.getState().setCurrentDataset(kang());
  useSettingsStore.setState({ embedding: "X_pca" });
  const u = useUnifiedViewStore.getState();
  u.clearOverlayState();
  u.setScatterOverlay({ type: "score", name: "Neighbour preservation (PCA, PC3 vs PC4)", reliability: ref });
  u.setOverlayDatasetId("kang");
});

describe("plot reliability in a saved session", () => {
  it("saves the plot it scored and asks for it again on open", async () => {
    const session = saveAndReset();
    expect(session.view?.overlay).toEqual({ type: "reliability", embedding: "X_pca", dims: [2, 3] });
    expect(describeSharedView(session)).toContain("the plot reliability overlay (PCA)");
    await restoreSession(session);
    expect(useUnifiedViewStore.getState().pendingOverlay).toEqual({
      datasetId: "kang",
      overlay: { type: "reliability", embedding: "X_pca", dims: [2, 3] },
    });
  });

  it("saves a 2D UMAP without dims, and skips an embedding the dataset lacks", async () => {
    useUnifiedViewStore.getState().setScatterOverlay({
      type: "score",
      name: "Neighbour preservation (UMAP)",
      reliability: { embedding: "X_umap", k: 15, chance: 0.15, random: 0.14 },
    });
    const session = saveAndReset();
    expect(session.view?.overlay).toEqual({ type: "reliability", embedding: "X_umap" });
    api.getDataset.mockResolvedValue(kang({ available_embeddings: ["X_pca"] }));
    const report = await restoreSession(session);
    expect(report.notes).toContain("Plot reliability of X_umap: the embedding is not in this dataset.");
    expect(useUnifiedViewStore.getState().pendingOverlay).toBeNull();
  });

  it("rejects a malformed reliability overlay", () => {
    const file = JSON.parse(sessionJson(captureSession(new Date("2026-10-21T20:00:00Z"))));
    file.view.overlay = { type: "reliability", embedding: "X_pca", dims: [1] };
    expect(() => parseSession(file)).toThrow(SessionError);
    file.view.overlay = { type: "reliability", embedding: "" };
    expect(() => parseSession(file)).toThrow(SessionError);
  });
});
