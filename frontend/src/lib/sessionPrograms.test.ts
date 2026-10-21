import { describe, it, expect, vi, beforeEach } from "vitest";

const api = vi.hoisted(() => ({ getDataset: vi.fn(), apiFetch: vi.fn() }));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
vi.mock("@/api/client", () => ({ apiFetch: api.apiFetch }));

import type { DatasetInfo } from "@/api/types";
import { parseSession, sessionJson, SessionError } from "@/lib/session";
import { captureSession, restoreSession } from "@/lib/sessionRestore";
import { describeSharedView } from "@/lib/shareUrl";
import { useDatasetStore } from "@/stores/datasetStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const kang: DatasetInfo = {
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 100,
  n_genes: 50,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [],
  cells_fingerprint: "fp-kang",
  created_at: "",
};

beforeEach(() => {
  vi.clearAllMocks();
  api.getDataset.mockResolvedValue(kang);
  useDatasetStore.getState().setCurrentDataset(kang);
  const u = useUnifiedViewStore.getState();
  u.clearOverlayState();
  u.setScatterOverlay({ type: "program", index: 3, name: "P3" });
  u.setOverlayDatasetId("kang");
});

describe("a gene program overlay in a saved session", () => {
  it("saves the program's number and asks for its usage again on open", async () => {
    const file = sessionJson(captureSession(new Date("2026-10-22T10:00:00Z")));
    const session = parseSession(JSON.parse(file));
    expect(session.view?.overlay).toEqual({ type: "program", index: 3 });
    expect(describeSharedView(session)).toContain("the gene program P3 usage overlay");
    useUnifiedViewStore.getState().clearOverlayState();
    useUnifiedViewStore.setState({ pendingOverlay: null });
    await restoreSession(session);
    expect(useUnifiedViewStore.getState().pendingOverlay).toEqual({
      datasetId: "kang",
      overlay: { type: "program", index: 3 },
    });
  });

  it("rejects a program overlay without a whole-number index", () => {
    const file = JSON.parse(sessionJson(captureSession(new Date("2026-10-22T10:00:00Z"))));
    for (const index of [0, 1.5, "3", undefined]) {
      file.view.overlay = { type: "program", index };
      expect(() => parseSession(file)).toThrow(SessionError);
    }
  });
});
