import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import type { IntegrationEvent, IntegrationPlan } from "@/api/integration";
import type { PreflightVerdict } from "@/api/preflight";

const { listDatasets, planIntegration, runIntegration, cancelPipelineRun } = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  planIntegration: vi.fn(),
  runIntegration: vi.fn(),
  cancelPipelineRun: vi.fn(),
}));
vi.mock("@/api/datasets", () => ({ listDatasets, getDataset: vi.fn() }));
vi.mock("@/api/integration", () => ({
  planIntegration,
  runIntegration,
  getJointSummary: vi.fn().mockResolvedValue({ dataset_id: "", joint: false }),
}));
vi.mock("@/api/pipelineRun", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/pipelineRun")>()),
  cancelPipelineRun,
}));

import { JointEmbedding } from "./JointEmbedding";

const ds = (id: string): DatasetInfo => ({
  id,
  name: id,
  filename: `${id}.h5ad`,
  n_cells: 4992,
  n_genes: 36601,
  status: "ready",
  available_embeddings: [],
  obs_columns: [],
  created_at: "",
});

const VERDICT: PreflightVerdict = {
  level: "tight",
  sentence: "About 5.1 GB at its peak (variable genes), roughly 2 to 6 min. scView has 6.2 GB free of its 10 GB limit.",
  need_bytes: 5.1 * 1024 ** 3,
  peak_bytes: 5.1 * 1024 ** 3,
  peak_stage: "variable genes",
  available_bytes: 6.2 * 1024 ** 3,
  limit_bytes: 10 * 1024 ** 3,
  used_bytes: 3.8 * 1024 ** 3,
  limited: true,
  source: "container limit",
  seconds: [120, 360],
  cached: { count: 0, bytes: 0, frees_enough: false },
  suggestions: [],
  error_band: 0.2,
  breakdown: [],
  notes: [],
};

const PLAN: IntegrationPlan = {
  datasets: ["s1", "s2"].map((id) => ({
    id, name: id, label: id, n_cells: 4279, n_genes: 36601, counts_source: "X", columns: [], error: null,
  })),
  merge: { intersection: 35477, union: 36601, suspicious_low_overlap: false, reconciliation: null, est_genes: 35477, warnings: [] },
  shared_columns: [],
  est_cells: 8558,
  blocking: [],
  preflight: VERDICT,
  ok: true,
};

describe("JointEmbedding run: memory estimate, watchdog, Cancel", () => {
  it("shows the plan's memory estimate, the run's memory warning, and cancels the new dataset's run", async () => {
    listDatasets.mockResolvedValue([ds("s1"), ds("s2")]);
    planIntegration.mockResolvedValue(PLAN);
    cancelPipelineRun.mockResolvedValue({ cancelling: true, step: "pca" });
    let emit!: (e: IntegrationEvent) => void;
    runIntegration.mockImplementation((_body, onEvent: (e: IntegrationEvent) => void) => {
      emit = onEvent;
      return new Promise(() => {});
    });
    render(<JointEmbedding />);
    for (const box of await screen.findAllByRole("checkbox")) fireEvent.click(box);
    expect(await screen.findByText(VERDICT.sentence)).toBeInTheDocument();
    expect(screen.getByText("Memory:")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Build joint embedding/ }));
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull(); // no id yet
    emit({ event: "stage", data: { stage: "merge", message: "Merging 2 datasets", dataset_id: "j1" } });
    emit({ event: "step_start", data: { step: "pca", index: 5, total: 12 } });
    emit({
      event: "watchdog",
      data: { type: "watchdog", kind: "memory", threshold_pct: 80, used_gb: 8.1, limit_gb: 10, pct: 81, limited: true, step: "pca" },
    });
    expect(await screen.findByText(/Memory at 81% of the 10(\.0)? GB limit/)).toBeInTheDocument();
    const cancel = await screen.findByRole("button", { name: "Cancel" });
    expect(cancel.getAttribute("title")).toMatch(/joint dataset is removed/);
    fireEvent.click(cancel);
    expect(cancelPipelineRun).toHaveBeenCalledWith("j1");
    expect(within(cancel).getByText("Cancelling...")).toBeInTheDocument();
    expect(screen.getByText(/Cancelling: PCA finishes first/)).toBeInTheDocument();
  });
});

describe("JointEmbedding dataset picker", () => {
  it("says it is loading, not that the library is empty, until the list arrives", async () => {
    let resolve!: (ds: DatasetInfo[]) => void;
    listDatasets.mockReturnValue(new Promise<DatasetInfo[]>((r) => (resolve = r)));
    render(<JointEmbedding />);
    expect(screen.getByText(/Loading datasets/)).toBeInTheDocument();
    expect(screen.queryByText(/No ready datasets/)).toBeNull();

    resolve([]);
    expect(await screen.findByText(/No ready datasets/)).toBeInTheDocument();
    expect(screen.queryByText(/Loading datasets/)).toBeNull();
  });
});
