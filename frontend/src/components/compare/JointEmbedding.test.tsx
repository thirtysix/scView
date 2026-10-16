import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { listDatasets } = vi.hoisted(() => ({ listDatasets: vi.fn() }));
vi.mock("@/api/datasets", () => ({ listDatasets, getDataset: vi.fn() }));
vi.mock("@/api/integration", () => ({
  planIntegration: vi.fn(),
  runIntegration: vi.fn(),
  getJointSummary: vi.fn().mockResolvedValue({ dataset_id: "", joint: false }),
}));

import { JointEmbedding } from "./JointEmbedding";

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
