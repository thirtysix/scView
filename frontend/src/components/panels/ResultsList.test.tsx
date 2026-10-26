import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

const api = vi.hoisted(() => ({
  listResults: vi.fn(),
  deleteResult: vi.fn(),
  pinResult: vi.fn(),
  restoreResult: vi.fn(),
  compareWithCurrent: vi.fn(),
}));
vi.mock("@/api/results", async (orig) => ({ ...(await orig<typeof import("@/api/results")>()), ...api }));
const exp = vi.hoisted(() => ({ downloadUrl: vi.fn() }));
vi.mock("@/api/export", () => exp);

import { ResultsList, columnOf } from "./ResultsList";
import { useObsRevisionStore } from "@/stores/obsRevisionStore";

const row = (over: Record<string, unknown>) => ({
  kind: "markers", key: "leiden", status: "superseded", version: "0.40.0", bytes: 2048, stale: false,
  created_at: "2026-10-26T10:00:00+00:00", ...over,
});
const LIST = {
  keep_superseded: 3, max_bytes: 200_000_000, total_bytes: 6144,
  results: [
    row({ id: "m1", label: "Markers of 'leiden' (t-test)" }),
    row({ id: "m2", label: "Markers of 'cluster' (wilcoxon)", key: "cluster", stale: true }),
    row({ id: "d1", kind: "de", key: "x", label: "CD14 Mono: STIM vs CTRL (Wilcoxon)", status: "primary" }),
  ],
};

describe("ResultsList", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listResults.mockResolvedValue(LIST);
    useObsRevisionStore.setState({ revisions: {} });
  });

  it("lists results with kind, earlier and stale badges, and filters by kind", async () => {
    render(<ResultsList datasetId="ds" />);
    const items = await screen.findAllByTestId("stored-result");
    expect(items).toHaveLength(3);
    expect(within(items[1]!).getByText("earlier data")).toBeInTheDocument();
    expect(within(items[0]!).queryByText("earlier data")).toBeNull();
    expect(screen.getByText(/3 stored, 6.14 kB of 200 MB/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "DE" }));
    expect(screen.getAllByTestId("stored-result")).toHaveLength(1);
  });

  it("restores an earlier markers result and bumps its column; a stale one cannot be restored", async () => {
    api.restoreResult.mockResolvedValue({ restored: "m1", kind: "markers" });
    render(<ResultsList datasetId="ds" />);
    await screen.findAllByTestId("stored-result");
    expect(screen.getByRole("button", { name: /Make Markers of 'cluster'.* current again/ })).toBeDisabled();
    expect(screen.queryByRole("button", { name: /Make CD14 Mono/ })).toBeNull();  // DE: not restorable
    fireEvent.click(screen.getByRole("button", { name: /Make Markers of 'leiden'.* current again/ }));
    await waitFor(() => expect(api.restoreResult).toHaveBeenCalledWith("ds", "m1"));
    expect(await screen.findByRole("status")).toHaveTextContent("is the current result again");
    expect(useObsRevisionStore.getState().revisions["ds\u0000leiden"]).toBe(1);
  });

  it("shows a restore refusal on its row", async () => {
    api.restoreResult.mockRejectedValue(new Error('API error 409: {"detail": "computed on earlier groups"}'));
    render(<ResultsList datasetId="ds" />);
    await screen.findAllByTestId("stored-result");
    fireEvent.click(screen.getByRole("button", { name: /Make Markers of 'leiden'.* current again/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("computed on earlier groups");
  });

  it("compares markers with the current ones", async () => {
    api.compareWithCurrent.mockResolvedValue({
      id: "m1", key: "leiden", kind: "markers", n: 50, earlier: { method: "t-test" }, current: { method: "wilcoxon" },
      groups: [{ group: "0", shared: 41, jaccard: 0.7, only_earlier: ["A"], only_current: ["B"] }],
    });
    render(<ResultsList datasetId="ds" />);
    await screen.findAllByTestId("stored-result");
    fireEvent.click(screen.getByRole("button", { name: /Compare Markers of 'leiden'/ }));
    const view = await screen.findByTestId("compare-markers");
    expect(view).toHaveTextContent("t-test");
    expect(within(view).getByText("41/50")).toBeInTheDocument();
  });

  it("pins, deletes after a confirm, and downloads", async () => {
    api.pinResult.mockResolvedValue({ pinned: true });
    api.deleteResult.mockResolvedValue({ deleted: "d1" });
    render(<ResultsList datasetId="ds" />);
    await screen.findAllByTestId("stored-result");
    fireEvent.click(screen.getByRole("button", { name: /^Pin CD14 Mono/ }));
    await waitFor(() => expect(api.pinResult).toHaveBeenCalledWith("ds", "d1", true));
    fireEvent.click(screen.getByRole("button", { name: /^Delete CD14 Mono/ }));
    expect(api.deleteResult).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete?" }));
    await waitFor(() => expect(api.deleteResult).toHaveBeenCalledWith("ds", "d1"));
    fireEvent.click(screen.getByRole("button", { name: /Download CD14 Mono/ }));
    expect(exp.downloadUrl).toHaveBeenCalledWith(expect.stringContaining("/datasets/ds/results/d1/table.tsv.gz"));
  });

  it("names the column of an enrichment result", () => {
    expect(columnOf(row({ id: "e", kind: "enrichment", key: "leiden|3", label: "" }) as never)).toBe("leiden");
  });
});
