import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { apiFetch, plotProps, codesOf } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  plotProps: { current: null as null | Record<string, unknown> },
  codesOf: vi.fn(),
}));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/components/plots/Plot", () => ({
  default: (props: Record<string, unknown>) => {
    plotProps.current = props;
    return <div data-testid="plot" />;
  },
}));
vi.mock("@/hooks/useObsCodes", () => ({
  useObsCodes: (_d: unknown, _e: unknown, column: string | null) => ({ codes: column ? codesOf(column) : null, error: false }),
}));

import { ObservationsPanel } from "./ObservationsPanel";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";

interface Trace { name: string; type: string; x: unknown; y: number[]; customdata?: unknown[][] }
const traces = () => (plotProps.current?.data ?? []) as Trace[];

const INFO = {
  id: "kang",
  name: "Kang",
  n_cells: 100,
  obs_columns: [
    { name: "cluster", dtype: "category", n_unique: 2, values: ["CD14 Mono", "B"] },
    { name: "Donor", dtype: "category", n_unique: 3, values: ["d1", "d2", "d10"] },
  ],
  active_clustering: "cluster",
} as unknown as DatasetInfo;

// 60 CD14 Mono cells (0-59), then 40 B (60-99).
const CLUSTER = Int32Array.from({ length: 100 }, (_, i) => (i < 60 ? 0 : 1));

function respond(url: string) {
  if (url.includes("/metadata/summary")) return Promise.resolve({ groupby: "cluster", counts: { B: 40, "CD14 Mono": 60 } });
  if (url.includes("row=cluster&col=Donor")) {
    return Promise.resolve({ row_values: ["B", "CD14 Mono"], col_values: ["d1", "d10", "d2"], counts: [[10, 20, 10], [30, 10, 20]] });
  }
  return Promise.reject(new Error(`unexpected ${url}`));
}

describe("ObservationsPanel composition bars with a selection (C035)", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockImplementation(respond);
    codesOf.mockReset().mockImplementation((c: string) => (c === "cluster" ? CLUSTER : null));
    plotProps.current = null;
    useDatasetStore.setState({ currentDatasetId: "kang", currentDataset: INFO });
    useSelectionStore.setState({ selectedCellIndices: null });
  });

  it("marks the selected cells in each bar, as a share in % mode, and follows a cleared selection", async () => {
    const sel = new Set<number>();
    for (let i = 0; i < 30; i++) sel.add(i); // 30 of CD14 Mono's 60
    for (let i = 60; i < 70; i++) sel.add(i); // 10 of B's 40
    useSelectionStore.setState({ selectedCellIndices: sel });
    render(<ObservationsPanel />);
    await waitFor(() => expect(traces()).toHaveLength(4));
    const diamonds = traces()[3]!;
    expect(diamonds.name).toBe("Selection (40 cells)");
    expect(diamonds.x).toEqual(["CD14 Mono", "B"]);
    expect(diamonds.y).toEqual([30, 10]);
    expect(screen.getByTestId("selection-chip")).toHaveTextContent("Selection: 40 cells");

    fireEvent.click(screen.getByRole("button", { name: "%" }));
    expect(traces()[3]!.y).toEqual([50, 25]);

    act(() => useSelectionStore.getState().clearSelection());
    expect(traces()).toHaveLength(3);
    expect(screen.queryByTestId("selection-chip")).toBeNull();
  });
});
