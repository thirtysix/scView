import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { DEComposition } from "@/api/de";

const { fetchDEComposition } = vi.hoisted(() => ({ fetchDEComposition: vi.fn() }));
vi.mock("@/api/de", () => ({ fetchDEComposition }));

import { DeCompositionNote } from "./DeCompositionNote";

const group = (group: string, share_a: number, share_b: number, flagged = false, one_sided = false) => ({
  group, n_a: Math.round(share_a * 1000), n_b: Math.round(share_b * 1000), share_a, share_b,
  diff_points: Math.round((share_a - share_b) * 1000) / 10, flagged, one_sided,
});

const SHIFTED: DEComposition = {
  column: "cell_type", label_a: "STIM", label_b: "CTRL", auto: true, n_a: 1000, n_b: 1000,
  groups: [group("Monocytes", 0.42, 0.18, true), group("pDC", 0.0, 0.12, true, true), group("B", 0.2, 0.2)],
  n_groups: 3, warn: true, message: "Monocytes is 42% of STIM cells ...", note: null,
  threshold_points: 10, min_cells: 10,
};

const params = { split_by: "stim", group_a: "STIM", group_b: "CTRL" };

function renderNote(over: Partial<Parameters<typeof DeCompositionNote>[0]> = {}) {
  const onCompareWithin = vi.fn();
  const onColumnChange = vi.fn();
  const utils = render(
    <DeCompositionNote
      datasetId="ds"
      params={params}
      columns={["stim", "cell_type", "leiden"]}
      column=""
      onColumnChange={onColumnChange}
      onCompareWithin={onCompareWithin}
      scopeColumns={["cell_type", "leiden"]}
      {...over}
    />,
  );
  return { ...utils, onCompareWithin, onColumnChange };
}

beforeEach(() => {
  fetchDEComposition.mockReset().mockResolvedValue(SHIFTED);
});

describe("DeCompositionNote", () => {
  it("lists the flagged groups with both shares and restricts the DE to one in a click", async () => {
    const { onCompareWithin } = renderNote();
    expect(await screen.findByText("Composition differs between the sides")).toBeInTheDocument();
    expect(fetchDEComposition.mock.calls[0]![1]).toEqual({ ...params, within_column: undefined, within_value: undefined, column: undefined });
    const note = screen.getByTestId("de-composition");
    expect(note).toHaveTextContent("Monocytes: 42% of STIM vs 18% of CTRL (+24 points)");
    expect(note).toHaveTextContent("pDC: 0% of STIM vs 12% of CTRL (−12 points)nearly absent from STIM");
    expect(note).not.toHaveTextContent("B: 20%");
    expect(note).toHaveTextContent("n = 1,000 STIM cells; 1,000 CTRL cells.");
    fireEvent.click(screen.getByRole("button", { name: "Compare within Monocytes" }));
    expect(onCompareWithin).toHaveBeenCalledWith("cell_type", "Monocytes");
  });

  it("shows every group in a table with header tooltips", async () => {
    renderNote();
    fireEvent.click(await screen.findByRole("button", { name: "Show the table (3 groups)" }));
    const headers = screen.getAllByRole("columnheader");
    expect(headers.map((h) => h.textContent)).toEqual(["Group", "STIM", "CTRL", "Points", "n"]);
    expect(headers.every((h) => h.getAttribute("title"))).toBe(true);
    expect(screen.getAllByRole("row")).toHaveLength(4);
  });

  it("says the largest difference when nothing is flagged", async () => {
    fetchDEComposition.mockResolvedValue({
      ...SHIFTED, warn: false, message: null, groups: [group("CD14 Mono", 0.28, 0.334), group("B", 0.078, 0.066)],
    });
    renderNote();
    expect(await screen.findByText(/the largest is CD14 Mono, 28% of STIM vs 33% of CTRL/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Compare within/ })).toBeNull();
  });

  it("shows the backend's note when the check is degenerate", async () => {
    fetchDEComposition.mockResolvedValue({
      ...SHIFTED, column: "leiden", warn: false, message: null, groups: [], n_groups: 0, n_a: 0, n_b: 0,
      note: "Composition by 'leiden', the compared column itself, is the comparison.",
    });
    renderNote({ params: { split_by: "leiden", group_a: "0", group_b: "1" } });
    expect(await screen.findByText(/the compared column itself/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Show the table/ })).toBeNull();
  });

  it("passes the picked column and the scope, and offers no chip outside the Within columns", async () => {
    const { onColumnChange } = renderNote({
      column: "leiden",
      params: { ...params, within_column: "donor", within_value: "d1" },
      scopeColumns: ["donor"],
    });
    await screen.findByText("Composition differs between the sides");
    expect(fetchDEComposition.mock.calls[0]![1]).toMatchObject({ column: "leiden", within_column: "donor", within_value: "d1" });
    expect(screen.queryByRole("button", { name: /Compare within/ })).toBeNull();
    const pick = screen.getByRole("combobox", { name: "Composition by" }) as HTMLSelectElement;
    // The compared column is not offered.
    expect(Array.from(pick.options).map((o) => o.value)).toEqual(["", "cell_type", "leiden"]);
    fireEvent.change(pick, { target: { value: "cell_type" } });
    expect(onColumnChange).toHaveBeenCalledWith("cell_type");
  });

  it("asks nothing for two identical groups, and reports a failed check", async () => {
    const { unmount } = renderNote({ params: { ...params, group_b: "STIM" } });
    expect(screen.queryByTestId("de-composition")).toBeNull();
    expect(fetchDEComposition).not.toHaveBeenCalled();
    unmount();
    fetchDEComposition.mockRejectedValue(new Error("API error 404: Column 'x' not found."));
    renderNote();
    await waitFor(() => expect(screen.getByText(/Composition check failed: API error 404/)).toBeInTheDocument());
  });
});
