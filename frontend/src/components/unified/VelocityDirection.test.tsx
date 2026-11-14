import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { VelocityDirection as Direction, VelocityRun } from "@/api/velocity";

const api = vi.hoisted(() => ({ checkVelocityDirection: vi.fn() }));
vi.mock("@/api/velocity", () => api);

import { choiceFrom, DEFAULT_DIRECTION, directionSpec, VelocityDirection, type DirectionChoice } from "./VelocityDirection";

const backward: Direction = {
  checked: true,
  kind: "order",
  label: "the order of celltype given",
  spec: { group_by: "celltype", order: ["Ery1", "Ery2", "Ery3"] },
  group_column: "celltype",
  embedding: "X_umap_original",
  k: 30,
  transitions: [
    { from: "Ery1", to: "Ery2", n_cells: 242, mean_cos: 0.335, low: 0.249, high: 0.421, share_forward: 0.715, status: "forward" },
    { from: "Ery2", to: "Ery3", n_cells: 169, mean_cos: -0.189, low: -0.301, high: -0.076, share_forward: 0.373, status: "backward" },
  ],
  n_backward: 1,
  cells: { n_cells: 1650, n_run_cells: 9815, mean_cos: 0.162, share_forward: 0.601, by_group: [] },
  note: "Direction check: it cannot say which is wrong.",
};

const run = {
  result_id: "r1",
  group_column: "celltype",
  clusters: [{ group: "Ery1" }, { group: "Ery2" }, { group: "Ery3" }],
  direction: backward,
} as unknown as VelocityRun;

function renderIt(props: Partial<Parameters<typeof VelocityDirection>[0]> = {}) {
  let choice: DirectionChoice = props.choice ?? DEFAULT_DIRECTION;
  const setChoice = vi.fn((c: DirectionChoice) => {
    choice = c;
  });
  const onChecked = vi.fn();
  const utils = render(
    <VelocityDirection
      datasetId="d1"
      run={run}
      embedding="X_umap_original"
      embeddingOk
      columns={[{ name: "stage", dtype: "category", n_unique: 7 }]}
      choice={choice}
      setChoice={setChoice}
      onChecked={onChecked}
      {...props}
    />,
  );
  return { ...utils, setChoice, onChecked, choice: () => choice };
}

describe("VelocityDirection", () => {
  beforeEach(() => {
    api.checkVelocityDirection.mockReset(); // in braces: a returned function would run as the test's cleanup
  });
  afterEach(cleanup);

  it("names a backward transition with its numbers and says what the check cannot tell", () => {
    renderIt();
    expect(screen.getByText(/Backward at Ery2 -> Ery3 \(-0.19, n = 169\)/)).toBeTruthy();
    expect(screen.getByText("BACKWARD")).toBeTruthy();
    expect(screen.getByText("-0.19 (-0.30 to -0.08)")).toBeTruthy();
    expect(screen.getByText("37%")).toBeTruthy();
    expect(screen.getByText(/cannot say which is wrong/)).toBeTruthy();
    expect(screen.getByText(/1,650 of 9,815 cells/)).toBeTruthy();
    expect(screen.queryByText(/Check on this plot/)).toBeNull();
  });

  it("says unchecked and why, and offers a re-check on the plot on screen", async () => {
    const r = { ...run, direction: { ...backward, embedding: "X_umap" } } as VelocityRun;
    api.checkVelocityDirection.mockResolvedValue({ direction: backward, warnings: [], run: { ...run } });
    const { onChecked } = renderIt({ run: r, embedding: "X_umap_original" });
    fireEvent.click(screen.getByText("Check on this plot"));
    await waitFor(() => expect(onChecked).toHaveBeenCalled());
    expect(api.checkVelocityDirection).toHaveBeenCalledWith("d1", { group_by: "celltype", embedding: "X_umap_original" });
    cleanup();
    renderIt({ run: { ...run, direction: { checked: false, reason: "No order to check against.", embedding: "X_umap", note: "" } } as VelocityRun });
    expect(screen.getByText(/Direction unchecked: No order to check against./)).toBeTruthy();
  });

  it("sends the order the user gives, and an error in words", async () => {
    api.checkVelocityDirection.mockImplementation(async () => {
      throw new Error("API error 422: 'celltype' has no group 'Ery9'.");
    });
    renderIt({ choice: { ...DEFAULT_DIRECTION, mode: "order", order: "Ery1 > Ery9" } });
    fireEvent.click(screen.getByText("Check direction"));
    await screen.findByText(/no group 'Ery9'/);
    expect(api.checkVelocityDirection).toHaveBeenCalledWith("d1", {
      group_by: "celltype", order: ["Ery1", "Ery9"], embedding: "X_umap_original",
    });
  });

  it("switching to an order fills in the table's groups", () => {
    const { setChoice } = renderIt();
    fireEvent.change(screen.getByLabelText("Check against"), { target: { value: "order" } });
    expect(setChoice).toHaveBeenCalledWith(expect.objectContaining({ mode: "order", order: "Ery1 > Ery2 > Ery3", root: "Ery1" }));
  });
});

describe("directionSpec / choiceFrom", () => {
  it("builds the spec for each mode and refuses an unfinished one", () => {
    expect(directionSpec(DEFAULT_DIRECTION, "g")).toEqual({ group_by: "g" });
    expect(directionSpec({ ...DEFAULT_DIRECTION, mode: "order", order: "A" }, "g")).toBeNull();
    expect(directionSpec({ ...DEFAULT_DIRECTION, mode: "order", order: " A > B >" }, null)).toEqual({ order: ["A", "B"] });
    expect(directionSpec({ ...DEFAULT_DIRECTION, mode: "root", root: "A" }, "g")).toEqual({ group_by: "g", root: "A" });
    expect(directionSpec({ ...DEFAULT_DIRECTION, mode: "column" }, "g")).toBeNull();
    expect(directionSpec({ ...DEFAULT_DIRECTION, mode: "column", column: "day", columnOrder: "d1 > d3" }, "g"))
      .toEqual({ group_by: "g", column: "day", column_order: ["d1", "d3"] });
  });

  it("reads a saved check back into the controls", () => {
    expect(choiceFrom(backward)).toEqual({ ...DEFAULT_DIRECTION, mode: "order", order: "Ery1 > Ery2 > Ery3" });
    expect(choiceFrom({ ...backward, kind: "pseudotime", spec: { column: "scview_dpt_pseudotime" } })).toEqual(DEFAULT_DIRECTION);
    expect(choiceFrom({ ...backward, kind: "column", spec: { column: "stage" } }).mode).toBe("column");
    expect(choiceFrom(null)).toEqual(DEFAULT_DIRECTION);
  });
});
