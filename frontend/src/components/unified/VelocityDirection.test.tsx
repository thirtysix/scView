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
  const onGene = vi.fn();
  const onExclude = vi.fn();
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
      onGene={onGene}
      onExclude={onExclude}
      {...props}
    />,
  );
  return { ...utils, setChoice, onChecked, onGene, onExclude, choice: () => choice };
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

  it("lists the genes that may turn the direction, opens one, and leaves them out on request", () => {
    const flags = {
      checked: true, rule: "A velocity gene is flagged when ... the MURK pattern.", n_velocity_genes: 311, n_flagged: 2,
      top_n: 50, n_in_top: 1, n_cells: 9815, sampled: false, steps: ["Ery2 -> Ery3"],
      genes: [
        { gene: "Cox6b2", r2: 0.7, rank: 28, steps: [{ from: "Ery2", to: "Ery3", status: "backward" as const, spliced_from: 0.27, spliced_to: 0.53, ratio_from: 0.18, ratio_to: 0.29, share_negative: 0.755 }] },
        { gene: "Dcxr", r2: 0.5, rank: 38, steps: [{ from: "Ery2", to: "Ery3", spliced_from: 0.06, spliced_to: 0.09, ratio_from: 0.008, ratio_to: 0.064, share_negative: 0.99 }] },
      ],
    };
    const { onGene, onExclude } = renderIt({ run: { ...run, direction: { ...backward, gene_flags: flags } } as VelocityRun });
    expect(screen.getByText(/2 of 311 velocity genes \(1 of the top 50 by r2\)/)).toBeTruthy();
    expect(screen.getByText(/the MURK pattern/)).toBeTruthy();
    const cox = screen.getByText("Cox6b2").closest("button")!;
    expect(cox.title).toMatch(/unspliced\/spliced 0.18 -> 0.29, spliced 0.27 -> 0.53, velocity below 0 in 76% of Ery2/);
    fireEvent.click(cox);
    expect(onGene).toHaveBeenCalledWith("Cox6b2");
    fireEvent.click(screen.getByText("Leave these out and run again"));
    expect(onExclude).toHaveBeenCalledWith(["Cox6b2", "Dcxr"]);
    cleanup();
    const forward = { ...backward, transitions: [backward.transitions![0]!], n_backward: 0, gene_flags: flags };
    renderIt({ run: { ...run, direction: forward } as VelocityRun });
    expect(screen.getByText(/No transition runs backward, so they matter less here/)).toBeTruthy();
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
