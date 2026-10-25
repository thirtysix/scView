import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render } from "@testing-library/react";
import { useSelectionStore } from "@/stores/selectionStore";
import { SELECTION_COLOR } from "@/lib/linkedSelection";

const { plotProps } = vi.hoisted(() => ({ plotProps: { current: null as Record<string, unknown> | null } }));
vi.mock("@/components/plots/Plot", () => ({
  default: (props: Record<string, unknown>) => {
    plotProps.current = props;
    return <div data-testid="plot" />;
  },
}));

import { ViolinPlot } from "./ViolinPlot";
import { StackedViolin } from "./StackedViolin";

type Trace = { name: string; y: ArrayLike<number>; line: { color: string } };
const traces = () => plotProps.current!.data as Trace[];

beforeEach(() => {
  plotProps.current = null;
  useSelectionStore.setState({ selectedCellIndices: null });
});

describe("C035 selection in the violins", () => {
  it("ViolinPlot draws an extra group last, in its own colour", () => {
    render(
      <ViolinPlot
        data={{ NK: [1, 2], B: [0, 3] }}
        extra={{ name: "Selection (2 cells)", values: Float32Array.from([5, 6]), color: SELECTION_COLOR }}
      />,
    );
    const t = traces();
    expect(t.map((x) => x.name)).toEqual(["NK", "B", "Selection (2 cells)"]);
    expect(Array.from(t[2]!.y)).toEqual([5, 6]);
    expect(t[2]!.line.color).toBe(SELECTION_COLOR);
  });

  it("StackedViolin adds a Selection column per gene, from the selected cells among those drawn", () => {
    const codes = Int32Array.from([0, 0, 1, 1]);
    const values = { G1: Float32Array.from([1, 2, 3, 4]), G2: Float32Array.from([10, 20, 30, 40]) };
    useSelectionStore.setState({ selectedCellIndices: new Set([1, 2, 3]) });
    const { rerender } = render(
      <StackedViolin genes={["G1", "G2"]} values={values} codes={codes} groupColumn="none" cells={Uint32Array.from([0, 1, 2])} />,
    );
    const sel = traces().filter((t) => t.name === "Selection (3 cells)");
    expect(sel).toHaveLength(2);
    expect(Array.from(sel[0]!.y)).toEqual([2, 3]); // cell 3 is not drawn
    expect(Array.from(sel[1]!.y)).toEqual([20, 30]);
    // All cells drawn: every selected one.
    rerender(<StackedViolin genes={["G1"]} values={values} codes={codes} groupColumn="none" />);
    expect(Array.from(traces().at(-1)!.y)).toEqual([2, 3, 4]);
    act(() => useSelectionStore.getState().clearSelection());
    expect(traces().some((t) => t.name.startsWith("Selection"))).toBe(false);
  });
});
