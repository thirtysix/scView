import { describe, it, expect, beforeEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useSelectionStore } from "@/stores/selectionStore";
import { selectGate } from "@/stores/gatesStore";
import { bitsetFromIndices, bitsetToSelection, type Gate, type GateEval } from "@/lib/gates";
import { useLinkedSelection } from "./useLinkedSelection";

beforeEach(() => {
  useSelectionStore.setState({ selectedCellIndices: null, selectionFilter: null, highlightedGroup: null });
});

describe("useLinkedSelection", () => {
  it("reads the store's selection, with one shared mask per selection", () => {
    const { result, rerender } = renderHook(() => useLinkedSelection(5));
    expect(result.current.selected).toBeNull();
    expect(result.current.mask).toBeNull();
    act(() => useSelectionStore.getState().setSelection(new Set([0, 3])));
    expect(result.current.size).toBe(2);
    expect([...result.current.mask!]).toEqual([1, 0, 0, 1, 0]);
    const mask = result.current.mask;
    rerender();
    expect(result.current.mask).toBe(mask);
    // Another panel asking for the same n gets the same mask.
    const other = renderHook(() => useLinkedSelection(5));
    expect(other.result.current.mask).toBe(mask);
  });

  it("treats an empty selection as none", () => {
    useSelectionStore.setState({ selectedCellIndices: new Set() });
    const { result } = renderHook(() => useLinkedSelection(3));
    expect(result.current.selected).toBeNull();
    expect(result.current.size).toBe(0);
  });

  it("selects a set of cells (dropping a highlighted group), and clears", () => {
    useSelectionStore.setState({ highlightedGroup: { column: "leiden", value: "3" } });
    const { result } = renderHook(() => useLinkedSelection());
    act(() => result.current.select(new Set([4, 2])));
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([4, 2]);
    expect(useSelectionStore.getState().highlightedGroup).toBeNull();
    act(() => result.current.select(new Set()));
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
    act(() => result.current.select(new Set([1])));
    act(() => result.current.clear());
    expect(result.current.selected).toBeNull();
  });

  it("follows a gate made the selection", () => {
    const { result } = renderHook(() => useLinkedSelection(10));
    const bits = bitsetFromIndices([1, 5, 9], 10);
    const gate: Gate = { id: "g1", name: "CD8", parentId: null, def: { kind: "cells", selection: bitsetToSelection(bits) }, createdAt: "" };
    const ev = { status: "ok", cells: bits, n: 3, parentN: 10, pct: 30 } as unknown as GateEval;
    act(() => selectGate("ds1", gate, ev));
    expect(result.current.size).toBe(3);
    expect([...result.current.mask!].reduce((a, b) => a + b, 0)).toBe(3);
  });
});
