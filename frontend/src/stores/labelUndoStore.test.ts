import { describe, it, expect, beforeEach } from "vitest";
import { latestUndo, MAX_UNDO, useLabelUndoStore, type LabelUndoEntry } from "./labelUndoStore";

const entry = (datasetId: string, label: string): LabelUndoEntry => ({
  datasetId,
  column: "manual_labels",
  label,
  nCells: 2,
  created: false,
  previous: [{ value: "Unassigned", indices: [0, 1] }],
  datasetCells: 100,
});

describe("labelUndoStore", () => {
  beforeEach(() => useLabelUndoStore.setState({ stacks: {}, lastColumn: {} }));

  it("undoes the latest labelling of each dataset first", () => {
    const s = useLabelUndoStore.getState();
    s.push(entry("a", "first"));
    s.push(entry("b", "other"));
    s.push(entry("a", "second"));
    expect(latestUndo(useLabelUndoStore.getState().stacks, "a")?.label).toBe("second");
    expect(useLabelUndoStore.getState().pop("a")?.label).toBe("second");
    expect(useLabelUndoStore.getState().pop("a")?.label).toBe("first");
    expect(useLabelUndoStore.getState().pop("a")).toBeUndefined();
    expect(latestUndo(useLabelUndoStore.getState().stacks, "b")?.label).toBe("other");
    expect(latestUndo(useLabelUndoStore.getState().stacks, null)).toBeUndefined();
  });

  it("remembers the column last labelled into and caps the depth", () => {
    const s = useLabelUndoStore.getState();
    for (let i = 0; i < MAX_UNDO + 5; i++) s.push(entry("a", `l${i}`));
    const after = useLabelUndoStore.getState();
    expect(after.stacks.a).toHaveLength(MAX_UNDO);
    expect(after.stacks.a![0]!.label).toBe("l5");
    expect(after.lastColumn.a).toBe("manual_labels");
    after.clear("a");
    expect(latestUndo(useLabelUndoStore.getState().stacks, "a")).toBeUndefined();
  });
});
