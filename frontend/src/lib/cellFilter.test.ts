import { describe, it, expect } from "vitest";
import {
  decodeMask,
  describeFilter,
  parseCellFilter,
  pruneFilter,
  type CellFilter,
} from "./cellFilter";

const filter = (items: CellFilter["items"], op: CellFilter["op"] = "and"): CellFilter => ({ op, items, layer: "" });

describe("describeFilter", () => {
  it("reads as a sentence, as in the brief", () => {
    expect(
      describeFilter(
        filter([
          { kind: "gene", gene: "CD14", op: ">", value: 1 },
          { kind: "category", column: "stim", values: ["STIM"] },
        ]),
      ),
    ).toBe("CD14 > 1 AND stim = STIM");
  });

  it("brackets a group among other items, lists sets and writes ranges", () => {
    expect(
      describeFilter(
        filter(
          [
            { kind: "range", column: "n_counts", min: 1000, max: null },
            {
              kind: "group",
              op: "and",
              conditions: [
                { kind: "category", column: "cell_type", values: ["B", "NK"] },
                { kind: "gene", gene: "MS4A1", op: "<=", value: 0.25 },
              ],
            },
          ],
          "or",
        ),
      ),
    ).toBe("n_counts ≥ 1000 OR (cell_type in {B, NK} AND MS4A1 ≤ 0.25)");
    expect(describeFilter(filter([{ kind: "range", column: "pct", min: 0.5, max: 2 }]))).toBe("0.5 ≤ pct ≤ 2");
    expect(describeFilter(filter([{ kind: "range", column: "pct", min: null, max: 2 }]))).toBe("pct ≤ 2");
  });

  it("leaves a lone group unbracketed and uses the pretty clustering name", () => {
    expect(
      describeFilter(
        filter([
          {
            kind: "group",
            op: "or",
            conditions: [
              { kind: "category", column: "scview_leiden_r0.5", values: ["3"] },
              { kind: "gene", gene: "CD3E", op: ">=", value: 2 },
            ],
          },
        ]),
      ),
    ).toBe("Leiden (res 0.5) = 3 OR CD3E ≥ 2");
  });
});

describe("pruneFilter", () => {
  it("drops incomplete conditions and emptied groups, keeps the rest", () => {
    const f = filter([
      { kind: "gene", gene: "", op: ">", value: 1 },
      { kind: "gene", gene: "CD14", op: ">", value: null },
      { kind: "category", column: "stim", values: [] },
      { kind: "range", column: "n", min: 5, max: 1 },
      { kind: "group", op: "or", conditions: [{ kind: "range", column: "n", min: null, max: null }] },
      { kind: "group", op: "or", conditions: [{ kind: "gene", gene: "LYZ", op: "<", value: 0 }] },
    ]);
    expect(pruneFilter(f)?.items).toEqual([
      { kind: "group", op: "or", conditions: [{ kind: "gene", gene: "LYZ", op: "<", value: 0 }] },
    ]);
    expect(pruneFilter(filter([{ kind: "gene", gene: "", op: ">", value: null }]))).toBeNull();
  });
});

describe("decodeMask", () => {
  it("reads bit i % 8 of byte i // 8, least significant first, below the cell count", () => {
    // bytes 0b00001001, 0b00000010 -> cells 0, 3, 9; byte 3 has bits past n
    const b64 = btoa(String.fromCharCode(0b00001001, 0b00000010, 0, 0xff));
    expect([...decodeMask(b64, 26)].sort((a, b) => a - b)).toEqual([0, 3, 9, 24, 25]);
    expect([...decodeMask(b64, 10)]).toEqual([0, 3, 9]);
  });
});

describe("parseCellFilter", () => {
  const ok = {
    op: "or",
    layer: "raw",
    items: [
      { kind: "gene", gene: "CD14", op: ">=", value: 1.5 },
      { kind: "group", op: "and", conditions: [{ kind: "range", column: "n", min: 1, max: null }] },
    ],
  };

  it("accepts a well-formed filter, layer defaulting to the dataset's", () => {
    expect(parseCellFilter(ok)).toEqual(ok);
    expect(parseCellFilter({ ...ok, layer: undefined }).layer).toBe("");
  });

  it.each([
    ["not an object", 3, /filter must be an object/],
    ["no items", { ...ok, items: [] }, /items must hold/],
    ["bad op", { ...ok, op: "xor" }, /op must be/],
    ["bad kind", { ...ok, items: [{ kind: "lasso" }] }, /kind must be/],
    ["text value", { ...ok, items: [{ kind: "gene", gene: "CD14", op: ">", value: "1" }] }, /value must be a number/],
    ["Infinity", { ...ok, items: [{ kind: "gene", gene: "CD14", op: ">", value: Infinity }] }, /value must be a number/],
    ["bad gene op", { ...ok, items: [{ kind: "gene", gene: "CD14", op: "==", value: 1 }] }, /op must be one of/],
    ["empty gene", { ...ok, items: [{ kind: "gene", gene: "", op: ">", value: 1 }] }, /gene must be/],
    ["values not a list", { ...ok, items: [{ kind: "category", column: "s", values: "A" }] }, /values must list/],
    ["number value", { ...ok, items: [{ kind: "category", column: "s", values: [1] }] }, /values\[0\]/],
    ["open range", { ...ok, items: [{ kind: "range", column: "n" }] }, /needs a minimum/],
    [
      "nested group",
      { ...ok, items: [{ kind: "group", op: "and", conditions: [{ kind: "group", op: "or", conditions: [] }] }] },
      /kind must be/,
    ],
    [
      "too many conditions",
      {
        ...ok,
        items: Array.from({ length: 3 }, () => ({
          kind: "group",
          op: "and",
          conditions: Array.from({ length: 20 }, () => ({ kind: "category", column: "s", values: ["A"] })),
        })),
      },
      /more than 40/,
    ],
  ])("rejects %s", (_name, input, message) => {
    expect(() => parseCellFilter(input)).toThrow(message);
  });
});
