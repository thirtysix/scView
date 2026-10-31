import { describe, expect, it } from "vitest";
import {
  buildMesh,
  hideCovered,
  meshColors,
  outlineNote,
  parseOutlines,
  planOutlines,
  triangulateRing,
  viewRect,
  type CellOutlines,
} from "./cellOutlines";

/** The little-endian wire format of GET .../spatial/outlines. */
function wire(offsets: number[], xy: number[]): ArrayBuffer {
  const out = new ArrayBuffer(8 + 4 * offsets.length + 4 * xy.length);
  new Int32Array(out, 0, 2).set([offsets.length - 1, xy.length / 2]);
  new Int32Array(out, 8, offsets.length).set(offsets);
  new Float32Array(out, 8 + 4 * offsets.length, xy.length).set(xy);
  return out;
}

function outlines(rings: number[][]): CellOutlines {
  const offsets = [0];
  for (const r of rings) offsets.push(offsets[offsets.length - 1]! + r.length / 2);
  return parseOutlines(wire(offsets, rings.flat()));
}

function triangleArea(xy: Float32Array, tri: Uint32Array, n: number): number {
  let total = 0;
  for (let i = 0; i < n; i += 3) {
    const [a, b, c] = [tri[i]!, tri[i + 1]!, tri[i + 2]!];
    total += Math.abs(
      (xy[b * 2]! - xy[a * 2]!) * (xy[c * 2 + 1]! - xy[a * 2 + 1]!) - (xy[c * 2]! - xy[a * 2]!) * (xy[b * 2 + 1]! - xy[a * 2 + 1]!),
    ) / 2;
  }
  return total;
}

describe("the wire format", () => {
  it("reads offsets and vertices without a copy, and refuses a short answer", () => {
    const buf = wire([0, 3, 3, 7], [0, 0, 1, 0, 1, 1, 5, 5, 6, 5, 6, 6, 5, 6]);
    const o = parseOutlines(buf);
    expect(o.nCells).toBe(3);
    expect([...o.offsets]).toEqual([0, 3, 3, 7]);
    expect(o.xy.buffer).toBe(buf);
    expect(() => parseOutlines(buf.slice(0, buf.byteLength - 4))).toThrow(/announces/);
  });
});

describe("the view", () => {
  it("is the camera's world rectangle", () => {
    const r = viewRect({ target: [100, -50, 0], zoom: 1 }, 400, 200)!;
    expect([r.x0, r.x1, r.y0, r.y1, r.scale]).toEqual([0, 200, -100, 0, 2]);
    expect(viewRect({ zoom: 1 }, 400, 200)).toBeNull();
    expect(viewRect({ target: [0, 0], zoom: 1 }, 0, 200)).toBeNull();
  });
});

describe("triangulating a ring", () => {
  const tri = (ring: number[]) => {
    const xy = Float32Array.from(ring);
    const n = ring.length / 2;
    const out = new Uint32Array((n - 2) * 3);
    const end = triangulateRing(xy, 0, n, out, 0);
    return { xy, out, end };
  };

  it("fans a convex ring in either winding", () => {
    const ccw = tri([0, 0, 4, 0, 4, 4, 0, 4]);
    expect(ccw.end).toBe(6);
    expect(triangleArea(ccw.xy, ccw.out, 6)).toBeCloseTo(16);
    const cw = tri([0, 4, 4, 4, 4, 0, 0, 0]);
    expect(triangleArea(cw.xy, cw.out, 6)).toBeCloseTo(16);
  });

  it("clips ears off a concave ring, staying inside it", () => {
    // An L of area 12, then a U of area 8 x 4 - 4 x 2 = 24: a fan from the first vertex would overlap.
    const l = tri([0, 0, 4, 0, 4, 2, 2, 2, 2, 4, 0, 4]);
    expect(l.end).toBe(12);
    expect(triangleArea(l.xy, l.out, 12)).toBeCloseTo(12);
    const u = tri([0, 0, 8, 0, 8, 4, 6, 4, 6, 2, 2, 2, 2, 4, 0, 4]);
    expect(triangleArea(u.xy, u.out, 18)).toBeCloseTo(24);
    const clockwise = tri([0, 4, 2, 4, 2, 2, 6, 2, 6, 4, 8, 4, 8, 0, 0, 0]);
    expect(triangleArea(clockwise.xy, clockwise.out, 18)).toBeCloseTo(24);
  });

  it("always gives n - 2 triangles, even for a ring that crosses itself", () => {
    const bowtie = tri([0, 0, 4, 4, 4, 0, 0, 4, -2, 2]);
    expect(bowtie.end).toBe(9);
    expect([...bowtie.out].every((v) => v < 5)).toBe(true);
  });

  it("offsets the indices by where the ring starts in the mesh", () => {
    const xy = Float32Array.from([9, 9, 9, 9, 0, 0, 1, 0, 1, 1]);
    const out = new Uint32Array(3);
    expect(triangulateRing(xy, 2, 3, out, 0)).toBe(3);
    expect([...out].sort()).toEqual([2, 3, 4]);
  });
});

describe("the mesh", () => {
  const o = outlines([
    [0, 0, 2, 0, 2, 2, 0, 2],
    [],
    [10, 10, 12, 10, 11, 12],
  ]);

  it("is in the tissue world, one polygon per cell asked for, with edges closing each ring", () => {
    const m = buildMesh(o, Int32Array.from([2, 0]));
    expect([...m.cells]).toEqual([2, 0]);
    expect([...m.startIndices]).toEqual([0, 3, 7]);
    expect([...m.positions.slice(0, 6)]).toEqual([10, -10, 12, -10, 11, -12]);
    expect(m.indices.length).toBe(3 + 6);
    expect([...m.edges.slice(8, 12)]).toEqual([11, -12, 10, -10]); // the third edge goes back to the first vertex
    expect([...m.covered]).toEqual([1, 0, 1]);
    expect(m.meanSize).toBeCloseTo(2);
  });

  it("colours each vertex as its cell's point, and hides the points under outlines", () => {
    const m = buildMesh(o, Int32Array.from([2]));
    const cellColors = Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect([...meshColors(m, cellColors)]).toEqual([9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12]);
    // Slots in draw order: slot 0 draws cell 2, slot 1 draws cell 0.
    const render = Uint8Array.from([9, 10, 11, 12, 1, 2, 3, 4]);
    const shown = hideCovered(render, [2, 0], m.covered);
    expect([...shown]).toEqual([9, 10, 11, 0, 1, 2, 3, 4]);
    expect(render[3]).toBe(12); // the colours everyone else uses stay
  });
});

describe("what a view draws", () => {
  // A column of 10 unit squares around (0, -i) in the tissue world.
  const n = 10;
  const o = outlines(Array.from({ length: n }, (_, i) => [-1, i - 1, 1, i - 1, 1, i + 1, -1, i + 1]));
  const positions = Float32Array.from(Array.from({ length: n }, (_, i) => [0, -i]).flat());
  const rect = (y0: number, y1: number) => ({ x0: -5, x1: 5, y0, y1, scale: 10 });

  it("stays points past the limit and says how many are in view", () => {
    const plan = planOutlines(o, positions, rect(-9, 0), 5);
    expect(plan).toEqual({ inView: 10, mesh: null, limit: 5 });
    expect(outlineNote(plan, "cell")).toBe("10 in view: zoom in to under 5 for the cell outlines.");
  });

  it("draws those in view with a margin for a small pan, and says so", () => {
    const plan = planOutlines(o, positions, rect(-4, 0), 7); // 5 in view, + 25% of 4 either side
    expect(plan.inView).toBe(5);
    expect([...plan.mesh!.cells]).toEqual([0, 1, 2, 3, 4, 5]);
    expect(outlineNote(plan, "nucleus")).toBe("Drawing the nucleus outlines in view (5).");
  });

  it("drops the margin when it would pass the limit", () => {
    const plan = planOutlines(o, positions, rect(-4, 0), 5);
    expect([...plan.mesh!.cells]).toEqual([0, 1, 2, 3, 4]);
  });

  it("counts vertices too, for a segmentation with many a cell", () => {
    const over = planOutlines(o, positions, rect(-4, 0), 100, 19);
    expect(over.mesh).toBeNull();
    expect(over.limit).toBe(4); // 4 vertices a cell: 19 hold 4 cells
    expect(planOutlines(o, positions, rect(-4, 0), 100, 20).mesh).not.toBeNull();
  });

  it("leaves cells with no shape and other sections (NaN) as they are", () => {
    const some = outlines([[0, 0, 1, 0, 1, 1], [], [0, 0, 1, 0], [0, 0, 1, 0, 1, 1]]);
    const at = Float32Array.from([0, 0, 0, 0, 0, 0, NaN, NaN]);
    const plan = planOutlines(some, at, rect(-1, 1));
    expect(plan.inView).toBe(1);
    expect([...plan.mesh!.cells]).toEqual([0]);
  });
});
