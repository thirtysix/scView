import { describe, it, expect, vi, afterEach } from "vitest";
import { cellsMatching, filterCells } from "./cells";
import type { CellFilter } from "@/lib/cellFilter";

afterEach(() => vi.unstubAllGlobals());

const filter: CellFilter = { op: "and", layer: "", items: [{ kind: "category", column: "stim", values: ["STIM"] }] };

describe("filterCells", () => {
  it("POSTs the filter and returns the matching cells' indices", async () => {
    const mask = btoa(String.fromCharCode(0b10000001, 0b1));
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify({ n_cells: 9, n_matching: 3, mask })));
    vi.stubGlobal("fetch", fetchMock);
    const cells = await cellsMatching("ds", filter);
    expect([...cells].sort((a, b) => a - b)).toEqual([0, 7, 8]);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/v1/datasets/ds/cells/filter");
    expect(JSON.parse(init.body)).toEqual({ filter, mask: true });
  });

  it("throws the server's reason, also for a validation error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Column 'x' not found." }), { status: 404 })),
    );
    await expect(filterCells("ds", filter, false)).rejects.toThrow(/^Column 'x' not found\.$/);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ detail: [{ msg: "Value error, A range needs a minimum, a maximum, or both." }] }),
          { status: 422 },
        ),
      ),
    );
    await expect(filterCells("ds", filter, false)).rejects.toThrow(/^A range needs a minimum/);
  });
});
