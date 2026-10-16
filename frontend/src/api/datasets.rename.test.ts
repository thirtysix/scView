import { describe, it, expect, vi, afterEach } from "vitest";
import { renameDataset } from "./datasets";

afterEach(() => vi.unstubAllGlobals());

describe("renameDataset", () => {
  it("PATCHes the new name", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ id: "a", name: "New" })));
    vi.stubGlobal("fetch", fetchMock);
    await expect(renameDataset("a", "New")).resolves.toEqual({ id: "a", name: "New" });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/v1/datasets/a");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({ name: "New" });
  });

  it("throws the server's reason, not the raw response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Name must not be empty." }), { status: 400 })),
    );
    await expect(renameDataset("a", " ")).rejects.toThrow(/^Name must not be empty\.$/);
  });
});
